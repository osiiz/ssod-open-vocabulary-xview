import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from tqdm import tqdm

from src.utils.coco_utils import build_coco_max_dets, torch2coco_results
from src.utils.import_config import import_py_config
from src.utils.torvis_utils import load_model


def collate_function(batch):
    return tuple(zip(*batch))


def draw_boxes(image, gt_annotations, pred_annotations, category_map, vis_score_thresh):
    for ann in gt_annotations:
        x, y, w, h = ann["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cat_name = category_map.get(ann["category_id"], str(ann["category_id"]))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (text_w, text_h), _ = cv2.getTextSize(
            cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 0, 0), -1)
        cv2.putText(
            image,
            cat_name,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    for ann in pred_annotations:
        if ann["score"] < vis_score_thresh:
            continue

        x, y, w, h = ann["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cat_name = category_map.get(ann["category_id"], str(ann["category_id"]))
        label = f"{cat_name} {ann['score']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )


def run_inference(
    model,
    preprocess,
    categories,
    img_dir,
    ann_file,
    score_thresh,
    batch_size,
    num_workers,
    num_imgs,
    device,
):
    dataset = CocoDetection(root=img_dir, annFile=ann_file, transform=preprocess)

    if num_imgs is not None:
        num_imgs = min(num_imgs, len(dataset))
        subset_indices = list(range(num_imgs))
        dataset_view = Subset(dataset, subset_indices)
    else:
        subset_indices = list(range(len(dataset)))
        dataset_view = dataset

    dataloader = DataLoader(
        dataset_view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_function,
    )

    detections = []
    processed = 0
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Inference", leave=False):
            images = [img.to(device) for img in images]
            predictions = model(images)

            for prediction in predictions:
                original_idx = subset_indices[processed]
                coco_img_id = dataset.ids[original_idx]
                processed += 1

                detections.extend(
                    torch2coco_results(
                        prediction,
                        coco_img_id,
                        categories,
                        score_thresh=score_thresh,
                    )
                )

    eval_img_ids = [dataset.ids[i] for i in subset_indices]
    return detections, eval_img_ids


def get_eval_img_ids(coco_gt, num_imgs=None):
    img_ids = coco_gt.getImgIds()
    if num_imgs is not None:
        img_ids = img_ids[: min(num_imgs, len(img_ids))]
    return img_ids


def resolve_predictions_path(predictions_json=None, predictions_source="auto"):
    if predictions_json is not None:
        return Path(predictions_json)

    if predictions_source == "none":
        return None

    default_paths = {
        "inference_test": Path("./results/inference_test/detection_results.json"),
        "inference_test_defaults": Path(
            "./results/inference_test_defaults/detection_results.json"
        ),
    }

    if predictions_source in default_paths:
        return default_paths[predictions_source]

    # auto mode: prefer standard inference outputs if available
    for candidate in [
        default_paths["inference_test"],
        default_paths["inference_test_defaults"],
    ]:
        if candidate.is_file():
            return candidate

    return None


def compute_ap50_per_image(
    coco_gt,
    coco_dt,
    eval_img_ids,
    include_empty_images=False,
    max_dets=(100, 500, 1000),
):
    per_image_metrics = []

    for img_id in tqdm(eval_img_ids, desc="Per-image AP50", leave=False):
        gt_annotations = coco_gt.imgToAnns.get(img_id, [])
        num_gt = len(gt_annotations)

        if not include_empty_images and num_gt == 0:
            continue

        if coco_dt is not None:
            pred_annotations = coco_dt.imgToAnns.get(img_id, [])
            num_pred = len(pred_annotations)
        else:
            num_pred = 0

        ap50 = 0.0
        if coco_dt is not None:
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.imgIds = [img_id]
            coco_eval.params.maxDets = list(max_dets)

            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()

            precision = coco_eval.eval.get("precision", None)
            if precision is not None and precision.size > 0:
                p50 = precision[0, :, :, 0, -1]
                p50 = p50[p50 > -1]
                if p50.size > 0:
                    ap50 = float(np.mean(p50))

        image_info = coco_gt.imgs[img_id]
        per_image_metrics.append(
            {
                "image_id": int(img_id),
                "file_name": image_info["file_name"],
                "ap50": round(ap50, 6),
                "num_gt": int(num_gt),
                "num_pred": int(num_pred),
            }
        )

    return per_image_metrics


def get_config_detections_per_img(model_config_file, default=1000):
    if model_config_file is None:
        return default

    try:
        cfg = import_py_config(model_config_file)
        return int(getattr(cfg, "box_detections_per_img", default))
    except Exception:
        return default


def get_detected_max_per_image(detections, default=1000):
    if not detections:
        return default

    counts = defaultdict(int)
    for det in detections:
        img_id = det.get("image_id")
        if img_id is not None:
            counts[int(img_id)] += 1

    return max(counts.values(), default=default)


def save_worst_visualizations(
    coco_gt,
    detections,
    worst_images,
    img_dir,
    output_dir,
    vis_score_thresh,
):
    category_map = {
        cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())
    }

    preds_by_image = defaultdict(list)
    for det in detections:
        preds_by_image[det["image_id"]].append(det)

    vis_dir = output_dir / "worst_images"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for rank, item in enumerate(worst_images, start=1):
        img_id = item["image_id"]
        file_name = item["file_name"]
        img_path = Path(img_dir) / file_name

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Advertencia: no se pudo cargar {img_path}")
            continue

        gt_annotations = coco_gt.imgToAnns.get(img_id, [])
        pred_annotations = preds_by_image.get(img_id, [])
        pred_annotations = sorted(
            pred_annotations, key=lambda x: x["score"], reverse=True
        )

        draw_boxes(
            image, gt_annotations, pred_annotations, category_map, vis_score_thresh
        )

        summary = (
            f"rank={rank} ap50={item['ap50']:.4f} "
            f"gt={item['num_gt']} pred={item['num_pred']}"
        )
        cv2.putText(
            image,
            summary,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        out_name = f"rank_{rank:02d}_ap50_{item['ap50']:.4f}_{Path(file_name).stem}.png"
        out_path = vis_dir / out_name
        cv2.imwrite(str(out_path), image)


def evaluate_per_image_performance(
    model_config_file,
    img_dir,
    ann_file,
    model_checkpoint=None,
    predictions_json=None,
    predictions_source="auto",
    output_dir="./results/per_image_eval",
    num_imgs=None,
    score_thresh=0.001,
    vis_score_thresh=0.3,
    device_name="cuda:0",
    batch_size=1,
    num_workers=4,
    k=10,
    include_empty_images=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_path = output_dir / "detection_results.json"
    metrics_path = output_dir / "per_image_metrics.json"
    worst_path = output_dir / "worst_k_images.json"

    coco_gt = COCO(ann_file)
    eval_img_ids = get_eval_img_ids(coco_gt, num_imgs=num_imgs)
    config_max_dets = get_config_detections_per_img(model_config_file)
    coco_max_dets = build_coco_max_dets(config_max_dets)

    detections_input_path = resolve_predictions_path(
        predictions_json=predictions_json,
        predictions_source=predictions_source,
    )

    if detections_input_path is not None:
        if not detections_input_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo de predicciones: {detections_input_path}"
            )

        with open(detections_input_path, "r") as f:
            detections = json.load(f)

        eval_img_ids_set = set(eval_img_ids)
        detections = [
            det
            for det in detections
            if int(det.get("image_id", -1)) in eval_img_ids_set
        ]
        if model_config_file is None:
            detected_max = get_detected_max_per_image(detections, default=1000)
            coco_max_dets = build_coco_max_dets(detected_max)
        print(f"Predicciones cargadas desde: {detections_input_path}")
    else:
        if model_config_file is None:
            raise ValueError(
                "Debes proporcionar --model_config_file cuando no se usa "
                "--predictions_json ni --predictions_source con archivo disponible."
            )

        model, preprocess, categories = load_model(
            model_config_file,
            model_checkpoint_override=model_checkpoint,
        )
        detections_per_img = int(
            getattr(model.roi_heads, "detections_per_img", config_max_dets)
        )
        coco_max_dets = build_coco_max_dets(detections_per_img)

        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        detections, eval_img_ids = run_inference(
            model=model,
            preprocess=preprocess,
            categories=categories,
            img_dir=img_dir,
            ann_file=ann_file,
            score_thresh=score_thresh,
            batch_size=batch_size,
            num_workers=num_workers,
            num_imgs=num_imgs,
            device=device,
        )

    with open(detections_path, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"Detecciones guardadas en: {detections_path}")

    coco_dt = coco_gt.loadRes(str(detections_path)) if len(detections) > 0 else None

    per_image_metrics = compute_ap50_per_image(
        coco_gt,
        coco_dt,
        eval_img_ids,
        include_empty_images=include_empty_images,
        max_dets=coco_max_dets,
    )

    per_image_metrics = sorted(
        per_image_metrics,
        key=lambda x: (x["ap50"], -x["num_gt"], x["image_id"]),
    )

    with open(metrics_path, "w") as f:
        json.dump(per_image_metrics, f, indent=2)

    worst_k = per_image_metrics[: min(k, len(per_image_metrics))]
    with open(worst_path, "w") as f:
        json.dump(worst_k, f, indent=2)

    save_worst_visualizations(
        coco_gt,
        detections,
        worst_k,
        img_dir,
        output_dir,
        vis_score_thresh,
    )

    print(f"Metricas por imagen guardadas en: {metrics_path}")
    print(
        f"Top-{len(worst_k)} peores imagenes guardadas en: {output_dir / 'worst_images'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera (o reutiliza) detecciones, calcula AP@0.50 por imagen y "
            "guarda las K imágenes con peor desempeño para inspección visual."
        )
    )
    parser.add_argument("--model_config_file", type=str, default=None)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument(
        "--predictions_json",
        type=str,
        default=None,
        help="Ruta a un detection_results.json ya generado.",
    )
    parser.add_argument(
        "--predictions_source",
        type=str,
        choices=["auto", "inference_test", "inference_test_defaults", "none"],
        default="auto",
        help=(
            "Fuente predeterminada de predicciones cuando no se pasa --predictions_json. "
            "auto intenta results/inference_test y luego results/inference_test_defaults."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./results/per_image_eval")
    parser.add_argument("--num_imgs", type=int, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.001)
    parser.add_argument("--vis_score_thresh", type=float, default=0.3)
    parser.add_argument("--device_name", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--include_empty_images", action="store_true")

    args = parser.parse_args()

    evaluate_per_image_performance(
        model_config_file=args.model_config_file,
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        model_checkpoint=args.model_checkpoint,
        predictions_json=args.predictions_json,
        predictions_source=args.predictions_source,
        output_dir=args.output_dir,
        num_imgs=args.num_imgs,
        score_thresh=args.score_thresh,
        vis_score_thresh=args.vis_score_thresh,
        device_name=args.device_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        k=args.k,
        include_empty_images=args.include_empty_images,
    )


if __name__ == "__main__":
    main()
