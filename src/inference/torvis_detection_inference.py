"""Inference script for object detection using torchvision models
on COCO dataset."""

import json
import fire
from pathlib import Path

import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Subset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.utils.torvis_utils import load_model
from src.utils.coco_utils import build_coco_max_dets, stats2dict, torch2coco_results


def collate_function(batch):
    """Custom collate function to handle batches with different
    image sizes. This is required only if preprocess doesn't include
    resizing to fixed size."""
    return tuple(zip(*batch))


def compute_class_ap50_metrics(coco_eval, coco_gt):
    import numpy as np

    precisions = coco_eval.eval["precision"]
    cat_ids = coco_eval.params.catIds
    categories_info = coco_gt.loadCats(cat_ids)
    cat_dict = {cat["id"]: cat["name"] for cat in categories_info}

    class_metrics = {}
    print("\n" + "=" * 55)
    print(" Average Precision (AP50) por Clase")
    print("=" * 55)

    for i, cat_id in enumerate(cat_ids):
        p = precisions[0, :, i, 0, -1]
        p = p[p > -1]

        cat_name = cat_dict[cat_id]
        if len(p) > 0:
            ap50 = np.mean(p)
            class_metrics[cat_name] = round(float(ap50), 4)
            print(f" {cat_name:<40} : {ap50:.4f}")
        else:
            class_metrics[cat_name] = -1.0
            print(f" {cat_name:<40} : N/A (Sin ejemplos)")

    print("=" * 55 + "\n")
    return class_metrics


def detection_inference(
    model_config_file: str,
    img_dir: str,
    ann_file: str,
    model_checkpoint: str | None = None,
    output_folder: str = "results/inference/",
    num_imgs: int | None = None,
    score_thresh: float = 0.001,
    device_name: str = "cuda:0",
    batch_size: int = 1,
    num_workers: int = 4,
) -> None:
    """Process and evaluate dataset with torchvision detection model
    and generate results in COCO format.

    Args:
        model_config_file:
            path to file with pytorch model config params.
        model_checkpoint:
            optional checkpoint path to override the model path defined in
            model_config_file. Useful to evaluate a specific trained checkpoint
            while keeping a single shared model config.
        img_dir:
            path to folder with images to process.
        ann_file:
            path to annotation file.
        output_folder:
            path to folder where result files will be created.
            Default is results/inference. Two files will be created:
            - detection_results.json: COCO format detection results
            - metrics.json: COCO evaluation metrics
        num_imgs:
            number of images to process from dataset. Default is None,
            which processes all images.
        score_thresh:
            score threshold to filter detections. Default is 0.5
        device_name:
            name of device used for inference. Default is "cuda:0"
        batch_size:
            batch size for dataloader. Default is 1
        num_workers:
            number of workers for dataloader. Default is 4
    """
    # Create output folder
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output files
    results_json = str(output_dir / "detection_results.json")
    metrics_json = str(output_dir / "metrics.json")

    # Load torchvision model, preprocess transforms and categories
    # from config file
    model, preprocess, categories = load_model(
        model_config_file,
        model_checkpoint_override=model_checkpoint,
    )

    # Move model to GPU
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Load full dataset
    coco_dataset = CocoDetection(root=img_dir, annFile=ann_file, transform=preprocess)

    # Get subset of dataset if num_imgs is specified
    if num_imgs is not None:
        coco_subset = Subset(coco_dataset, range(num_imgs))
        img_indices = coco_subset.indices
    else:
        coco_subset = coco_dataset
        img_indices = range(len(coco_dataset))

    # Build dataloader
    dataloader = DataLoader(
        coco_subset,
        batch_size=batch_size,
        collate_fn=collate_function,
        shuffle=False,  # Otherwise img ids won't match
    )
    batch_size_eff = dataloader.batch_size or 1

    # Inference loop
    coco_results = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):

            # Move images to device
            images = [img.to(device) for img in images]

            predictions = model(images)

            # Get img coco id for each image in batch
            # and convert predictions to coco format

            for j, results in enumerate(predictions):
                # Get image index in subset
                img_idx = i * batch_size_eff + j
                # Get image index in full dataset
                original_idx = img_indices[img_idx]
                # Get image COCO identifier in annotation file
                coco_img_id = coco_dataset.ids[original_idx]

                detections = torch2coco_results(
                    predictions[j],
                    coco_img_id,
                    categories if categories is not None else [],
                    score_thresh=score_thresh,
                )

                coco_results.extend(detections)

    # Save coco results to JSON file
    with open(results_json, "w") as f:
        json.dump(coco_results, f, indent=4)

    print(f"Results saved to: {results_json}")

    # Evaluate COCO metrics
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    # Restrict evaluation to subset
    coco_eval.params.imgIds = [coco_dataset.ids[i] for i in img_indices]

    detections_per_img = int(getattr(model.roi_heads, "detections_per_img", 1000))
    coco_eval.params.maxDets = build_coco_max_dets(detections_per_img)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    class_metrics_json = str(output_dir / "metrics_per_class.json")
    class_metrics = compute_class_ap50_metrics(coco_eval, coco_gt)

    with open(class_metrics_json, "w") as f:
        json.dump(class_metrics, f, indent=4)

    metrics = stats2dict(coco_eval)

    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_json}")


if __name__ == "__main__":
    fire.Fire(detection_inference)
