import argparse
import csv
import json
from pathlib import Path

from src.inference.ov_coco_eval import evaluate_ov_predictions
from src.inference.common import str2bool
from src.inference.dino_inference import run_grounding_dino_inference


def parse_score_text_pairs(raw_value: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for token in str(raw_value).split(","):
        chunk = token.strip()
        if not chunk:
            continue

        if ":" in chunk:
            score_raw, text_raw = chunk.split(":", maxsplit=1)
        else:
            score_raw = chunk
            text_raw = chunk

        score_thresh = float(score_raw)
        text_thresh = float(text_raw)
        if not (0.0 <= score_thresh <= 1.0 and 0.0 <= text_thresh <= 1.0):
            raise ValueError(
                "score/text thresholds must be in [0,1]: "
                f"{score_thresh}:{text_thresh}"
            )
        pairs.append((score_thresh, text_thresh))

    if not pairs:
        raise ValueError("score_text_pairs cannot be empty")

    return pairs


def variant_name(score_thresh: float, text_thresh: float) -> str:
    return f"s{score_thresh:.3f}_t{text_thresh:.3f}".replace(".", "p").replace("-", "m")


def load_coco_annotations(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a COCO JSON object in {path}")

    return payload


def build_eval_subset_coco(
    coco_annotations: dict, sample_images: int
) -> tuple[dict, int, int]:
    images = list(coco_annotations.get("images", []))
    total_images = len(images)

    sampled_count = max(0, int(sample_images))
    sampled_images = images[:sampled_count]
    sampled_image_ids = {
        int(image["id"])
        for image in sampled_images
        if isinstance(image, dict) and "id" in image
    }

    sampled_annotations = [
        annotation
        for annotation in coco_annotations.get("annotations", [])
        if isinstance(annotation, dict)
        and int(annotation.get("image_id", -1)) in sampled_image_ids
    ]

    subset_coco = {
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": coco_annotations.get("categories", []),
    }

    for optional_key in ("info", "licenses"):
        if optional_key in coco_annotations:
            subset_coco[optional_key] = coco_annotations[optional_key]

    return subset_coco, len(sampled_images), total_images


def materialize_eval_ann_file(
    ann_file: Path,
    output_root: Path,
    sample_images: int | None,
) -> tuple[Path, int, int]:
    coco_annotations = load_coco_annotations(ann_file)
    total_images = len(coco_annotations.get("images", []))

    if sample_images is None or int(sample_images) >= total_images:
        return ann_file, total_images, total_images

    subset_coco, sampled_images, _ = build_eval_subset_coco(
        coco_annotations=coco_annotations,
        sample_images=sample_images,
    )

    subset_ann_file = output_root / f"_eval_subset_first_{sampled_images}.json"
    subset_ann_file.write_text(json.dumps(subset_coco, indent=2), encoding="utf-8")
    return subset_ann_file, sampled_images, total_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run threshold sweep for Grounding DINO and compare aware/agnostic metrics"
    )
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--score_text_pairs", type=str, required=True)
    parser.add_argument("--sample_images", type=int, default=600)

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--xview_classes_path", type=Path, required=True)
    parser.add_argument("--xview_macro_classes_path", type=Path, required=True)
    parser.add_argument("--prompt_file", type=Path, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--save_raw_predictions", type=str2bool, default=False)

    parser.add_argument("--max_dets", type=int, default=1000)
    parser.add_argument(
        "--artifact_mode",
        type=str,
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
    )

    args = parser.parse_args()
    pairs = parse_score_text_pairs(args.score_text_pairs)
    args.output_root.mkdir(parents=True, exist_ok=True)

    eval_ann_file, eval_images, total_images = materialize_eval_ann_file(
        ann_file=args.ann_file,
        output_root=args.output_root,
        sample_images=args.sample_images,
    )

    if eval_ann_file == args.ann_file:
        print(f"Evaluation scope: full annotation set | images={total_images}")
    else:
        print(
            "Evaluation scope: sampled subset "
            f"| images={eval_images}/{total_images} | ann_file={eval_ann_file}"
        )

    rows = []
    for idx, (score_thresh, text_thresh) in enumerate(pairs, start=1):
        run_name = variant_name(score_thresh, text_thresh)
        run_dir = args.output_root / run_name
        raw_dir = run_dir / "dino_raw"
        aware_dir = run_dir / "class-aware"
        agnostic_dir = run_dir / "class-agnostic"

        print(
            f"[{idx}/{len(pairs)}] Running thresholds "
            f"score={score_thresh} text={text_thresh} | output={run_dir}"
        )

        run_grounding_dino_inference(
            img_dir=args.img_dir,
            ann_file=args.ann_file,
            output_folder=raw_dir,
            model_id=args.model_id,
            xview_classes_path=args.xview_classes_path,
            xview_macro_classes_path=args.xview_macro_classes_path,
            prompt_file=args.prompt_file,
            score_thresh=score_thresh,
            text_thresh=text_thresh,
            device=args.device,
            batch_size=args.batch_size,
            max_images=args.sample_images,
            use_amp=args.use_amp,
            save_raw_predictions=args.save_raw_predictions,
        )

        evaluate_ov_predictions(
            ann_file=eval_ann_file,
            detection_results=raw_dir / "detection_results.json",
            output_folder=aware_dir,
            mode="aware",
            max_dets=args.max_dets,
            artifact_mode=args.artifact_mode,
            materialize_raw=False,
        )
        evaluate_ov_predictions(
            ann_file=eval_ann_file,
            detection_results=raw_dir / "detection_results.json",
            output_folder=agnostic_dir,
            mode="agnostic",
            max_dets=args.max_dets,
            artifact_mode=args.artifact_mode,
            materialize_raw=False,
        )

        aware_metrics = json.loads(
            (aware_dir / "metrics.json").read_text(encoding="utf-8")
        )
        agnostic_metrics = json.loads(
            (agnostic_dir / "metrics.json").read_text(encoding="utf-8")
        )

        row = {
            "run": run_name,
            "score_thresh": score_thresh,
            "text_thresh": text_thresh,
            "aware_AP": float(aware_metrics.get("AP", -1.0)),
            "aware_AP50": float(aware_metrics.get("AP50", -1.0)),
            "aware_AR_1000": float(aware_metrics.get("AR_1000", -1.0)),
            "agnostic_AP": float(agnostic_metrics.get("AP", -1.0)),
            "agnostic_AP50": float(agnostic_metrics.get("AP50", -1.0)),
            "agnostic_AR_1000": float(agnostic_metrics.get("AR_1000", -1.0)),
        }
        rows.append(row)

    rows.sort(
        key=lambda item: (item["aware_AP50"], item["agnostic_AP50"]), reverse=True
    )

    summary_json = args.output_root / "summary.json"
    summary_csv = args.output_root / "summary.csv"

    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print(
        "Sweep finished | "
        f"best={best['run']} aware_AP50={best['aware_AP50']:.4f} "
        f"agnostic_AP50={best['agnostic_AP50']:.4f}"
    )
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
