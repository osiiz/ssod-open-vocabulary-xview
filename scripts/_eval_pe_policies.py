"""Helper script: evalua precision/recall per-class de varias políticas PE.

Uso interno desde build_combined_pe.py o ejecución directa.
"""
import json
import contextlib
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def iou_xywh(box, gts):
    bx, by, bw, bh = box
    gx, gy, gw, gh = gts.T
    ix1 = np.maximum(bx, gx); iy1 = np.maximum(by, gy)
    ix2 = np.minimum(bx + bw, gx + gw); iy2 = np.minimum(by + bh, gy + gh)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = bw * bh + gw * gh - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def evaluate_policy(name, det_path, coco, cat_id_to_name, gt_by, gt_total):
    with open(det_path) as f:
        dets = json.load(f)
    by_img_cat = defaultdict(list)
    for i, d in enumerate(dets):
        by_img_cat[(d["image_id"], d["category_id"])].append(i)
    is_tp = np.zeros(len(dets), dtype=bool)
    for (img_id, cid), idxs in by_img_cat.items():
        if cid < 0:
            continue
        gts = gt_by.get((img_id, cid))
        if gts is None:
            continue
        scores = np.array([dets[i]["score"] for i in idxs])
        order = np.argsort(-scores)
        used = np.zeros(gts.shape[0], dtype=bool)
        for k in order:
            i = idxs[k]
            ious = iou_xywh(np.asarray(dets[i]["bbox"], dtype=float), gts)
            ious_m = np.where(used, -1, ious)
            best = int(np.argmax(ious_m))
            if ious_m[best] >= 0.5:
                is_tp[i] = True
                used[best] = True

    print(f"\n=== {name.upper()} ({len(dets):,} dets) ===")
    head = f"{'Clase':<22} {'N':>8} {'TP':>6} {'Prec':>7} {'Recall':>7} {'GT':>9}"
    print(head)
    print("-" * len(head))
    per_class = {}
    tp_total = fp_total = 0
    for cid, name_c in cat_id_to_name.items():
        idxs_c = [i for i in range(len(dets)) if dets[i]["category_id"] == cid]
        n = len(idxs_c)
        tp = int(sum(is_tp[i] for i in idxs_c))
        fp = n - tp
        tp_total += tp
        fp_total += fp
        prec = tp / n if n else float("nan")
        rec = tp / gt_total[cid] if gt_total[cid] else float("nan")
        per_class[name_c] = {"n": n, "tp": tp, "prec": prec, "recall": rec, "gt": gt_total[cid]}
        print(f"{name_c:<22} {n:>8,} {tp:>6,} {prec:>7.3f} {rec:>7.3f} {gt_total[cid]:>9,}")
    tot_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    gt_sum = sum(gt_total.values())
    glob_recall = tp_total / gt_sum
    print(f"{'TOTAL':<22} {len(dets):>8,} {tp_total:>6,} {tot_prec:>7.3f} {glob_recall:>7.3f} {gt_sum:>9,}")
    return {
        "name": name,
        "n": len(dets),
        "tp": tp_total,
        "prec": tot_prec,
        "recall": glob_recall,
        "per_class": per_class,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", required=True,
                        help="Pares name=path/to/detection_results.json")
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.ann_file))
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    gt_by = {}
    for img_id in coco.getImgIds():
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            gt_by.setdefault((img_id, a["category_id"]), []).append(a["bbox"])
    gt_by = {k: np.array(v, dtype=float) for k, v in gt_by.items()}
    gt_total = defaultdict(int)
    for (_, cid), boxes in gt_by.items():
        gt_total[cid] += len(boxes)

    results = []
    for spec in args.policies:
        name, path = spec.split("=", 1)
        r = evaluate_policy(name, path, coco, cat_id_to_name, gt_by, gt_total)
        results.append(r)

    print("\n\n=== GLOBAL SUMMARY ===")
    head = f"{'Policy':<25} {'N PE':>10} {'TP':>10} {'Precision':>10} {'Recall':>10}"
    print(head)
    print("-" * len(head))
    for m in results:
        print(f"{m['name']:<25} {m['n']:>10,} {m['tp']:>10,} {m['prec']:>10.4f} {m['recall']:>10.4f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
