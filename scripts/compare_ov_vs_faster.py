"""
Experimento 2: compara las pseudoetiquetas de los detectores OV (Grounding DINO,
Rex-Omni) con las del detector base Faster10 sobre `train_unlabeled_eval`.

Mide la complementariedad entre fuentes de PE bajo dos formulaciones:

  Métrica A (precisión contra "Faster-TPs-as-GT")
    Los TPs de Faster10 contra el GT real se usan como GT artificial. Cada
    detección OV se reevalua matching greedy IoU=0.5 + clase contra ese
    conjunto. P_synth = TP_synth / N_PE. Comparada con P_real cuantifica
    qué fracción de las PE OV están descubriendo objetos que Faster falla.

  Métrica B (overlap a nivel de GT)
    Para cada GT del set unlabeled_eval comprobamos qué método lo cubre:
    Faster10 (>= threshold óptimo), DINO (política X), Rex (política X).
    Reportamos cardinales de los conjuntos cubiertos y su intersección.

Uso:
    python scripts/compare_ov_vs_faster.py \\
        --ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \\
        --faster results/ssod/faster10_unlabeled_raw/detection_results.json \\
        --faster_thresh 0.1 \\
        --dino_policy dino_optimal_high=results/pe_policies/dino_optimal_high/detection_results.json \\
        --rex_policy  rex_optimal_high=results/pe_policies/rex_optimal_high/detection_results.json \\
        --output_dir docs/results_reports/ov_vs_faster/
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def iou_xywh(box: np.ndarray, gts: np.ndarray) -> np.ndarray:
    bx, by, bw, bh = box
    gx, gy, gw, gh = gts.T
    ix1 = np.maximum(bx, gx)
    iy1 = np.maximum(by, gy)
    ix2 = np.minimum(bx + bw, gx + gw)
    iy2 = np.minimum(by + bh, gy + gh)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = bw * bh + gw * gh - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def match_to_gt(
    dets: list[dict],
    gt_by_img_cat: dict[tuple[int, int], np.ndarray],
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """
    Matching greedy detección→GT a IoU>=iou_thr, ordenado por score desc, class-aware.

    Devuelve:
      is_tp: array bool por detección (True si emparejó con un GT no usado)
      gt_covered: dict (img,cat)->bool array indicando qué GTs fueron cubiertos
    """
    by_img_cat = defaultdict(list)
    for i, d in enumerate(dets):
        by_img_cat[(d["image_id"], d["category_id"])].append(i)

    is_tp = np.zeros(len(dets), dtype=bool)
    gt_covered: dict[tuple[int, int], np.ndarray] = {}

    for (img_id, cid), idxs in by_img_cat.items():
        if cid < 0:
            continue
        gts = gt_by_img_cat.get((img_id, cid))
        if gts is None:
            continue
        scores = np.array([dets[i].get("score", 0.0) for i in idxs])
        order = np.argsort(-scores)
        used = np.zeros(gts.shape[0], dtype=bool)
        for k in order:
            i = idxs[k]
            ious = iou_xywh(np.asarray(dets[i]["bbox"], dtype=float), gts)
            ious_m = np.where(used, -1, ious)
            best = int(np.argmax(ious_m))
            if ious_m[best] >= iou_thr:
                is_tp[i] = True
                used[best] = True
        gt_covered[(img_id, cid)] = used

    return is_tp, gt_covered


def tps_as_synthetic_gt(
    dets: list[dict], is_tp: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    """
    Construye un dict {(img, cat) -> array bboxes} con las bboxes de las
    detecciones marcadas como TP. Se usa como GT artificial para Métrica A.
    """
    grouped: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for i, d in enumerate(dets):
        if is_tp[i]:
            grouped[(d["image_id"], d["category_id"])].append(d["bbox"])
    return {k: np.array(v, dtype=float) for k, v in grouped.items()}


def per_class_precision(
    dets: list[dict], is_tp: np.ndarray, cat_id_to_name: dict[int, str]
) -> tuple[dict, dict]:
    per_cls = {}
    n_tot = tp_tot = 0
    for cid, name in cat_id_to_name.items():
        idxs = [i for i in range(len(dets)) if dets[i]["category_id"] == cid]
        n = len(idxs)
        tp = int(sum(is_tp[i] for i in idxs))
        per_cls[name] = {
            "n": n,
            "tp": tp,
            "prec": (tp / n) if n else float("nan"),
        }
        n_tot += n
        tp_tot += tp
    glob = {
        "n": n_tot,
        "tp": tp_tot,
        "prec": (tp_tot / n_tot) if n_tot else 0.0,
    }
    return per_cls, glob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_file", type=Path, required=True)
    ap.add_argument("--faster", type=Path, required=True,
                    help="JSON con detecciones de Faster10 (raw, score>=0)")
    ap.add_argument("--faster_thresh", type=float, default=0.1,
                    help="Umbral óptimo aplicado a Faster10 antes del matching")
    ap.add_argument("--dino_policy", action="append", default=[],
                    help="name=path; puede repetirse")
    ap.add_argument("--rex_policy", action="append", default=[],
                    help="name=path; puede repetirse")
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.ann_file))
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # GT por (img, cat)
    gt_by: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for img_id in coco.getImgIds():
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            gt_by[(img_id, a["category_id"])].append(a["bbox"])
    gt_by = {k: np.array(v, dtype=float) for k, v in gt_by.items()}
    gt_count_per_cat = defaultdict(int)
    for (_, cid), boxes in gt_by.items():
        gt_count_per_cat[cid] += len(boxes)
    gt_total = sum(gt_count_per_cat.values())

    # --- Faster10 ---
    print(f"\n=== Cargando Faster10 ({args.faster}) ===")
    with open(args.faster) as f:
        faster_dets_raw = json.load(f)
    faster_dets = [d for d in faster_dets_raw if d.get("score", 0.0) >= args.faster_thresh]
    print(f"  raw={len(faster_dets_raw):,}, filtradas score>={args.faster_thresh}: "
          f"{len(faster_dets):,}")
    faster_is_tp, faster_gt_covered = match_to_gt(faster_dets, gt_by)
    faster_per_cls, faster_glob = per_class_precision(faster_dets, faster_is_tp, cat_id_to_name)
    print(f"  Faster10 TPs={faster_glob['tp']:,} P={faster_glob['prec']:.4f}")

    # GT artificial = bboxes de los TPs de Faster
    synth_gt = tps_as_synthetic_gt(faster_dets, faster_is_tp)
    synth_count_per_cat = defaultdict(int)
    for (_, cid), boxes in synth_gt.items():
        synth_count_per_cat[cid] += len(boxes)
    synth_total = sum(synth_count_per_cat.values())
    print(f"  GT sintético (Faster TPs) total={synth_total:,}")

    # --- Por cada política OV ---
    policies = [("dino", n_p.split("=", 1)) for n_p in args.dino_policy] + \
               [("rex",  n_p.split("=", 1)) for n_p in args.rex_policy]

    report: dict[str, dict] = {
        "faster": {
            "n": len(faster_dets),
            "tp": int(faster_is_tp.sum()),
            "prec": faster_glob["prec"],
            "thresh": args.faster_thresh,
            "per_class": faster_per_cls,
        },
        "policies": {},
    }

    for det_name, (pol_name, pol_path) in policies:
        print(f"\n=== Política {pol_name} ({det_name}) ===")
        with open(pol_path) as f:
            ov_dets = json.load(f)
        print(f"  N PE = {len(ov_dets):,}")

        # Matching contra GT real
        ov_tp_real, ov_gt_covered_real = match_to_gt(ov_dets, gt_by)
        per_cls_real, glob_real = per_class_precision(ov_dets, ov_tp_real, cat_id_to_name)
        print(f"  vs GT real      : TP={glob_real['tp']:,} P_real={glob_real['prec']:.4f}")

        # Matching contra GT sintético (Faster TPs)
        ov_tp_synth, _ = match_to_gt(ov_dets, synth_gt)
        per_cls_synth, glob_synth = per_class_precision(ov_dets, ov_tp_synth, cat_id_to_name)
        print(f"  vs Faster-TPs   : TP={glob_synth['tp']:,} P_synth={glob_synth['prec']:.4f}")

        # Métrica B (GT-side): cuántos GTs cubre cada método
        n_gt_faster = 0
        n_gt_ov = 0
        n_gt_both = 0
        n_gt_only_ov = 0
        n_gt_only_faster = 0
        per_cls_overlap = {}
        for cid, name in cat_id_to_name.items():
            covf_total = covov_total = covboth_total = 0
            for (img_id, gt_cid), gt_boxes in gt_by.items():
                if gt_cid != cid:
                    continue
                covf = faster_gt_covered.get((img_id, cid))
                covov = ov_gt_covered_real.get((img_id, cid))
                if covf is None:
                    covf = np.zeros(len(gt_boxes), dtype=bool)
                if covov is None:
                    covov = np.zeros(len(gt_boxes), dtype=bool)
                covf_total += int(covf.sum())
                covov_total += int(covov.sum())
                covboth_total += int((covf & covov).sum())
            per_cls_overlap[name] = {
                "gt": gt_count_per_cat.get(cid, 0),
                "cov_faster": covf_total,
                "cov_ov": covov_total,
                "cov_both": covboth_total,
                "only_ov": covov_total - covboth_total,
                "only_faster": covf_total - covboth_total,
            }
            n_gt_faster += covf_total
            n_gt_ov += covov_total
            n_gt_both += covboth_total
            n_gt_only_ov += (covov_total - covboth_total)
            n_gt_only_faster += (covf_total - covboth_total)
        print(f"  GTs cubiertos por Faster        : {n_gt_faster:,}")
        print(f"  GTs cubiertos por {pol_name:<20}: {n_gt_ov:,}")
        print(f"  GTs cubiertos por ambos         : {n_gt_both:,}")
        print(f"  GTs solo OV (complementario)    : {n_gt_only_ov:,}")
        print(f"  GTs solo Faster                 : {n_gt_only_faster:,}")
        # Complementariedad redefinida: |TP(OV) \ TP(Faster)| / |TP(Faster) ∪ TP(OV)|
        # |TP(F) ∪ TP(OV)| = n_gt_faster + n_gt_only_ov (porque n_gt_only_ov + n_gt_faster
        # = n_gt_only_ov + n_gt_only_faster + n_gt_both = |F ∪ OV|).
        n_gt_union = n_gt_faster + n_gt_only_ov
        if n_gt_union:
            print(f"  Fracción de OV-TPs nuevos       : "
                  f"{n_gt_only_ov / n_gt_union:.4f}  (complementariedad sobre unión)")

        report["policies"][pol_name] = {
            "detector": det_name,
            "n_pe": len(ov_dets),
            "tp_real": glob_real["tp"],
            "prec_real": glob_real["prec"],
            "tp_synth": glob_synth["tp"],
            "prec_synth": glob_synth["prec"],
            "frac_overlap_with_faster": (
                glob_synth["tp"] / glob_real["tp"] if glob_real["tp"] else 0.0
            ),
            "n_gt_faster": n_gt_faster,
            "n_gt_ov": n_gt_ov,
            "n_gt_both": n_gt_both,
            "n_gt_only_ov": n_gt_only_ov,
            "n_gt_only_faster": n_gt_only_faster,
            "n_gt_union": n_gt_union,
            "frac_ov_complementary_over_union": (
                n_gt_only_ov / n_gt_union if n_gt_union else 0.0
            ),
            "per_class_real": per_cls_real,
            "per_class_synth": per_cls_synth,
            "per_class_overlap": per_cls_overlap,
        }

    # --- Triple intersección Faster ∩ DINO ∩ Rex (al GT-level, primera política de cada) ---
    if args.dino_policy and args.rex_policy:
        d_name, d_path = args.dino_policy[0].split("=", 1)
        r_name, r_path = args.rex_policy[0].split("=", 1)
        print(f"\n=== Triple intersección (Faster ∩ {d_name} ∩ {r_name}) a nivel GT ===")
        with open(d_path) as f:
            d_dets = json.load(f)
        with open(r_path) as f:
            r_dets = json.load(f)
        _, d_cov = match_to_gt(d_dets, gt_by)
        _, r_cov = match_to_gt(r_dets, gt_by)
        n_all3 = 0
        for key, gt_boxes in gt_by.items():
            cf = faster_gt_covered.get(key, np.zeros(len(gt_boxes), dtype=bool))
            cd = d_cov.get(key, np.zeros(len(gt_boxes), dtype=bool))
            cr = r_cov.get(key, np.zeros(len(gt_boxes), dtype=bool))
            n_all3 += int((cf & cd & cr).sum())
        n_only_faster = 0
        n_only_dino = 0
        n_only_rex = 0
        for key, gt_boxes in gt_by.items():
            cf = faster_gt_covered.get(key, np.zeros(len(gt_boxes), dtype=bool))
            cd = d_cov.get(key, np.zeros(len(gt_boxes), dtype=bool))
            cr = r_cov.get(key, np.zeros(len(gt_boxes), dtype=bool))
            n_only_faster += int((cf & ~cd & ~cr).sum())
            n_only_dino   += int((~cf & cd & ~cr).sum())
            n_only_rex    += int((~cf & ~cd & cr).sum())
        report["triple_intersection"] = {
            "dino_policy": d_name,
            "rex_policy": r_name,
            "all3": n_all3,
            "only_faster": n_only_faster,
            "only_dino": n_only_dino,
            "only_rex": n_only_rex,
            "gt_total": gt_total,
        }
        print(f"  Cubiertos por los 3              : {n_all3:,}")
        print(f"  Cubiertos solo por Faster        : {n_only_faster:,}")
        print(f"  Cubiertos solo por {d_name}      : {n_only_dino:,}")
        print(f"  Cubiertos solo por {r_name}      : {n_only_rex:,}")

    # Resumen tabular
    print("\n\n=== RESUMEN ===")
    head = f"{'Policy':<25} {'N PE':>8} {'TP_real':>8} {'P_real':>7} {'P_synth':>8} {'overlap':>8} {'compl%':>7}"
    print(head)
    print("-" * len(head))
    for name, m in report["policies"].items():
        comp = m["frac_ov_complementary_over_union"] * 100
        print(f"{name:<25} {m['n_pe']:>8,} {m['tp_real']:>8,} "
              f"{m['prec_real']:>7.4f} {m['prec_synth']:>8.4f} "
              f"{m['frac_overlap_with_faster']:>8.4f} {comp:>6.1f}%")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nGuardado: {out_json}")


if __name__ == "__main__":
    main()
