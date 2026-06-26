#!/usr/bin/env python3
"""Construye PE a partir dun único prompt set (sen prompt-ensembling).

Toma as deteccións brutas dun só prompt set (p. ex. set1_direct, cos nomes das
9 macro-clases) e selecciona, por clase, o mesmo número de PE que a política
ensemble de referencia (--ref_policy), para illar a contribución do ensembling.

  - mode=score  : escóllense as de maior score por clase (Grounding DINO).
  - mode=random : escóllense ao azar por clase (Rex-Omni, cuxo score non é
                  discriminante).

As deteccións brutas teñen category_id=-1 e a macro-clase en class_name; aquí
remapéase class_name -> category_id (1..9) coas categorías do COCO de referencia.

Opcionalmente (--exclusion_output) emítense as deteccións NON seleccionadas como
zonas de exclusión (mesmo criterio que RO ensemble: ensemble menos política).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_dets(path):
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get("annotations", d.get("detections", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_single", type=Path, required=True,
                    help="Deteccións brutas. Pode ser un único prompt set ou o "
                         "ficheiro combinado (filtrado con --prompt_set).")
    ap.add_argument("--prompt_set", type=str, default=None,
                    help="Se se indica, fíltranse as deteccións ao prompt set co "
                         "campo prompt_set igual a este valor (p. ex. set1_direct).")
    ap.add_argument("--ref_policy", type=Path, required=True,
                    help="Política ensemble cuxo reparto per-clase se replica.")
    ap.add_argument("--categories", type=Path, required=True,
                    help="COCO annotations co mapa de categorías (name->id).")
    ap.add_argument("--mode", choices=["score", "random"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exclusion_output", type=Path, default=None,
                    help="Se se indica, escríbense as deteccións non seleccionadas como zonas.")
    args = ap.parse_args()

    name2id = {c["name"]: c["id"] for c in json.load(open(args.categories))["categories"]}

    raw = load_dets(args.raw_single)
    if args.prompt_set is not None:
        raw = [x for x in raw if x.get("prompt_set") == args.prompt_set]
        print(f"Filtrado a prompt_set={args.prompt_set}: {len(raw)} deteccións.")
    dets = []
    dropped = 0
    for x in raw:
        cid = name2id.get(x.get("class_name"))
        if cid is None:
            dropped += 1
            continue
        dets.append({
            "image_id": x["image_id"],
            "category_id": cid,
            "bbox": x["bbox"],
            "score": x.get("score", 0.0),
        })

    target = Counter(d["category_id"] for d in load_dets(args.ref_policy))

    by_class = defaultdict(list)
    for i, d in enumerate(dets):
        by_class[d["category_id"]].append(i)

    rng = random.Random(args.seed)
    selected_idx = set()
    print(f"{'clase':>3}{'obxectivo':>11}{'dispoñible':>12}{'escollido':>11}")
    total_sel = total_tgt = 0
    for cid in sorted(target):
        tgt = target[cid]
        pool = by_class.get(cid, [])
        if args.mode == "score":
            order = sorted(pool, key=lambda i: -dets[i]["score"])
        else:
            order = list(pool)
            rng.shuffle(order)
        pick = order[:tgt]
        selected_idx.update(pick)
        total_sel += len(pick)
        total_tgt += tgt
        flag = "" if len(pick) >= tgt else "  <-- FALTA"
        print(f"{cid:>3}{tgt:>11}{len(pool):>12}{len(pick):>11}{flag}")
    print(f"TOTAL obxectivo={total_tgt}  escollido={total_sel}  (descartadas sen mapa: {dropped})")

    selected = [dets[i] for i in selected_idx]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(selected, open(args.output, "w"))
    print(f"PE positivas -> {args.output} ({len(selected)})")

    if args.exclusion_output is not None:
        zones = [dets[i] for i in range(len(dets)) if i not in selected_idx]
        args.exclusion_output.parent.mkdir(parents=True, exist_ok=True)
        json.dump(zones, open(args.exclusion_output, "w"))
        print(f"Zonas de exclusión -> {args.exclusion_output} ({len(zones)})")


if __name__ == "__main__":
    main()
