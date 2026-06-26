"""PoC: extraer scores de los logits de Rex-Omni.

Objetivo: validar empiricamente si los logits del modelo (al generar los tokens
de clase y coordenadas) son informativos como senal de confianza, o si saturan
a ~1.0 (modelos autorregresivos suelen ser muy confiados sobre el siguiente token
una vez fijado el contexto estructural).

Salida: para una imagen de prueba, imprime por cada deteccion:
  - clase detectada y coordenadas
  - prob maxima y entropia de los tokens de clase
  - prob maxima y entropia de los tokens de coordenadas (x0, y0, x1, y1)

Si las probabilidades estan todas cerca de 1.0 -> los scores no son utiles.
Si hay variacion significativa -> vale la pena implementar la extraccion completa.

Uso:
    python scripts/poc_rex_output_scores.py --image_path <imagen.tif> --categories aircraft plane car
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image

# Tokens estructurales de Rex-Omni
SPECIAL_TOKENS = {
    "obj_start": "<|object_ref_start|>",
    "obj_end": "<|object_ref_end|>",
    "box_start": "<|box_start|>",
    "box_end": "<|box_end|>",
}


def _entropy(probs: torch.Tensor) -> float:
    """Shannon entropy en nats sobre el vector de probabilidades."""
    p = probs.clamp(min=1e-12)
    return float(-(p * torch.log(p)).sum())


def _summarize_token_step(logits_step: torch.Tensor, generated_token_id: int) -> dict:
    """Para un paso de generacion, devuelve prob del token generado y entropia."""
    probs = torch.softmax(logits_step.float(), dim=-1)
    return {
        "prob_generated": float(probs[generated_token_id]),
        "entropy": _entropy(probs),
        "top5_probs": probs.topk(5).values.tolist(),
    }


def _is_coord_token(text: str) -> bool:
    """True si el token es <NUM>, donde NUM es un entero positivo."""
    return text.startswith("<") and text.endswith(">") and text[1:-1].isdigit()


def analyze_bifurcation_points(
    gen_ids: torch.Tensor,
    scores: tuple,
    tok,
    special_ids: dict,
) -> None:
    """
    Analiza puntos de decision donde Rex tiene libertad real:
      - Despues de <|box_start|>: P('None') vs P(token de coordenada) -> presencia de clase
      - Despues del 4o coord token: P(',') vs P('<|box_end|>') -> mas detecciones?
    """
    print(f"\n{'='*80}")
    print("BIFURCATION POINTS (puntos de decision real)")
    print(f"{'='*80}\n")

    # Decodificar token a token para encontrar las posiciones
    box_start_id = special_ids["box_start"]
    box_end_id = special_ids["box_end"]
    comma_id = tok.convert_tokens_to_ids(",")

    # Buscar IDs de "None" (puede ser un solo token o varios)
    none_ids = tok.encode("None", add_special_tokens=False)
    print(
        f"Token IDs: box_start={box_start_id}, box_end={box_end_id}, comma={comma_id}"
    )
    print(f"'None' encoded as: {none_ids} -> {[tok.decode([i]) for i in none_ids]}")
    print()

    gen_list = gen_ids.tolist()

    # Bifurcacion 1: despues de cada <|box_start|>
    print("--- Bifurcacion 1: presencia de clase (token tras <|box_start|>) ---")
    print(
        f"{'pos':<5} {'class':<20} {'gen':<10} {'P(gen)':<10} {'P(None)':<10}"
        f" {'P(coord)':<12} {'detection?':<12}"
    )
    print("-" * 80)
    for step_idx, tok_id in enumerate(gen_list):
        if tok_id == box_start_id:
            # Buscar el nombre de clase: tokens entre el <|object_ref_start|> previo
            # y este punto (saltando obj_end y box_start)
            cls_name = "?"
            for back in range(step_idx - 1, max(0, step_idx - 10), -1):
                if gen_list[back] == special_ids["obj_end"]:
                    cls_tokens = []
                    for k in range(back - 1, -1, -1):
                        if gen_list[k] == special_ids["obj_start"]:
                            break
                        cls_tokens.insert(0, gen_list[k])
                    cls_name = tok.decode(cls_tokens, skip_special_tokens=False).strip()
                    break

            if step_idx + 1 >= len(scores):
                continue
            next_step = step_idx + 1  # paso de generacion del token siguiente
            next_tok_id = gen_list[next_step] if next_step < len(gen_list) else -1
            probs = torch.softmax(scores[next_step][0].float(), dim=-1)

            p_gen = float(probs[next_tok_id])
            p_none = float(probs[none_ids[0]]) if none_ids else 0.0

            # P de que el siguiente token sea cualquier coord token: agregar prob
            # sobre todos los IDs cuyo decode sea "<NUM>"
            # Aproximacion: solo miramos el top-1 + alternativas comunes
            top10 = probs.topk(10)
            p_coord = 0.0
            for p, idx in zip(top10.values.tolist(), top10.indices.tolist()):
                txt = tok.decode([idx])
                if _is_coord_token(txt):
                    p_coord += p

            gen_txt = tok.decode([next_tok_id])
            detection = (
                "SI"
                if _is_coord_token(gen_txt)
                else ("NO" if "None" in gen_txt else "??")
            )
            print(
                f"{step_idx:<5} {cls_name:<20} {repr(gen_txt):<10} {p_gen:<10.4f}"
                f" {p_none:<10.4f} {p_coord:<12.4f} {detection:<12}"
            )

    # Bifurcacion 2: entre detecciones dentro del mismo bloque de clase
    print("\n--- Bifurcacion 2: continuar con mas detecciones? ---")
    comma_label = "P(comma)"
    print(
        f"{'pos':<5} {'gen':<10} {'P(gen)':<10} {comma_label:<10}"
        f" {'P(box_end)':<14} {'continuar?':<12}"
    )
    print("-" * 80)
    # Detectar tras cada secuencia de 4 coord tokens: el 5o token deberia ser
    # ',' o '<|box_end|>'
    in_box_block = False
    coord_count = 0
    for step_idx, tok_id in enumerate(gen_list):
        if tok_id == box_start_id:
            in_box_block = True
            coord_count = 0
            continue
        if tok_id == box_end_id:
            in_box_block = False
            continue
        if not in_box_block:
            continue
        txt = tok.decode([tok_id])
        if _is_coord_token(txt):
            coord_count += 1
            if coord_count == 4:
                # Despues de 4 coords viene la bifurcacion ',' vs '<|box_end|>'
                bifurc_step = step_idx + 1
                if bifurc_step >= len(scores):
                    continue
                next_tok_id = (
                    gen_list[bifurc_step] if bifurc_step < len(gen_list) else -1
                )
                probs = torch.softmax(scores[bifurc_step][0].float(), dim=-1)
                p_gen = float(probs[next_tok_id])
                p_comma = float(probs[comma_id])
                p_box_end = float(probs[box_end_id])
                gen_txt = tok.decode([next_tok_id])
                cont = (
                    "SI"
                    if "," in gen_txt
                    else (
                        "NO"
                        if "box_end" in gen_txt or next_tok_id == box_end_id
                        else "??"
                    )
                )
                print(
                    f"{bifurc_step:<5} {repr(gen_txt):<10} {p_gen:<10.4f}"
                    f" {p_comma:<10.4f} {p_box_end:<14.4f} {cont:<12}"
                )
                coord_count = 0  # reset para la siguiente caja


def analyze_detection_scores(
    model, processor, image: Image.Image, categories: list[str]
) -> None:
    """Genera con output_scores=True y analiza por deteccion."""
    print(f"\nImagen: {image.size}  categorias: {categories}")

    # Construir el prompt como hace el wrapper
    cat_str = ", ".join(categories)
    user_text = (
        f"Please detect all instances of the following objects in the image: {cat_str}. "
        f"Output bounding boxes for each detected instance."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # IDs de los tokens estructurales
    tok = processor.tokenizer
    special_ids = {k: tok.convert_tokens_to_ids(v) for k, v in SPECIAL_TOKENS.items()}
    print(f"Special token IDs: {special_ids}")

    print("Generando con output_scores=True ...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            top_p=0.8,
            top_k=1,
            repetition_penalty=1.05,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # generated_ids: (1, prompt_len + gen_len)
    # scores: tuple of length gen_len, cada uno (1, vocab_size)
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = out.sequences[0, prompt_len:]  # (gen_len,)
    scores = out.scores  # tuple de gen_len tensors

    print(f"Tokens generados: {len(gen_ids)}")
    print(f"Logits por paso: {scores[0].shape}  (esperado: (1, vocab_size))")

    # Recorrer los tokens identificando los markers estructurales
    decoded_full = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"\n--- Texto generado completo (primeros 500 chars) ---")
    print(decoded_full[:500])
    print("---\n")

    # Estados:
    #   "outside"      -> esperando <|object_ref_start|>
    #   "class"        -> recogiendo tokens de clase hasta <|object_ref_end|>
    #   "between"      -> esperando <|box_start|>
    #   "coords"       -> recogiendo coordenadas hasta <|box_end|>
    state = "outside"
    current_class_tokens: list[tuple[int, int]] = []  # (step_idx, token_id)
    current_coord_tokens: list[tuple[int, int]] = []
    detections = []

    for step_idx, tok_id in enumerate(gen_ids.tolist()):
        if tok_id == special_ids["obj_start"]:
            state = "class"
            current_class_tokens = []
            current_coord_tokens = []
        elif tok_id == special_ids["obj_end"]:
            state = "between"
        elif tok_id == special_ids["box_start"]:
            state = "coords"
        elif tok_id == special_ids["box_end"]:
            if current_class_tokens and current_coord_tokens:
                detections.append(
                    {
                        "class_tokens": current_class_tokens,
                        "coord_tokens": current_coord_tokens,
                    }
                )
            state = "outside"
        else:
            if state == "class":
                current_class_tokens.append((step_idx, tok_id))
            elif state == "coords":
                # Filtrar tokens que no sean coordenadas (e.g. la coma ",")
                # Las coordenadas tienen forma "<NUM>" — los detectamos por su decode
                tok_text = tok.decode([tok_id], skip_special_tokens=False)
                if (
                    tok_text.startswith("<")
                    and tok_text.endswith(">")
                    and tok_text[1:-1].isdigit()
                ):
                    current_coord_tokens.append((step_idx, tok_id))

    print(f"Detecciones extraidas: {len(detections)}")

    if not detections:
        print("  (No se parseo ninguna deteccion. Revisa el output.)")
        return

    # Analizar cada deteccion
    print(f"\n{'='*80}")
    print(
        f"{'#':<3} {'class_tokens_min_p':<20} {'class_mean_H':<14}"
        f" {'coord_tokens_min_p':<20} {'coord_mean_H':<14}"
    )
    print(f"{'='*80}")

    for i, det in enumerate(detections[:30]):  # limitar a 30 para no spammear
        # Stats de tokens de clase
        cls_stats = [
            _summarize_token_step(scores[step][0], tok_id)
            for step, tok_id in det["class_tokens"]
        ]
        cls_probs = [s["prob_generated"] for s in cls_stats]
        cls_ents = [s["entropy"] for s in cls_stats]

        # Stats de tokens de coordenadas
        coord_stats = [
            _summarize_token_step(scores[step][0], tok_id)
            for step, tok_id in det["coord_tokens"]
        ]
        coord_probs = [s["prob_generated"] for s in coord_stats]
        coord_ents = [s["entropy"] for s in coord_stats]

        # Decodificar para debug
        cls_text = tok.decode(
            [t for _, t in det["class_tokens"]], skip_special_tokens=False
        ).strip()
        coord_text = tok.decode(
            [t for _, t in det["coord_tokens"]], skip_special_tokens=False
        ).strip()

        min_p_cls = min(cls_probs) if cls_probs else float("nan")
        mean_h_cls = sum(cls_ents) / len(cls_ents) if cls_ents else float("nan")
        min_p_coord = min(coord_probs) if coord_probs else float("nan")
        mean_h_coord = sum(coord_ents) / len(coord_ents) if coord_ents else float("nan")

        print(
            f"{i:<3} {min_p_cls:<20.4f} {mean_h_cls:<14.4f}"
            f" {min_p_coord:<20.4f} {mean_h_coord:<14.4f}"
            f"  cls='{cls_text}' coords='{coord_text[:40]}'"
        )

    # Resumen estadistico
    print(f"\n--- Resumen agregado ---")
    all_cls_probs = []
    all_coord_probs = []
    for det in detections:
        all_cls_probs.extend(
            [
                _summarize_token_step(scores[s][0], t)["prob_generated"]
                for s, t in det["class_tokens"]
            ]
        )
        all_coord_probs.extend(
            [
                _summarize_token_step(scores[s][0], t)["prob_generated"]
                for s, t in det["coord_tokens"]
            ]
        )

    def _stats(name: str, vals: list[float]) -> None:
        if not vals:
            print(f"  {name}: (sin datos)")
            return
        vals_sorted = sorted(vals)
        n = len(vals)
        print(
            f"  {name}: n={n}  min={min(vals):.4f}  "
            f"p25={vals_sorted[n//4]:.4f}  median={vals_sorted[n//2]:.4f}  "
            f"p75={vals_sorted[3*n//4]:.4f}  max={max(vals):.4f}  "
            f"mean={sum(vals)/n:.4f}"
        )

    _stats("class token probs", all_cls_probs)
    _stats("coord token probs", all_coord_probs)

    print("\nInterpretacion:")
    print(
        "  Si todos los valores son ~1.0 -> el modelo esta saturado, scores no informativos."
    )
    print(
        "  Si hay variacion significativa (p25 < 0.7) -> vale la pena la implementacion completa."
    )

    # Bifurcacion points: donde Rex tiene libertad real de decision
    analyze_bifurcation_points(gen_ids, scores, tok, special_ids)


def test_inference_with_scores(
    model_id: str, image: Image.Image, categories: list[str]
) -> None:
    """Verifica que el flujo completo de inference(return_scores=True) funciona."""
    print(f"\n{'='*80}\nTest: inference(return_scores=True)\n{'='*80}")
    from rex_omni import RexOmniWrapper

    wrapper = RexOmniWrapper(
        model_path=model_id,
        backend="transformers",
        attn_implementation="eager",
    )
    results = wrapper.inference(
        images=[image],
        task="detection",
        categories=categories,
        return_scores=True,
    )
    extracted = results[0]["extracted_predictions"]
    for cls_name, anns in extracted.items():
        if not anns:
            continue
        print(f"\n  Class '{cls_name}' ({len(anns)} detecciones):")
        for j, ann in enumerate(anns[:5]):
            print(
                f"    [{j}] score={ann.get('score', '?'):.4f}"
                f"  cls_presence={ann.get('cls_presence', '?'):.4f}"
                f"  loc_score={ann.get('loc_score', '?'):.4f}"
                f"  coords={[round(c,1) for c in ann['coords']]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=Path,
        default=Path(
            "results/preprocess/tile_images/train_unlabeled_eval/images/100_1_0.tif"
        ),
        help="Imagen de prueba",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["aircraft", "building", "vehicle", "ship"],
        help="Categorias para detectar",
    )
    parser.add_argument("--model_id", default="IDEA-Research/Rex-Omni")
    parser.add_argument(
        "--test_full_flow",
        action="store_true",
        help="Solo testear el flujo completo inference(return_scores=True), saltar analisis bruto",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        # Buscar cualquier imagen del eval set
        candidates = list(
            Path("results/preprocess/tile_images/train_unlabeled_eval/images").glob(
                "*.tif"
            )
        )[:1]
        if not candidates:
            raise FileNotFoundError(f"No se encontro imagen en {args.image_path}")
        args.image_path = candidates[0]
        print(f"Usando imagen por defecto: {args.image_path}")

    image = Image.open(args.image_path).convert("RGB")

    print(f"Cargando modelo {args.model_id} ...")
    from rex_omni import RexOmniWrapper

    wrapper = RexOmniWrapper(
        model_path=args.model_id,
        backend="transformers",
        attn_implementation="eager",
    )

    if args.test_full_flow:
        # Solo test del flujo completo con inference(return_scores=True)
        results = wrapper.inference(
            images=[image],
            task="detection",
            categories=args.categories,
            return_scores=True,
        )
        extracted = results[0]["extracted_predictions"]
        print(
            f"\n=== inference(return_scores=True) — {sum(len(a) for a in extracted.values())} dets ==="
        )
        for cls_name, anns in extracted.items():
            if not anns:
                continue
            print(f"\n  Class '{cls_name}' ({len(anns)} detecciones):")
            for j, ann in enumerate(anns[:10]):
                print(
                    f"    [{j}] score={ann.get('score', 0):.4f}"
                    f"  cls_presence={ann.get('cls_presence', 0):.4f}"
                    f"  loc_score={ann.get('loc_score', 0):.4f}"
                    f"  coords={[round(c,1) for c in ann['coords']]}"
                )
    else:
        analyze_detection_scores(
            wrapper.model, wrapper.processor, image, args.categories
        )


if __name__ == "__main__":
    main()
