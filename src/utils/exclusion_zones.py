"""
Subclases de RegionProposalNetwork e RoIHeads de torchvision que respetan
zonas de exclusión definidas no target dict.

Mecánica:
    - O matcher estándar de torchvision marca como label = -1 os anchors
      ou proposals cuxo IoU con GT cae entre os dous limiares de IoU
      (``BETWEEN_THRESHOLDS``). O ``BalancedPositiveNegativeSampler``
      ignora calquera anchor con label == -1 ao construír o loss.
    - Aproveitamos ese gancho: tras o matching contra GT real, comprobamos
      o IoU de cada anchor / proposal contra as zonas de exclusión do
      target. Se algún supera o limiar ``exclusion_iou_thresh`` (default
      0.5), marcamos o seu label como -1.
    - As zonas de exclusión NON contribúen como GT positivo nin como fondo
      negativo: simplemente eliminan a contribución dos anchors / proposals
      que caen sobre elas.

Convención do target dict:
    target["exclusion_zones"] : Tensor[M, 4] en formato XYXY (mesma
        canvas_size que ``target["boxes"]``). Pode estar ausente ou ser
        baleiro; en ambos casos a clase fai un no-op no paso de exclusión.

Inxección no modelo:
    Tras construír un FasterRCNN estándar, substituír os atributos
    ``model.rpn`` e ``model.roi_heads`` polas subclases empregando
    ``rebuild_rpn_with_exclusion`` e ``rebuild_roi_heads_with_exclusion``,
    que copian todos os atributos do orixinal e só sobrescriben os métodos
    de asignación.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import box_iou


DEFAULT_EXCLUSION_IOU_THRESH = 0.5


def _exclusion_mask(
    candidate_boxes: Tensor,
    exclusion_zones: Tensor | None,
    iou_thresh: float,
) -> Tensor | None:
    """Devolve un tensor bool [N] indicando, por cada candidate_box, se cae
    nunha zona de exclusión (max IoU con calquera zona >= iou_thresh).
    Devolve ``None`` se non hai zonas válidas."""
    if exclusion_zones is None or exclusion_zones.numel() == 0:
        return None
    iou = box_iou(exclusion_zones, candidate_boxes)  # [M, N]
    return (iou >= iou_thresh).any(dim=0)


class RPNWithExclusion(RegionProposalNetwork):
    """RPN con soporte para zonas de exclusión no target dict.

    Atributos extra:
        exclusion_iou_thresh: limiar de IoU para marcar un anchor como
            ignorado cando solapa cunha zona de exclusión.
    """

    exclusion_iou_thresh: float = DEFAULT_EXCLUSION_IOU_THRESH

    def assign_targets_to_anchors(
        self,
        anchors: List[Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        labels, matched_gt_boxes = super().assign_targets_to_anchors(anchors, targets)

        for i, target in enumerate(targets):
            zones = target.get("exclusion_zones")
            mask = _exclusion_mask(anchors[i], zones, self.exclusion_iou_thresh)
            if mask is None:
                continue
            # Só convertimos FONDO (label == 0) en ignorado. Os anchors
            # positivos (label == 1, IoU alto con GT/PE real) preséravanse:
            # unha caixa positiva próxima a unha zona segue sendo sinal válida.
            mask = mask & (labels[i] == 0.0)
            labels[i] = labels[i].clone()
            labels[i][mask] = -1.0

        return labels, matched_gt_boxes


class RoIHeadsWithExclusion(RoIHeads):
    """RoIHeads con soporte para zonas de exclusión no target dict.

    Atributos extra:
        exclusion_iou_thresh: limiar de IoU para marcar un proposal como
            ignorado cando solapa cunha zona de exclusión.

    Notas:
        ``select_training_samples`` chama internamente a
        ``assign_targets_to_proposals``. Sobrescribimos esta última e
        accedemos á ``self._current_targets`` que poboamos no forward.
        Como o forward de RoIHeads recibe ``targets`` como argumento e a
        chamada a ``assign_targets_to_proposals`` non o reenvía, optamos
        polo enfoque limpo: sobrescribimos o forward para gardar o ref a
        targets antes da chamada interna.
    """

    exclusion_iou_thresh: float = DEFAULT_EXCLUSION_IOU_THRESH

    def forward(self, features, proposals, image_shapes, targets=None):
        # Gardamos targets temporalmente para que assign_targets_to_proposals
        # poida lelos.  Limpamos despois para non sufrir fugas de memoria
        # entre iteracións.
        self._current_targets = targets
        try:
            return super().forward(features, proposals, image_shapes, targets)
        finally:
            self._current_targets = None

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs, labels = super().assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        targets = getattr(self, "_current_targets", None)
        if targets is None:
            return matched_idxs, labels

        for i, target in enumerate(targets):
            zones = target.get("exclusion_zones")
            mask = _exclusion_mask(proposals[i], zones, self.exclusion_iou_thresh)
            if mask is None:
                continue
            # Só convertimos FONDO (label == 0) en ignorado; preservamos os
            # proposals positivos (label > 0) aínda que solapen cunha zona.
            mask = mask & (labels[i] == 0)
            labels[i] = labels[i].clone()
            labels[i][mask] = -1

        return matched_idxs, labels


def replace_rpn_with_exclusion(
    model,
    exclusion_iou_thresh: float = DEFAULT_EXCLUSION_IOU_THRESH,
) -> None:
    """Substitúe ``model.rpn`` por unha instancia de ``RPNWithExclusion``
    mutando ``__class__`` no instancia orixinal. Iso preserva todos os
    pesos, buffers, atributos internos e hooks de torchvision sen
    necesidade de reimplementar a inicialización."""
    model.rpn.__class__ = RPNWithExclusion
    model.rpn.exclusion_iou_thresh = exclusion_iou_thresh


def replace_roi_heads_with_exclusion(
    model,
    exclusion_iou_thresh: float = DEFAULT_EXCLUSION_IOU_THRESH,
) -> None:
    """Substitúe ``model.roi_heads`` por unha instancia de
    ``RoIHeadsWithExclusion`` mutando ``__class__``."""
    model.roi_heads.__class__ = RoIHeadsWithExclusion
    model.roi_heads.exclusion_iou_thresh = exclusion_iou_thresh
    model.roi_heads._current_targets = None


def add_exclusion_zones_support(
    model,
    exclusion_iou_thresh: float = DEFAULT_EXCLUSION_IOU_THRESH,
) -> None:
    """Helper que aplica as dúas substitucións nun modelo Faster R-CNN."""
    replace_rpn_with_exclusion(model, exclusion_iou_thresh)
    replace_roi_heads_with_exclusion(model, exclusion_iou_thresh)
