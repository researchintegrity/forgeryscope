"""
Single-image duplication detection.

Runs the steps of the single-image example of the README, with every model
loaded once:

1. Detect the panels of the figure (western blots and microscopy).
2. Select candidate panel pairs by embedding similarity.
3. Locate the duplicated regions of each candidate pair with LightGlue
   matching and geometric verification.
4. When no pair of blot panels is a candidate, compare individual western
   blot lanes.

Example:
    detector = DuplicationDetector(device="cuda")
    result = detector.detect(PanelExtractor._load_image("figure.png"))
    for detection in result.detections:
        print(detection.level, detection.panel_type, detection.panel_ids)
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ultralytics import YOLO

from forgeryscope.detector import PanelExtractor
from forgeryscope.embedder import Embedder
from forgeryscope.matcher.geometry import get_intersections
from forgeryscope.matcher.lane import create_lane_match_masks, find_lanes_in_blot_panels
from forgeryscope.matcher.lightglue import LightGlueOverlap, create_duplicate_masks, merge_masks_by_max_cliques
from forgeryscope.model_zoo import get_model_path, load_aliked_wblot_weights

BLOTS = "Blots"
MICROSCOPY = "Microscopy"

Box = Tuple[float, float, float, float]
Polygon = List[Tuple[float, float]]


@dataclass
class PipelineConfig:
    """Thresholds of the single-image example."""
    panel_conf_threshold: float = 0.7
    panel_iou_threshold: float = 0.4
    excluded_panel_labels: Tuple[str, ...] = ("Graphs", "Flow Cytometry", "Body Imaging")
    intersection_margin: float = 10
    wblot_overlap_threshold: float = 0.85
    wblot_duplicate_threshold: float = 0.84
    microscopy_embedding_threshold: float = 0.58
    inlier_threshold: int = 8
    match_score_threshold: float = 0.73
    lane_similarity_threshold: float = 0.65
    lane_overlap_threshold: int = 5


@dataclass
class Panel:
    id: int
    label: str
    confidence: float
    bbox: Box


@dataclass
class PanelPair:
    """A candidate pair of panels kept after matching."""
    panel_ids: Tuple[int, int]
    embedding_score: float
    # False when a blot pair is kept on embedding similarity alone
    # (no keypoint match): its regions are then the whole panels.
    keypoint_match: bool
    inliers: int
    total_matches: int
    match_score: float
    # Duplicated region in each panel, in image coordinates
    regions: Tuple[Polygon, Polygon]


@dataclass
class LaneMatch:
    """A pair of similar western blot lanes."""
    panel_ids: Tuple[int, int]
    similarity: float
    lane_boxes: Tuple[Box, Box]


@dataclass
class Detection:
    """A group of duplicated regions."""
    level: str  # "panel": panel or region matching, "lane": lane matching
    panel_type: str
    panel_ids: List[int]
    mask: np.ndarray  # bool, image height x width
    pairs: List[PanelPair] = field(default_factory=list)
    lane_matches: List[LaneMatch] = field(default_factory=list)

    @property
    def similarity(self) -> float:
        """Highest embedding similarity behind this detection."""
        scores = [p.embedding_score for p in self.pairs] + [m.similarity for m in self.lane_matches]
        return max(scores, default=0.0)


@dataclass
class DetectionResult:
    panels: List[Panel]
    detections: List[Detection]

    @property
    def authentic(self) -> bool:
        return not self.detections


class _PredictOnDevice:
    """A YOLO model whose predict() runs on a fixed device (lane matching
    calls predict() without one, which would pick any available GPU)."""

    def __init__(self, model: YOLO, device: str):
        self.model = model
        self.device = device

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, device=self.device, **kwargs)


class DuplicationDetector:
    """Detects duplicated panels, panel regions and blot lanes in a figure."""

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[PipelineConfig] = None,
        cache_dir: Optional[str] = None,
        model_base_url: Optional[str] = None,
    ):
        self.device = device
        self.config = config or PipelineConfig()
        zoo = {"cache_dir": cache_dir, "model_base_url": model_base_url}

        self.panel_extractor = PanelExtractor(
            weights_path="yolo_panel_extractor",
            device=device,
            conf_threshold=self.config.panel_conf_threshold,
            iou_threshold=self.config.panel_iou_threshold,
            **zoo,
        )
        self.panel_extractor.EXCLUDED_LABELS = set(self.config.excluded_panel_labels)
        lane_model = YOLO(get_model_path("yolo_lane_extractor", cache_dir=cache_dir, base_url=model_base_url))
        self.lane_extractor = _PredictOnDevice(lane_model, device)

        self.wblot_duplicate_embedder = Embedder("wblot_duplicate_embedder", device=device, **zoo)
        self.wblot_overlap_embedder = Embedder("wblot_overlap_embedder", device=device, **zoo)
        self.wblot_lane_embedder = Embedder("wblot_lane_embedder", device=device, **zoo)
        self.micro_overlap_embedder = Embedder("micro_overlap_embedder", device=device, **zoo)

        self.matcher_micro = LightGlueOverlap(
            max_keypoints=4096,
            matcher_features="sift",
            device=device,
            depth_confidence=0.9,
            width_confidence=0.9,
        )
        self.matcher_blot = LightGlueOverlap(
            max_keypoints=512,
            matcher_features="aliked",
            device=device,
            depth_confidence=-1,
            width_confidence=-1,
            estimator_method="MAGSAC",
            reprojThreshold=3.0,
            estimator_confidence=0.9999,
            estimator_maxIters=5000,
            estimator_refineIters=10,
        )
        load_aliked_wblot_weights(self.matcher_blot, cache_dir=cache_dir, base_url=model_base_url)

    def detect(self, img: np.ndarray, progress: Optional[Callable[[str], None]] = None) -> DetectionResult:
        """
        Detect duplicated content in one figure.

        Args:
            img: RGB image (H, W, 3), uint8
            progress: Called with a short message as each step starts

        Returns:
            DetectionResult with the detected panels and the detections
            (no detections: the figure looks authentic).
        """
        report = progress or (lambda message: None)
        config = self.config

        report("Detecting panels")
        panel_tuples, crops = [], []
        for panel in self.panel_extractor.extract_panels(img):
            crop = PanelExtractor.crop_panels(img, [panel])
            if crop:  # boxes that are empty once clipped to the image have no crop
                panel_tuples.append(panel)
                crops.append(crop[0])
        panels = [
            Panel(id=i, label=label, confidence=float(conf), bbox=_box((x0, y0, x1, y1)))
            for i, (label, conf, x0, y0, x1, y1) in enumerate(panel_tuples)
        ]
        if not panels:
            return DetectionResult(panels=[], detections=[])

        blot_ids = [p.id for p in panels if p.label == BLOTS]
        micro_ids = [p.id for p in panels if p.label == MICROSCOPY]
        report(f"Comparing {len(blot_ids)} blot and {len(micro_ids)} microscopy panels")

        blot_scores: Dict[Tuple[int, int], float] = {}
        for embedder, threshold in (
            (self.wblot_overlap_embedder, config.wblot_overlap_threshold),
            (self.wblot_duplicate_embedder, config.wblot_duplicate_threshold),
        ):
            for i, j, score in _similar_pairs(blot_ids, crops, embedder, threshold):
                blot_scores[(i, j)] = max(score, blot_scores.get((i, j), score))
        micro_pairs = _similar_pairs(micro_ids, crops, self.micro_overlap_embedder, config.microscopy_embedding_threshold)

        # Overlapping panels are not compared with each other
        intersections = get_intersections(panel_tuples, margin=config.intersection_margin)
        candidates = [(BLOTS, score, i, j) for (i, j), score in blot_scores.items()]
        candidates += [(MICROSCOPY, score, i, j) for i, j, score in micro_pairs]
        candidates = [
            (label, score, i, j) for label, score, i, j in candidates
            if (i, j) not in intersections and (j, i) not in intersections
        ]

        detections = self._match_panels(img, panel_tuples, crops, candidates, report) if candidates else []
        if blot_ids and not blot_scores:
            detections += self._match_lanes(img, panel_tuples, crops, blot_ids, report)
        return DetectionResult(panels=panels, detections=detections)

    def _match_panels(self, img, panel_tuples, crops, candidates, report) -> List[Detection]:
        config = self.config
        report(f"Matching {len(candidates)} candidate panel pairs")
        match_results = create_duplicate_masks(
            img,
            panel_tuples,
            crops,
            pd.DataFrame(candidates, columns=["label", "score", "idx1", "idx2"]),
            self.matcher_micro,
            self.matcher_blot,
            to_bbox_micro=False,
            to_bbox_blot=True,
            fallback_for_wblot=True,
            test_transforms_blot=False,
            test_transforms_micro=True,
        )
        scores = {(i, j): score for _, score, i, j in candidates}

        kept, masks = [], []
        for info in match_results:
            result = info["match_result"]
            if info["panel_label"] != BLOTS and (
                result["inliers"] < config.inlier_threshold
                or result["mean_match_score"] < config.match_score_threshold
            ):
                continue
            kept.append(info)
            masks.append((info["mask0"] | info["mask1"]).astype(np.uint8))

        detections = []
        merged_masks, merged_info = merge_masks_by_max_cliques(masks, kept)
        for mask, group in zip(merged_masks, merged_info):
            members = [kept[k] for k in group["original_indices"]]
            detections.append(Detection(
                level="panel",
                panel_type=members[0]["panel_label"],
                panel_ids=[int(i) for i in group["panel_ids"]],
                mask=mask.astype(bool),
                pairs=[_panel_pair(info, scores) for info in members],
            ))
        return detections

    def _match_lanes(self, img, panel_tuples, crops, blot_ids, report) -> List[Detection]:
        report("Comparing western blot lanes")
        lanes = find_lanes_in_blot_panels(
            panels=panel_tuples,
            blot_panels_ids=blot_ids,
            crops_list=crops,
            segmentator=self.lane_extractor,
            blot_duplicate_detector=self.wblot_lane_embedder,
            similarity_threshold=self.config.lane_similarity_threshold,
            overlap_threshold=self.config.lane_overlap_threshold,
        )
        matches = lanes["best_matches"]
        if not matches:
            return []

        # create_lane_match_masks returns one mask per panel pair, sorted by pair
        masks = create_lane_match_masks(img.shape, matches, lanes=lanes["lanes"])
        by_pair = defaultdict(list)
        for match in matches:
            by_pair[tuple(sorted((match.panel_idx1, match.panel_idx2)))].append(match)

        return [
            Detection(
                level="lane",
                panel_type=BLOTS,
                panel_ids=sorted({int(p) for p in pair}),
                mask=mask.astype(bool),
                lane_matches=[
                    LaneMatch(
                        panel_ids=(int(m.panel_idx1), int(m.panel_idx2)),
                        similarity=float(m.similarity),
                        lane_boxes=(_box(m.bbox1_absolute), _box(m.bbox2_absolute)),
                    )
                    for m in group
                ],
            )
            for (pair, group), mask in zip(sorted(by_pair.items()), masks)
        ]


def _similar_pairs(
    panel_ids: Sequence[int],
    crops: Sequence[np.ndarray],
    embedder: Embedder,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Pairs of panels whose embeddings are at least `threshold` similar."""
    if len(panel_ids) < 2:
        return []
    embeddings = embedder.get_embedding_batch([crops[i] for i in panel_ids]).cpu()
    return [
        (panel_ids[a], panel_ids[b], float(score))
        for a, b, score in Embedder.find_similar_pairs(embeddings, threshold=threshold)
    ]


def _panel_pair(info: dict, scores: Dict[Tuple[int, int], float]) -> PanelPair:
    i, j = int(info["panel_id0"]), int(info["panel_id1"])
    result = info["match_result"]
    return PanelPair(
        panel_ids=(i, j),
        embedding_score=scores[(i, j)],
        keypoint_match="fallback" not in result,
        inliers=int(result["inliers"]),
        total_matches=int(result["total_matches"]),
        match_score=float(result["mean_match_score"]),
        regions=(_polygon(info["poly_coords0"]), _polygon(info["poly_coords1"])),
    )


def _box(coords) -> Box:
    x0, y0, x1, y1 = (round(float(v), 1) for v in coords[-4:])
    return (x0, y0, x1, y1)


def _polygon(coords) -> Polygon:
    return [(round(float(x), 1), round(float(y), 1)) for x, y in coords]
