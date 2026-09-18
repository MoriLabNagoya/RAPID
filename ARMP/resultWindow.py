# -*- coding: utf-8 -*-
"""Bridge between the main measurement GUI and the standalone mask editor."""
import os
import csv
import heapq
from collections import deque
from datetime import datetime

from scipy import ndimage

import cv2
import numpy as np
from skimage import morphology

from ARMP.AnalyseRoot import AnalyseRoot as ar, RootProp
from ARMP.mask_editor import SemanticMaskEditor, launch_standalone


class RootResult:
    """Open and navigate plant ROIs in the semantic root annotation editor."""

    def __init__(self, master, mainFrame):
        self.master = master
        self.mainFrame = mainFrame
        self.curRoot = int(mainFrame.curRoot)
        if self.curRoot < 0 or self.curRoot >= len(mainFrame.RootAll):
            raise ValueError("No root ROI is selected.")

        self.eachroot = None
        self.OrigImage = None
        self.semantic = None
        self.editor = None

        self._load_roi_state(self.curRoot)

        self.editor = SemanticMaskEditor(
            master,
            image=self.OrigImage,
            mask=self.semantic,
            reference_mask=None,
            title=self._editor_title(),
            history_path=self._history_path(),
            on_apply=self._apply_semantic_mask,
            allow_open_image=False,
            resolution=(float(getattr(mainFrame, "resolution", 1.0))
                        if getattr(mainFrame, "bSetResolution", False) else 1.0),
            unit=(str(getattr(mainFrame, "unit", "px"))
                  if getattr(mainFrame, "bSetResolution", False) else "px"),
            on_roi_step=self._navigate_roi,
            roi_index=self.curRoot,
            roi_count=len(mainFrame.RootAll),
            on_save_validation=self._save_validation_results,
            language=getattr(mainFrame, "ui_language", "en"),
        )

    def _history_path(self):
        history_dir = os.path.join(self.mainFrame.ProjectDir, "editor_history")
        os.makedirs(history_dir, exist_ok=True)
        return os.path.join(
            history_dir,
            f"{self.mainFrame.file_name}_roi_{self.curRoot + 1:03d}.npz",
        )

    def _editor_title(self):
        processed = bool(getattr(self.eachroot, "bProcessed", False))
        if getattr(self.mainFrame, "ui_language", "en") == "ja":
            state = "処理済み" if processed else "未処理"
            return (
                f"RAPID - 根アノテーションエディター - ROI {self.curRoot + 1}/"
                f"{len(self.mainFrame.RootAll)}（{state}）"
            )
        state = "processed" if processed else "unprocessed"
        return (
            f"RAPID - Root annotation editor - ROI {self.curRoot + 1}/"
            f"{len(self.mainFrame.RootAll)} ({state})"
        )

    def _load_roi_state(self, index):
        self.curRoot = int(index)
        self.mainFrame.curRoot = self.curRoot
        self.eachroot = self.mainFrame.RootAll[self.curRoot]
        self.OrigImage = self._roi_rgb_image()
        self.semantic = self._initial_semantic_mask()

    def _navigate_roi(self, delta):
        """A/D: apply dirty edits, then load the previous/next YOLO ROI."""
        target = self.curRoot + int(delta)
        if target < 0 or target >= len(self.mainFrame.RootAll):
            return False

        # Preserve current ROI before leaving it. Clean ROIs are not reprocessed.
        if self.editor is not None and bool(getattr(self.editor, "_dirty", False)):
            self.editor.apply_to_project()

        self._load_roi_state(target)
        self.editor.load_context(
            image=self.OrigImage,
            mask=self.semantic,
            reference_mask=None,
            title=self._editor_title(),
            history_path=self._history_path(),
            roi_index=self.curRoot,
            roi_count=len(self.mainFrame.RootAll),
        )
        return True

    def _roi_rgb_image(self):
        img = self.eachroot.mImgCV
        if img is None:
            bb = self.eachroot.bbox
            src = self.mainFrame.curImage[
                int(bb[1]):int(bb[3]),
                int(bb[0]):int(bb[2]),
            ]
            if src.ndim == 2:
                return cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            return src.astype(np.uint8).copy()
        if img.ndim == 2:
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # Generatebb currently receives RGB->BGR converted data in gui_main.
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    def _initial_semantic_mask(self):
        direct = getattr(self.eachroot, "semantic_mask", None)
        if direct is not None and np.shape(direct) == self.OrigImage.shape[:2]:
            return np.asarray(direct, dtype=np.uint8).copy()

        try:
            _, semantic = self.mainFrame._roi_export_masks(self.eachroot)
            if (
                semantic is not None
                and semantic.shape == self.OrigImage.shape[:2]
                and np.any(semantic)
            ):
                return semantic.astype(np.uint8)
        except Exception:
            pass

        # Fallback for projects saved from a full-image semantic mask.
        full_sem_path = os.path.join(self.mainFrame.ProjectDir, "mask_labels.png")
        if os.path.isfile(full_sem_path):
            data = np.fromfile(full_sem_path, dtype=np.uint8)
            full_sem = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if full_sem is not None:
                bb = self.eachroot.bbox
                roi = full_sem[
                    int(bb[1]):int(bb[3]),
                    int(bb[0]):int(bb[2]),
                ]
                if roi.shape == self.OrigImage.shape[:2]:
                    return roi.astype(np.uint8)

        return np.zeros(self.OrigImage.shape[:2], dtype=np.uint8)

    @staticmethod
    def _component_props(semantic, img_type, roi_image=None):
        """Convert an instance-aware semantic mask into measurable RootProp objects."""
        processor = ar(roi_image, 5 if roi_image is not None else 1, 1, 0)
        props = []
        instance_label = np.zeros(semantic.shape, dtype=np.uint16)
        instance_id = 0

        def append_components(class_mask, root_type, source_value):
            nonlocal instance_id
            nlabels, labels = cv2.connectedComponents(
                class_mask.astype(np.uint8), connectivity=8
            )
            for component_id in range(1, nlabels):
                component = (labels == component_id).astype(np.uint8)
                if int(component.sum()) == 0:
                    continue
                measure_mask = (
                    component
                    if root_type == 1
                    else morphology.skeletonize(component > 0).astype(np.uint8)
                )
                if not np.any(measure_mask):
                    continue
                try:
                    if (
                        root_type == 1
                        and roi_image is not None
                        and hasattr(processor, "GetLeafFromMaskImproved")
                    ):
                        prop = processor.GetLeafFromMaskImproved(component)
                    else:
                        prop = processor.ProcessLoad(measure_mask, img_type, root_type)
                    prop.type = root_type
                    prop.mask = measure_mask.copy()
                except Exception:
                    pts = list(map(tuple, np.argwhere(measure_mask > 0)))
                    start = pts[0] if pts else (0, 0)
                    prop = RootProp(
                        0.0, 0.0, pts, root_type, start, measure_mask, 0
                    )
                prop.label_value = int(source_value)
                props.append(prop)
                instance_id += 1
                instance_label[measure_mask > 0] = instance_id

        append_components(semantic == 1, 1, 1)
        append_components(semantic == 2, 2, 2)
        for value in sorted(int(v) for v in np.unique(semantic) if int(v) >= 3):
            append_components(semantic == value, 3, value)

        if instance_id <= 255:
            instance_label = instance_label.astype(np.uint8)
        return props, instance_label

    @staticmethod
    def _binary_metrics(pred, gt):
        pred = np.asarray(pred, dtype=bool)
        gt = np.asarray(gt, dtype=bool)
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        tn = int(np.logical_and(~pred, ~gt).sum())
        dice_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn
        precision_den = tp + fp
        recall_den = tp + fn
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Dice": (2.0 * tp / dice_den) if dice_den else 1.0,
            "IoU": (tp / iou_den) if iou_den else 1.0,
            "Precision": (tp / precision_den) if precision_den else 1.0,
            "Recall": (tp / recall_den) if recall_den else 1.0,
            "PredPixels": int(pred.sum()),
            "GTPixels": int(gt.sum()),
        }

    @staticmethod
    def _centroid_rc(binary):
        coords = np.argwhere(np.asarray(binary) > 0)
        if coords.size == 0:
            return None
        return float(coords[:, 0].mean()), float(coords[:, 1].mean())

    @staticmethod
    def _component_masks(binary):
        binary = (np.asarray(binary) > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        items = []
        for component_id in range(1, n):
            component = labels == component_id
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area > 0:
                items.append((area, component))
        items.sort(key=lambda item: item[0], reverse=True)
        return [mask for _, mask in items]

    def _leaf_instance_masks(self, semantic):
        leaf = (np.asarray(semantic) == 1).astype(np.uint8)
        if not np.any(leaf):
            return []
        try:
            processor = ar(self.OrigImage, 5, 1, 0)
            prop = processor.GetLeafFromMaskImproved(leaf)
            labels = np.asarray(getattr(prop, "leaf_instance_labels", None))
            if labels.shape == leaf.shape and np.max(labels) > 0:
                return [
                    labels == value
                    for value in sorted(int(v) for v in np.unique(labels) if int(v) > 0)
                ]
        except Exception:
            pass
        return self._component_masks(leaf)

    @staticmethod
    def _match_masks_by_iou(pred_masks, gt_masks):
        pairs = []
        candidates = []
        for pi, pmask in enumerate(pred_masks):
            p = np.asarray(pmask, dtype=bool)
            for gi, gmask in enumerate(gt_masks):
                g = np.asarray(gmask, dtype=bool)
                inter = int(np.logical_and(p, g).sum())
                union = int(np.logical_or(p, g).sum())
                iou = inter / union if union else 0.0
                candidates.append((iou, inter, pi, gi))

        used_p = set()
        used_g = set()
        for iou, inter, pi, gi in sorted(candidates, reverse=True):
            if pi in used_p or gi in used_g or inter <= 0:
                continue
            used_p.add(pi)
            used_g.add(gi)
            pairs.append((pi, gi, float(iou)))

        for gi in range(len(gt_masks)):
            if gi not in used_g:
                pairs.append((None, gi, 0.0))
        for pi in range(len(pred_masks)):
            if pi not in used_p:
                pairs.append((pi, None, 0.0))
        return pairs

    def _props_by_root_label(self, semantic):
        if hasattr(self.mainFrame, "_semantic_to_props"):
            props, _ = self.mainFrame._semantic_to_props(semantic, self.OrigImage)
        else:
            props, _ = self._component_props(semantic, self.mainFrame.imgType, self.OrigImage)

        grouped = {}
        for prop in props:
            rtype = 3 if int(getattr(prop, "type", 0)) > 2 else int(getattr(prop, "type", 0))
            if rtype < 2:
                continue
            value = int(getattr(prop, "label_value", rtype))
            grouped.setdefault(value, []).append(prop)

        for value in grouped:
            grouped[value].sort(
                key=lambda p: (
                    float(getattr(p, "length", 0.0)),
                    int(np.count_nonzero(getattr(p, "mask", 0))),
                ),
                reverse=True,
            )
        return grouped

    def _leaf_reference_point(self, semantic):
        return self._centroid_rc(np.asarray(semantic) == 1)

    def _root_measurement(self, prop, semantic, primary_path=None):
        if prop is None:
            return {}
        rtype = 3 if int(getattr(prop, "type", 0)) > 2 else int(getattr(prop, "type", 0))
        path = list(getattr(prop, "path", []) or [])
        leaf_point = self._leaf_reference_point(semantic)

        if rtype == 2 and hasattr(self.mainFrame, "_orient_primary_path"):
            oriented = self.mainFrame._orient_primary_path(path, leaf_point)
        elif rtype >= 3 and hasattr(self.mainFrame, "_orient_lateral_path"):
            oriented = self.mainFrame._orient_lateral_path(path, primary_path or [])
        else:
            oriented = path

        if hasattr(self.mainFrame, "_path_length_px"):
            path_px = float(self.mainFrame._path_length_px(oriented))
        else:
            path_px = float(getattr(prop, "length", 0.0))

        geom = (
            self.mainFrame._direction_metrics(oriented)
            if hasattr(self.mainFrame, "_direction_metrics") else None
        )
        if geom:
            euclid_px = float(geom.get("euclidean_px", 0.0))
            direction_deg = float(geom.get("direction_deg", 0.0))
            vertical_deg = float(geom.get("vertical_signed_deg", 0.0))
        else:
            euclid_px = float(getattr(prop, "distance", 0.0))
            direction_deg = ""
            vertical_deg = ""

        insertion = ""
        if rtype >= 3 and primary_path and hasattr(self.mainFrame, "_lateral_insertion_angle"):
            angle = self.mainFrame._lateral_insertion_angle(oriented, primary_path)
            if angle is not None:
                insertion = float(angle)

        straightness = (euclid_px / path_px) if path_px > 0 else 0.0
        tortuosity = (path_px / euclid_px) if euclid_px > 0 else 0.0
        centroid = self._centroid_rc(getattr(prop, "mask", np.zeros_like(semantic)))

        return {
            "path": oriented,
            "PathLengthPx": path_px,
            "EuclideanPx": euclid_px,
            "Straightness": straightness,
            "Tortuosity": tortuosity,
            "DirectionDeg": direction_deg,
            "VerticalAngleDeg": vertical_deg,
            "InsertionAngleDeg": insertion,
            "CentroidRow": centroid[0] if centroid else "",
            "CentroidCol": centroid[1] if centroid else "",
        }

    def _leaf_measurement(self, mask):
        if mask is None:
            return {}
        mask = np.asarray(mask, dtype=bool)
        centroid = self._centroid_rc(mask)
        return {
            "AreaPx": int(mask.sum()),
            "CentroidRow": centroid[0] if centroid else "",
            "CentroidCol": centroid[1] if centroid else "",
        }

    @staticmethod
    def _root_label_values(semantic):
        return sorted(int(v) for v in np.unique(semantic) if int(v) >= 2)

    @staticmethod
    def _short_root_bridge(support, start, target, max_steps=30):
        """Shortest 8-connected bridge inside the union root support."""
        h, w = support.shape
        sr, sc = map(int, start)
        tr, tc = map(int, target)
        if (sr, sc) == (tr, tc):
            return [(sr, sc)]
        if not support[sr, sc] or not support[tr, tc]:
            return None

        q = deque([(sr, sc)])
        parent = {(sr, sc): None}
        depth = {(sr, sc): 0}
        neighbours = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )

        found = None
        while q:
            r0, c0 = q.popleft()
            d0 = depth[(r0, c0)]
            if d0 >= max_steps:
                continue
            for dr, dc in neighbours:
                rr, cc = r0 + dr, c0 + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                key = (rr, cc)
                if key in parent or not support[rr, cc]:
                    continue
                parent[key] = (r0, c0)
                depth[key] = d0 + 1
                if key == (tr, tc):
                    found = key
                    q.clear()
                    break
                q.append(key)

        if found is None:
            return None

        path = []
        cur = found
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def _recover_root_mask(self, semantic, label_value, max_gap_px=12.0, max_foreign_px=12):
        """Reconnect one root label across short overlaps with other roots.

        The semantic PNG can show only one label at a pixel.  At a primary/lateral
        junction, pixels displayed as the other root can split this label into
        multiple components.  This function reconnects only short gaps through
        the union root support (semantic >= 2), preserving shared junction pixels
        without absorbing long portions of neighbouring roots.
        """
        semantic = np.asarray(semantic, dtype=np.uint8)
        seed = semantic == int(label_value)
        support = semantic >= 2

        info = {
            "seed_pixels": int(seed.sum()),
            "shared_pixels_recovered": 0,
            "components_before": 0,
            "components_after": 0,
        }
        if not np.any(seed):
            return seed.copy(), info

        n0, _ = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
        info["components_before"] = max(0, int(n0) - 1)

        recovered = seed.copy()
        max_steps = max(8, int(round(max_gap_px * 2.5)))

        # Iteratively connect the nearest pair of same-label fragments.
        for _ in range(32):
            n, labels = cv2.connectedComponents(recovered.astype(np.uint8), connectivity=8)
            component_count = int(n) - 1
            if component_count <= 1:
                break

            comps = [(labels == i) for i in range(1, n)]
            best = None

            for i in range(len(comps)):
                # Distance and nearest-pixel indices to component i.
                dist, nearest = ndimage.distance_transform_edt(
                    ~comps[i], return_indices=True
                )
                for j in range(i + 1, len(comps)):
                    coords_j = np.argwhere(comps[j])
                    if coords_j.size == 0:
                        continue
                    vals = dist[coords_j[:, 0], coords_j[:, 1]]
                    k = int(np.argmin(vals))
                    gap = float(vals[k])
                    if best is not None and gap >= best[0]:
                        continue
                    p_j = tuple(int(x) for x in coords_j[k])
                    p_i = (
                        int(nearest[0, p_j[0], p_j[1]]),
                        int(nearest[1, p_j[0], p_j[1]]),
                    )
                    best = (gap, p_i, p_j)

            if best is None or best[0] > float(max_gap_px):
                break

            _, p0, p1 = best
            bridge = self._short_root_bridge(
                support, p0, p1, max_steps=max_steps
            )
            if not bridge:
                break

            foreign = sum(
                1 for rr, cc in bridge
                if int(semantic[rr, cc]) >= 2
                and int(semantic[rr, cc]) != int(label_value)
            )
            if foreign > int(max_foreign_px):
                break

            old_components = component_count
            for rr, cc in bridge:
                recovered[rr, cc] = True

            n_new, _ = cv2.connectedComponents(
                recovered.astype(np.uint8), connectivity=8
            )
            if int(n_new) - 1 >= old_components:
                # The bridge did not actually join two fragments.
                break

        n1, _ = cv2.connectedComponents(recovered.astype(np.uint8), connectivity=8)
        info["components_after"] = max(0, int(n1) - 1)
        info["shared_pixels_recovered"] = int(np.logical_and(recovered, ~seed).sum())
        return recovered, info

    @staticmethod
    def _path_length_px(path):
        if path is None or len(path) < 2:
            return 0.0
        pts = np.asarray(path, dtype=float)
        dif = np.diff(pts[:, :2], axis=0)
        return float(np.sqrt(np.sum(dif * dif, axis=1)).sum())

    @staticmethod
    def _dijkstra_farthest(skel, start, want_parent=False):
        """Weighted 8-neighbour geodesic search on a skeleton."""
        h, w = skel.shape
        start = (int(start[0]), int(start[1]))
        dist = {start: 0.0}
        parent = {start: None} if want_parent else None
        heap = [(0.0, start)]
        neighbours = (
            (-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
            (0, -1, 1.0),                       (0, 1, 1.0),
            (1, -1, 2 ** 0.5),  (1, 0, 1.0),  (1, 1, 2 ** 0.5),
        )

        farthest = start
        farthest_dist = 0.0

        while heap:
            d0, (r0, c0) = heapq.heappop(heap)
            if d0 != dist.get((r0, c0)):
                continue
            if d0 > farthest_dist:
                farthest = (r0, c0)
                farthest_dist = d0

            for dr, dc, step in neighbours:
                rr, cc = r0 + dr, c0 + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not skel[rr, cc]:
                    continue
                nd = d0 + step
                key = (rr, cc)
                if nd < dist.get(key, float("inf")):
                    dist[key] = nd
                    if want_parent:
                        parent[key] = (r0, c0)
                    heapq.heappush(heap, (nd, key))

        return farthest, farthest_dist, parent

    def _skeleton_diameter_path(self, binary):
        """Return the longest geodesic path of the largest/longest skeleton component."""
        skel = morphology.skeletonize(np.asarray(binary) > 0)
        if not np.any(skel):
            return [], 0

        n, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
        best_path = []
        best_length = -1.0

        for component_id in range(1, n):
            comp = labels == component_id
            coords = np.argwhere(comp)
            if coords.size == 0:
                continue

            start = tuple(int(x) for x in coords[0])
            a, _, _ = self._dijkstra_farthest(comp, start, want_parent=False)
            b, length, parent = self._dijkstra_farthest(comp, a, want_parent=True)

            path = []
            cur = b
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur) if parent is not None else None
            path.reverse()

            if length > best_length:
                best_length = float(length)
                best_path = path

        return best_path, max(0, int(n) - 1)

    def _measure_root_mask(self, root_mask, semantic, label_value, primary_path=None):
        """Measure one reconstructed root label as one biological root record."""
        root_mask = np.asarray(root_mask, dtype=bool)
        if not np.any(root_mask):
            return {}

        path, skeleton_components = self._skeleton_diameter_path(root_mask)
        leaf_point = self._leaf_reference_point(semantic)

        if int(label_value) == 2 and hasattr(self.mainFrame, "_orient_primary_path"):
            path = self.mainFrame._orient_primary_path(path, leaf_point)
        elif int(label_value) >= 3 and hasattr(self.mainFrame, "_orient_lateral_path"):
            path = self.mainFrame._orient_lateral_path(path, primary_path or [])

        path_px = self._path_length_px(path)

        if hasattr(self.mainFrame, "_direction_metrics") and len(path) >= 2:
            geom = self.mainFrame._direction_metrics(path)
        else:
            geom = None

        if geom:
            euclid_px = float(geom.get("euclidean_px", 0.0))
            direction_deg = float(geom.get("direction_deg", 0.0))
            vertical_deg = float(geom.get("vertical_signed_deg", 0.0))
        elif len(path) >= 2:
            p0 = np.asarray(path[0], dtype=float)
            p1 = np.asarray(path[-1], dtype=float)
            d = p1 - p0
            euclid_px = float(np.linalg.norm(d))
            direction_deg = float(np.degrees(np.arctan2(d[0], d[1])) % 360.0)
            vertical_deg = float(((direction_deg - 90.0 + 180.0) % 360.0) - 180.0)
        else:
            euclid_px = 0.0
            direction_deg = ""
            vertical_deg = ""

        insertion = ""
        if (
            int(label_value) >= 3
            and primary_path
            and hasattr(self.mainFrame, "_lateral_insertion_angle")
            and len(path) >= 2
        ):
            angle = self.mainFrame._lateral_insertion_angle(path, primary_path)
            if angle is not None:
                insertion = float(angle)

        centroid = self._centroid_rc(root_mask)
        straightness = euclid_px / path_px if path_px > 0 else 0.0
        tortuosity = path_px / euclid_px if euclid_px > 0 else 0.0

        return {
            "path": path,
            "AreaPx": int(root_mask.sum()),
            "PathLengthPx": path_px,
            "EuclideanPx": euclid_px,
            "Straightness": straightness,
            "Tortuosity": tortuosity,
            "DirectionDeg": direction_deg,
            "VerticalAngleDeg": vertical_deg,
            "InsertionAngleDeg": insertion,
            "CentroidRow": centroid[0] if centroid else "",
            "CentroidCol": centroid[1] if centroid else "",
            "SkeletonComponents": skeleton_components,
        }

    @staticmethod
    def _abs_error(pred_value, gt_value):
        if pred_value == "" or gt_value == "":
            return ""
        return abs(float(pred_value) - float(gt_value))

    @staticmethod
    def _relative_error_pct(pred_value, gt_value):
        if pred_value == "" or gt_value == "":
            return ""
        gt_value = float(gt_value)
        if abs(gt_value) < 1e-12:
            return ""
        return abs(float(pred_value) - gt_value) / abs(gt_value) * 100.0

    @staticmethod
    def _upsert_csv(path, fieldnames, rows, roi_key):
        existing = []
        if os.path.isfile(path):
            try:
                with open(path, "r", newline="", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        if (
                            row.get("SourceFile") == str(roi_key[0])
                            and row.get("ROINumber") == str(roi_key[1])
                        ):
                            continue
                        existing.append(row)
            except Exception:
                existing = []

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing + rows)

    def _save_validation_results(self, prediction, ground_truth, gt_path=None):
        """Save overlap-aware pixel metrics and one-row-per-root measurements."""
        pred = np.asarray(prediction, dtype=np.uint8)
        gt = np.asarray(ground_truth, dtype=np.uint8)
        if pred.shape != gt.shape:
            raise ValueError("Prediction and GT masks must have exactly the same shape.")

        source_file = os.path.basename(
            str(getattr(self.mainFrame, "imgpath", "")
                or getattr(self.mainFrame, "file_name", ""))
        )
        roi_number = int(self.curRoot) + 1
        roi_total = len(self.mainFrame.RootAll)
        method = str(getattr(self.mainFrame, "processing_method", ""))
        scale = float(getattr(self.mainFrame, "resolution", 1.0))
        unit = str(getattr(self.mainFrame, "unit", "px"))
        timestamp = datetime.now().isoformat(timespec="seconds")
        gt_file = os.path.basename(gt_path) if gt_path else ""

        validation_dir = os.path.join(self.mainFrame.ProjectDir, "validation")
        os.makedirs(validation_dir, exist_ok=True)

        # Reconstruct each root independently so a junction pixel can belong to
        # both the primary root and a lateral root.
        root_values = sorted(
            set(self._root_label_values(pred)) | set(self._root_label_values(gt))
        )
        pred_root_masks = {}
        gt_root_masks = {}
        pred_recovery = {}
        gt_recovery = {}

        for value in root_values:
            pred_root_masks[value], pred_recovery[value] = self._recover_root_mask(
                pred, value
            )
            gt_root_masks[value], gt_recovery[value] = self._recover_root_mask(
                gt, value
            )

        # ---------------- pixel-level validation ----------------
        metric_scopes = [
            ("All organs", "", pred > 0, gt > 0),
            ("Leaf", 1, pred == 1, gt == 1),
            ("All roots", "2+", pred >= 2, gt >= 2),
        ]
        for value in root_values:
            name = "Primary root" if value == 2 else f"Lateral root {value}"
            metric_scopes.append(
                (name, value, pred_root_masks[value], gt_root_masks[value])
            )

        pixel_rows = []
        for scope, label_value, pmask, gmask in metric_scopes:
            row = {
                "SavedAt": timestamp,
                "SourceFile": source_file,
                "ROINumber": roi_number,
                "ROITotal": roi_total,
                "Method": method,
                "GTFile": gt_file,
                "Scope": scope,
                "LabelValue": label_value,
            }
            row.update(self._binary_metrics(pmask, gmask))
            pixel_rows.append(row)

        pixel_fields = [
            "SavedAt", "SourceFile", "ROINumber", "ROITotal", "Method", "GTFile",
            "Scope", "LabelValue", "TP", "FP", "FN", "TN",
            "Dice", "IoU", "Precision", "Recall", "PredPixels", "GTPixels",
        ]
        pixel_csv = os.path.join(validation_dir, "validation_pixel_metrics.csv")
        self._upsert_csv(
            pixel_csv, pixel_fields, pixel_rows, (source_file, roi_number)
        )

        # ---------------- organ-level measurements ----------------
        organ_rows = []

        # Leaf number is intentionally NOT estimated here. Label 1 is exported as
        # one aggregate leaf region for the ROI; manual leaf count can be recorded
        # separately by the user.
        pred_leaf = pred == 1
        gt_leaf = gt == 1
        p_leaf = self._leaf_measurement(pred_leaf) if np.any(pred_leaf) else {}
        g_leaf = self._leaf_measurement(gt_leaf) if np.any(gt_leaf) else {}
        leaf_iou = self._binary_metrics(pred_leaf, gt_leaf)["IoU"]

        pred_leaf_area_px = p_leaf.get("AreaPx", "")
        gt_leaf_area_px = g_leaf.get("AreaPx", "")
        pred_leaf_area = (
            pred_leaf_area_px * scale * scale if pred_leaf_area_px != "" else ""
        )
        gt_leaf_area = (
            gt_leaf_area_px * scale * scale if gt_leaf_area_px != "" else ""
        )

        organ_rows.append({
            "SavedAt": timestamp,
            "SourceFile": source_file,
            "ROINumber": roi_number,
            "ROITotal": roi_total,
            "Method": method,
            "GTFile": gt_file,
            "OrganType": "Leaf",
            "LabelValue": 1,
            "PredExists": int(np.any(pred_leaf)),
            "GTExists": int(np.any(gt_leaf)),
            "MatchIoU": leaf_iou,
            "PredAreaPx": pred_leaf_area_px,
            "GTAreaPx": gt_leaf_area_px,
            "PredArea": pred_leaf_area,
            "GTArea": gt_leaf_area,
            "AreaAbsError": self._abs_error(pred_leaf_area, gt_leaf_area),
            "AreaRelativeErrorPct": self._relative_error_pct(
                pred_leaf_area, gt_leaf_area
            ),
            "PredCentroidRow": p_leaf.get("CentroidRow", ""),
            "PredCentroidCol": p_leaf.get("CentroidCol", ""),
            "GTCentroidRow": g_leaf.get("CentroidRow", ""),
            "GTCentroidCol": g_leaf.get("CentroidCol", ""),
            "LengthUnit": unit,
            "AreaUnit": f"{unit}^2",
            "Note": "Leaf count is manual; this row measures the total leaf region.",
        })

        # Primary paths are measured first because lateral insertion angle uses them.
        pred_primary_meas = {}
        gt_primary_meas = {}
        if 2 in root_values:
            pred_primary_meas = self._measure_root_mask(
                pred_root_masks[2], pred, 2, None
            )
            gt_primary_meas = self._measure_root_mask(
                gt_root_masks[2], gt, 2, None
            )
        pred_primary_path = pred_primary_meas.get("path", [])
        gt_primary_path = gt_primary_meas.get("path", [])

        # Exactly one row for each root label: primary 2, lateral 3,4,5,...
        for value in root_values:
            pmask = pred_root_masks[value]
            gmask = gt_root_masks[value]
            pmeas = self._measure_root_mask(
                pmask, pred, value, pred_primary_path if value >= 3 else None
            ) if np.any(pmask) else {}
            gmeas = self._measure_root_mask(
                gmask, gt, value, gt_primary_path if value >= 3 else None
            ) if np.any(gmask) else {}

            pred_length_px = pmeas.get("PathLengthPx", "")
            gt_length_px = gmeas.get("PathLengthPx", "")
            pred_length = pred_length_px * scale if pred_length_px != "" else ""
            gt_length = gt_length_px * scale if gt_length_px != "" else ""

            pred_euclid_px = pmeas.get("EuclideanPx", "")
            gt_euclid_px = gmeas.get("EuclideanPx", "")
            pred_euclid = pred_euclid_px * scale if pred_euclid_px != "" else ""
            gt_euclid = gt_euclid_px * scale if gt_euclid_px != "" else ""

            root_iou = self._binary_metrics(pmask, gmask)["IoU"]

            organ_rows.append({
                "SavedAt": timestamp,
                "SourceFile": source_file,
                "ROINumber": roi_number,
                "ROITotal": roi_total,
                "Method": method,
                "GTFile": gt_file,
                "OrganType": "Primary root" if value == 2 else "Lateral root",
                "LabelValue": value,
                "PredExists": int(np.any(pmask)),
                "GTExists": int(np.any(gmask)),
                "MatchIoU": root_iou,

                "PredAreaPx": pmeas.get("AreaPx", ""),
                "GTAreaPx": gmeas.get("AreaPx", ""),
                "PredArea": "",
                "GTArea": "",
                "AreaAbsError": "",
                "AreaRelativeErrorPct": "",

                "PredCentroidRow": pmeas.get("CentroidRow", ""),
                "PredCentroidCol": pmeas.get("CentroidCol", ""),
                "GTCentroidRow": gmeas.get("CentroidRow", ""),
                "GTCentroidCol": gmeas.get("CentroidCol", ""),

                "PredPathLengthPx": pred_length_px,
                "GTPathLengthPx": gt_length_px,
                "PredPathLength": pred_length,
                "GTPathLength": gt_length,
                "PathLengthAbsError": self._abs_error(pred_length, gt_length),
                "PathLengthRelativeErrorPct": self._relative_error_pct(
                    pred_length, gt_length
                ),

                "PredEuclideanPx": pred_euclid_px,
                "GTEuclideanPx": gt_euclid_px,
                "PredEuclidean": pred_euclid,
                "GTEuclidean": gt_euclid,

                "PredStraightness": pmeas.get("Straightness", ""),
                "GTStraightness": gmeas.get("Straightness", ""),
                "PredTortuosity": pmeas.get("Tortuosity", ""),
                "GTTortuosity": gmeas.get("Tortuosity", ""),
                "PredDirectionDeg": pmeas.get("DirectionDeg", ""),
                "GTDirectionDeg": gmeas.get("DirectionDeg", ""),
                "PredVerticalAngleDeg": pmeas.get("VerticalAngleDeg", ""),
                "GTVerticalAngleDeg": gmeas.get("VerticalAngleDeg", ""),
                "PredInsertionAngleDeg": pmeas.get("InsertionAngleDeg", ""),
                "GTInsertionAngleDeg": gmeas.get("InsertionAngleDeg", ""),

                "PredComponentsBeforeRecovery": pred_recovery[value]["components_before"],
                "GTComponentsBeforeRecovery": gt_recovery[value]["components_before"],
                "PredComponentsAfterRecovery": pred_recovery[value]["components_after"],
                "GTComponentsAfterRecovery": gt_recovery[value]["components_after"],
                "PredSharedPixelsRecovered": pred_recovery[value]["shared_pixels_recovered"],
                "GTSharedPixelsRecovered": gt_recovery[value]["shared_pixels_recovered"],

                "LengthUnit": unit,
                "AreaUnit": f"{unit}^2",
                "Note": (
                    "One row per root label; short junction gaps are reconstructed "
                    "through the union root support."
                ),
            })

        organ_fields = [
            "SavedAt", "SourceFile", "ROINumber", "ROITotal", "Method", "GTFile",
            "OrganType", "LabelValue", "PredExists", "GTExists", "MatchIoU",

            "PredAreaPx", "GTAreaPx", "PredArea", "GTArea",
            "AreaAbsError", "AreaRelativeErrorPct",
            "PredCentroidRow", "PredCentroidCol",
            "GTCentroidRow", "GTCentroidCol",

            "PredPathLengthPx", "GTPathLengthPx",
            "PredPathLength", "GTPathLength",
            "PathLengthAbsError", "PathLengthRelativeErrorPct",
            "PredEuclideanPx", "GTEuclideanPx",
            "PredEuclidean", "GTEuclidean",
            "PredStraightness", "GTStraightness",
            "PredTortuosity", "GTTortuosity",
            "PredDirectionDeg", "GTDirectionDeg",
            "PredVerticalAngleDeg", "GTVerticalAngleDeg",
            "PredInsertionAngleDeg", "GTInsertionAngleDeg",

            "PredComponentsBeforeRecovery", "GTComponentsBeforeRecovery",
            "PredComponentsAfterRecovery", "GTComponentsAfterRecovery",
            "PredSharedPixelsRecovered", "GTSharedPixelsRecovered",

            "LengthUnit", "AreaUnit", "Note",
        ]
        organ_csv = os.path.join(
            validation_dir, "validation_organ_measurements.csv"
        )
        self._upsert_csv(
            organ_csv, organ_fields, organ_rows, (source_file, roi_number)
        )

        return pixel_csv, organ_csv

    def _apply_semantic_mask(self, semantic):
        semantic = np.asarray(semantic, dtype=np.uint8)
        if semantic.shape != self.OrigImage.shape[:2]:
            raise ValueError("Edited mask size does not match this ROI.")

        if hasattr(self.mainFrame, "_semantic_to_props"):
            props, rlabel = self.mainFrame._semantic_to_props(
                semantic, self.OrigImage
            )
        else:
            props, rlabel = self._component_props(
                semantic, self.mainFrame.imgType, self.OrigImage
            )

        self.eachroot.prop = props
        self.eachroot.roilabel = rlabel
        self.eachroot.semantic_mask = semantic.copy()
        self.eachroot.segmask = (semantic >= 2).astype(np.uint8)
        self.eachroot.bProcessed = True

        # Refresh the project output and all derived measurements immediately.
        if hasattr(self.mainFrame, "RefreshAfterEdit"):
            self.mainFrame.RefreshAfterEdit()
        else:
            self.mainFrame.SaveResult()


def OpenStandaloneEditor(master=None, language="en"):
    editor = launch_standalone(master, language=language)
    if master is not None:
        editor.open_image()
    return editor
