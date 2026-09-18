# -*- coding: utf-8 -*-
"""Standalone/integrated semantic root-mask editor.

Brush/label values:
    0 = Background
    1 = Leaf
    2 = Primary root
    3..255 = individual Lateral roots

Normal mouse-wheel changes the active label value.  Ctrl+wheel zooms and
Shift+wheel changes brush width.

The editor deliberately has no dependency on TensorFlow or Ultralytics, so it can
also be launched as a small drawing/annotation tool by itself.
"""
from __future__ import annotations

import os
import math
import inspect
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage import morphology
from skimage.segmentation import watershed

CLASS_NAMES = {
    0: "Background / Eraser",
    1: "Leaf",
    2: "Primary root",
    3: "Lateral root",
}

# Fixed semantic colors for background, leaf, and primary root.  Lateral-root
# instance labels (3..255) are assigned deterministic, distinct colors so
# neighbouring lateral roots can be visually distinguished while their numeric
# label values remain unchanged.
CLASS_COLORS = {
    0: np.array([45, 45, 45], dtype=np.uint8),
    1: np.array([255, 215, 0], dtype=np.uint8),
    2: np.array([255, 64, 64], dtype=np.uint8),
    # Keep the semantic lateral-root entry for compatibility with gui_main.py,
    # which indexes CLASS_COLORS by root type (0/1/2/3).  The editor itself
    # still uses label_color(value) to give labels 3..255 distinct colours.
    3: np.array([64, 180, 255], dtype=np.uint8),
}
MAX_LABEL_VALUE = 255

EDITOR_JA = {
    "Root annotation editor":"根アノテーションエディター","Current brush label":"現在のブラシラベル",
    "Primary root":"主根","Lateral root":"側根","Leaf":"葉","Background / Eraser":"背景 / 消しゴム",
    "New lateral  [N]":"新しい側根  [N]","Delete":"削除","Delete[Del]":"削除[Del]","Root / Leaf summary":"根 / 葉の概要",
    "Drawing tool":"描画ツール","Brush [B]":"ブラシ [B]","Line [L]":"線 [L]","Fill [F]":"塗りつぶし [F]","Picker [I]":"スポイト [I]",
    "◀ Previous ROI [A]":"◀ 前の ROI [A]","Next ROI [D] ▶":"次の ROI [D] ▶","Brush width (image px)":"ブラシ幅（画像 px）",
    "Display":"表示","Image":"画像","Labels":"ラベル","Labels [Space]":"ラベル [Space]","Difference":"差分","Skeleton":"スケルトン",
    "Label view:":"ラベル表示:","Label view [V]:":"ラベル表示 [V]:",
    "All":"すべて","One label":"1 ラベル","One label [W/S]":"1 ラベル [W/S]","History":"履歴","Undo":"元に戻す","Redo":"やり直す",
    "Persist history locally":"履歴をローカル保存","History is capped at 40 steps.":"履歴は最大 40 ステップです。",
    "Open image":"画像を開く","Open mask":"マスクを開く","Load GT mask":"GT マスクを読み込む",
    "Save mask":"マスクを保存","Save color labels":"カラーラベルを保存","Save validation CSV":"検証 CSV を保存",
    "Clear mask":"マスクを消去","Fit":"ウィンドウに合わせる","Apply to project":"プロジェクトに適用",
    "Apply to project [Ctrl+S]":"プロジェクトに適用 [Ctrl+S]",
    "Ready":"準備完了","Value":"値","Active":"選択中","Lateral roots":"側根",
    "Wheel: label value   Right-click: pick label   Ctrl+wheel: zoom   Shift+wheel: brush width":"ホイール: ラベル値   右クリック: ラベル取得   Ctrl+ホイール: ズーム   Shift+ホイール: ブラシ幅",
    "TP=green   FP=red   FN=blue":"TP=緑   FP=赤   FN=青",
    "Initial":"初期状態","Open image":"画像を開く","Open mask":"マスクを開く",
    "Lateral root labels":"側根ラベル","Delete lateral root":"側根を削除","ROI navigation":"ROI 移動",
    "Validation export":"検証結果の出力","Unsaved edits":"未保存の編集","Apply":"適用",
    "Save labels":"ラベルを保存","Images":"画像","All files":"すべてのファイル",
    "Mask":"マスク","Ground-truth mask":"正解マスク","PNG semantic mask":"PNG セマンティックマスク",
    "PNG color labels":"PNG カラーラベル","Load GT mask for current ROI":"現在の ROI の GT マスクを読み込む",
}


def _lateral_instance_color(value: int) -> np.ndarray:
    """Return a deterministic RGB color for one lateral-root instance.

    Consecutive labels use a golden-angle hue step, which keeps adjacent
    instance IDs visually separated.  Saturation/value are also varied in a
    small cycle so labels remain distinguishable after hue quantisation.
    """
    value = max(3, int(value))
    idx = value - 3

    # Keep label 3 close to the original lateral-root blue for continuity.
    if idx == 0:
        return np.array([64, 180, 255], dtype=np.uint8)

    # Golden-angle stepping distributes consecutive IDs across the colour wheel.
    hue_deg = (210.0 + idx * 137.50776405) % 360.0
    sat_levels = (0.72, 0.88, 1.00)
    val_levels = (1.00, 0.88, 0.76)
    sat = sat_levels[idx % len(sat_levels)]
    val = val_levels[(idx // len(sat_levels)) % len(val_levels)]

    hsv = np.uint8([[[int(round(hue_deg / 2.0)) % 180,
                         int(round(255 * sat)),
                         int(round(255 * val))]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return np.asarray(rgb, dtype=np.uint8)

def label_type(value: int) -> int:
    value = int(value)
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value == 2:
        return 2
    return 3

def label_name(value: int) -> str:
    return CLASS_NAMES[label_type(value)]

def label_color(value: int) -> np.ndarray:
    value = int(value)
    if value >= 3:
        return _lateral_instance_color(value)
    return CLASS_COLORS[label_type(value)]

def rgb_hex(rgb) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def estimate_leaf_count_from_mask(leaf_mask: np.ndarray, image_rgb: Optional[np.ndarray] = None) -> int:
    """Count leaves with the same V2 distance-core/watershed logic as AnalyseRoot.

    This replaces the editor-only legacy estimator so the summary shown in the
    mask editor is consistent with the automatic analysis. If the leaf mask is
    manually edited, the same algorithm is re-run on the edited mask.
    """
    leaf = (np.asarray(leaf_mask) > 0).astype(np.uint8)
    if not np.any(leaf):
        return 0

    if image_rgb is not None and getattr(image_rgb, "ndim", 0) == 3:
        try:
            hsv = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].astype(np.float32)
        except Exception:
            saturation = np.full(leaf.shape, 255.0, dtype=np.float32)
    else:
        saturation = np.full(leaf.shape, 255.0, dtype=np.float32)

    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(leaf, kernel, iterations=3)
    legacy_count = max(cv2.connectedComponentsWithStats(eroded, connectivity=8)[0] - 1, 1)

    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(leaf, connectivity=8)
    next_label = 1

    for cc in range(1, n_cc):
        area = int(cc_stats[cc, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(cc_stats[cc, cv2.CC_STAT_LEFT])
        y = int(cc_stats[cc, cv2.CC_STAT_TOP])
        w = int(cc_stats[cc, cv2.CC_STAT_WIDTH])
        h = int(cc_stats[cc, cv2.CC_STAT_HEIGHT])
        comp = (cc_labels[y:y+h, x:x+w] == cc).astype(np.uint8)

        if area < 220 or min(h, w) < 8:
            next_label += 1
            continue

        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        dist_s = cv2.GaussianBlur(dist, (0, 0), 1.0)
        max_dist = float(dist_s.max())
        if max_dist <= 1.0:
            next_label += 1
            continue

        win = int(np.clip(round(max_dist * 0.8), 7, 25))
        if win % 2 == 0:
            win += 1
        peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win))
        local_max = dist_s >= (cv2.dilate(dist_s, peak_kernel) - 1e-6)
        peak_mask = (local_max & (dist_s >= max(2.0, 0.22 * max_dist)) & (comp > 0)).astype(np.uint8)
        n_peak, peak_labels, _, _ = cv2.connectedComponentsWithStats(peak_mask, connectivity=8)

        sat_crop = saturation[y:y+h, x:x+w]
        comp_sat = sat_crop[comp > 0]
        sat_reference = float(np.median(comp_sat)) if comp_sat.size else 0.0
        yy, xx = np.ogrid[:h, :w]
        candidates = []

        for pk in range(1, n_peak):
            coords = np.argwhere(peak_labels == pk)
            if coords.size == 0:
                continue
            vals = dist_s[coords[:, 0], coords[:, 1]]
            idx = int(np.argmax(vals))
            rr, cc2 = int(coords[idx, 0]), int(coords[idx, 1])
            radius = float(dist_s[rr, cc2])

            rad = max(2, int(round(0.60 * radius)))
            disk = (((yy - rr) ** 2 + (xx - cc2) ** 2) <= rad ** 2) & (comp > 0)
            local_sat = float(np.mean(sat_crop[disk])) if np.any(disk) else float(sat_crop[rr, cc2])

            annulus_r1 = max(3.0, 1.10 * radius)
            annulus_r2 = max(5.0, 2.00 * radius)
            d2 = (yy - rr) ** 2 + (xx - cc2) ** 2
            annulus = ((d2 >= annulus_r1 ** 2) & (d2 <= annulus_r2 ** 2) & (comp > 0))
            local_background = float(np.percentile(dist_s[annulus], 75)) if np.any(annulus) else 0.0
            prominence = radius - local_background
            prominent_peak = prominence >= max(1.0, 0.14 * radius)

            strong_core = radius >= 0.60 * max_dist
            smaller_leaf_core = (
                radius >= 0.42 * max_dist and
                local_sat >= max(20.0, 0.65 * sat_reference) and
                prominent_peak
            )
            if strong_core or smaller_leaf_core:
                candidates.append((radius, rr, cc2, local_sat, prominence))

        candidates.sort(key=lambda t: t[0], reverse=True)

        def suppress_peaks(pool, sep_ratio):
            selected_local = []
            for cand in pool:
                radius, rr, cc2, _, _ = cand
                keep = True
                for old in selected_local:
                    old_r, old_rr, old_cc, _, _ = old
                    min_sep = max(7.0, sep_ratio * (radius + old_r))
                    if math.hypot(rr - old_rr, cc2 - old_cc) < min_sep:
                        keep = False
                        break
                if keep:
                    selected_local.append(cand)
            return selected_local

        selected_first = suppress_peaks(candidates, 0.60)
        selected = selected_first

        if len(selected_first) > 5:
            strict_pool = []
            for cand in candidates:
                radius, rr, cc2, local_sat, prominence = cand
                strict_core = (
                    radius >= 0.50 * max_dist and
                    local_sat >= max(20.0, 0.72 * sat_reference) and
                    prominence >= max(1.2, 0.18 * radius)
                )
                very_strong_core = radius >= 0.68 * max_dist
                if strict_core or very_strong_core:
                    strict_pool.append(cand)
            strict_selected = suppress_peaks(strict_pool, 0.82)
            if 2 <= len(strict_selected) < len(selected_first):
                selected = strict_selected

        if len(selected) <= 1:
            next_label += 1
            continue
        if len(selected) > 6:
            selected = selected[:6]

        markers = np.zeros(comp.shape, dtype=np.int32)
        for marker_id, (_, rr, cc2, _, _) in enumerate(selected, start=1):
            markers[rr, cc2] = marker_id

        ws = watershed(-dist_s, markers=markers, mask=(comp > 0), connectivity=2)
        ws_ids, ws_counts = np.unique(ws[ws > 0], return_counts=True)
        min_region = max(40, int(round(area * 0.025)))
        valid_pairs = [(int(i), int(a)) for i, a in zip(ws_ids, ws_counts) if int(a) >= min_region]

        if len(valid_pairs) >= 3:
            region_areas = np.asarray([a for _, a in valid_pairs], dtype=np.float32)
            median_area = float(np.median(region_areas))
            tiny_cut = max(float(min_region), 0.25 * median_area)
            filtered_pairs = [(i, a) for i, a in valid_pairs if a >= tiny_cut]
            if len(filtered_pairs) >= 2:
                valid_pairs = filtered_pairs

        if len(valid_pairs) <= 1:
            next_label += 1
            continue
        if len(valid_pairs) > 6:
            valid_pairs.sort(key=lambda t: t[1], reverse=True)
            valid_pairs = valid_pairs[:6]

        next_label += len(valid_pairs)

    count = int(next_label - 1)
    fallback = count <= 0 or count > max(8, legacy_count * 3)
    return int(legacy_count if fallback else count)


class SemanticMaskEditor:
    """A multi-class image/mask editor with persistent undo history."""

    def __init__(
        self,
        master: tk.Misc,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        reference_mask: Optional[np.ndarray] = None,
        title: str = "Root annotation editor",
        history_path: Optional[str] = None,
        on_apply: Optional[Callable[[np.ndarray], None]] = None,
        allow_open_image: bool = True,
        resolution: float = 1.0,
        unit: str = "px",
        on_roi_step: Optional[Callable[[int], bool]] = None,
        roi_index: Optional[int] = None,
        roi_count: Optional[int] = None,
        on_save_validation: Optional[Callable[[np.ndarray, np.ndarray, Optional[str]], object]] = None,
        language: str = "en",
    ):
        self._owns_root = False
        self.language = "ja" if language == "ja" else "en"
        self.window = tk.Toplevel(master) if master is not None else tk.Tk()
        self.window.title(self._tr(title) if title == "Root annotation editor" else title)
        self.window.geometry("1320x820")
        self.window.minsize(920, 620)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self.on_apply = on_apply
        self.on_roi_step = on_roi_step
        self.on_save_validation = on_save_validation
        self.roi_index = roi_index
        self.roi_count = roi_count
        self.roi_text_var = tk.StringVar(value="")
        self.reference_path = None
        self.reference_is_ground_truth = False
        self.save_validation_button = None
        self.apply_button = None
        self.allow_open_image = allow_open_image
        self.history_path = history_path
        self.max_history = 40
        self.resolution = float(resolution) if resolution and float(resolution) > 0 else 1.0
        self.unit = str(unit or "px")

        self.image = self._normalize_image(image)
        if self.image is None:
            self.image = np.zeros((720, 960, 3), dtype=np.uint8)
        self.mask = self._normalize_mask(mask, self.image.shape[:2])
        # Overlap-aware representation. ``self.mask`` remains a conventional
        # single-channel projection for backward compatibility, while each
        # non-background label also owns an independent boolean layer.  A
        # primary-root pixel and one or more lateral-root pixels can therefore
        # coexist at the same image coordinate without breaking either root.
        self.label_layers = self._layers_from_projection(self.mask)
        self._sync_projection_from_layers()
        # Snapshot last applied to the RAPID project.  Keep both the portable
        # projection and the overlap layers so an overlap-only edit is detected.
        self._project_mask_baseline = self.mask.copy()
        self._project_layers_baseline = self._copy_layers(self.label_layers)
        self.reference_mask = (
            self._normalize_mask(reference_mask, self.image.shape[:2])
            if reference_mask is not None else None
        )

        self.scale = 1.0
        self._photo = None
        self._dragging = False
        self._stroke_changed = False
        self._line_start = None
        self._last_brush_point = None
        self._dirty = False
        self._redraw_pending = False
        self._history_save_job = None
        # Reusing the scaled base image avoids resizing a large photograph for
        # every mouse-motion event. Only the small semantic mask is updated.
        self._base_display_cache_key = None
        self._base_display_cache = None

        self.tool_var = tk.StringVar(value="brush")
        self.class_var = tk.IntVar(value=2)
        # Last non-background label shown in "One label" mode. The eraser
        # changes the brush value to 0 without hiding the displayed root.
        self.view_label_value = 2
        self.brush_var = tk.IntVar(value=3)
        self.show_image_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=True)
        self.show_diff_var = tk.BooleanVar(value=False)
        self.show_skeleton_var = tk.BooleanVar(value=False)
        # "all" shows every non-background mask value; "active" shows only
        # the currently selected brush/label value.
        self.label_view_var = tk.StringVar(value="all")
        self.persist_history_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value=self._tr("Ready"))
        self.metrics_var = tk.StringVar(value="")
        self.summary_leaf_var = tk.StringVar(value=("葉: --" if self.language == "ja" else "Leaf: --"))
        self.summary_primary_var = tk.StringVar(value=("主根: --" if self.language == "ja" else "Primary root: --"))
        self.summary_lateral_var = tk.StringVar(value=("側根: --" if self.language == "ja" else "Lateral roots: --"))
        self.summary_active_var = tk.StringVar(value=("選択中: --" if self.language == "ja" else "Active: --"))
        self._summary_cache = None

        self.history = []
        self.history_layers = []
        self.history_actions = []
        self.history_index = -1

        self._build_ui()
        self._bind_shortcuts()

        loaded = False
        if self.history_path and os.path.isfile(self.history_path):
            loaded = self._load_history(self.history_path)
            if loaded:
                self.persist_history_var.set(True)
                self.status_var.set("保存された編集履歴を復元しました" if self.language == "ja" else "Saved editing history restored")
        if not loaded:
            self._reset_history("Initial")
        self._update_apply_button_state()

        self._fit_to_window(after_idle=True)
        self._request_redraw()

    @staticmethod
    def _normalize_image(image):
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8).copy()

    @staticmethod
    def _normalize_mask(mask, shape):
        if mask is None:
            return np.zeros(shape, dtype=np.uint8)
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.shape != tuple(shape):
            arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        arr = arr.astype(np.uint8)
        # A conventional 0/255 binary mask means root/background and is mapped
        # to Primary root.  Otherwise preserve label values 0..255 so each
        # value >2 can represent one lateral-root instance.
        vals = np.unique(arr)
        if len(vals) <= 2 and set(int(v) for v in vals).issubset({0, 255}):
            arr = np.where(arr > 0, 2, 0).astype(np.uint8)
        return arr.astype(np.uint8)

    @staticmethod
    def _layers_from_projection(mask):
        """Create independent per-label masks from a legacy 2-D label image.

        A normal PNG cannot encode more than one integer at a pixel.  This
        conversion therefore starts with one layer per visible label.  Once the
        editor is active, layers may overlap and that extra information is kept
        separately from the projection.
        """
        arr = np.asarray(mask, dtype=np.uint8)
        layers = {}
        for value in np.unique(arr):
            value = int(value)
            if value <= 0:
                continue
            layer = (arr == value)
            if np.any(layer):
                layers[value] = layer.astype(bool, copy=True)
        return layers

    @staticmethod
    def _copy_layers(layers):
        return {int(k): np.asarray(v, dtype=bool).copy() for k, v in (layers or {}).items()}

    @staticmethod
    def _layers_equal(a, b):
        a = a or {}
        b = b or {}
        keys = set(int(k) for k in a) | set(int(k) for k in b)
        for key in keys:
            aa = a.get(key)
            bb = b.get(key)
            if aa is None:
                if bb is not None and np.any(bb):
                    return False
                continue
            if bb is None:
                if np.any(aa):
                    return False
                continue
            if np.shape(aa) != np.shape(bb) or not np.array_equal(aa, bb):
                return False
        return True

    def _ensure_label_layer(self, value):
        value = int(value)
        if value <= 0:
            return None
        layer = self.label_layers.get(value)
        if layer is None or layer.shape != self.mask.shape:
            layer = np.zeros(self.mask.shape, dtype=bool)
            self.label_layers[value] = layer
        return layer

    def _label_layer(self, value):
        """Return one label's true overlap-aware layer."""
        value = int(value)
        layer = self.label_layers.get(value)
        if layer is None:
            return np.zeros(self.mask.shape, dtype=bool)
        return np.asarray(layer, dtype=bool)

    def _existing_layer_values(self):
        return sorted(int(v) for v, m in self.label_layers.items()
                      if int(v) > 0 and m is not None and np.any(m))

    def _sync_projection_from_layers(self):
        """Rebuild the legacy 2-D semantic projection from overlap layers.

        Display/storage priority matches RAPID's historical semantic PNG:
        leaf -> primary -> lateral instances.  The projection may show only one
        value at an overlap, but ``label_layers`` preserves every membership.
        """
        shape = self.mask.shape
        projection = np.zeros(shape, dtype=np.uint8)
        for value in (1, 2):
            layer = self.label_layers.get(value)
            if layer is not None:
                projection[np.asarray(layer, dtype=bool)] = value
        for value in sorted(v for v in self.label_layers if int(v) >= 3):
            layer = self.label_layers.get(value)
            if layer is not None:
                projection[np.asarray(layer, dtype=bool)] = min(255, int(value))
        self.mask = projection
        return projection

    def _root_union_layer(self):
        out = np.zeros(self.mask.shape, dtype=bool)
        for value, layer in self.label_layers.items():
            if int(value) >= 2 and layer is not None:
                out |= np.asarray(layer, dtype=bool)
        return out

    def _foreground_union_layer(self):
        out = np.zeros(self.mask.shape, dtype=bool)
        for value, layer in self.label_layers.items():
            if int(value) > 0 and layer is not None:
                out |= np.asarray(layer, dtype=bool)
        return out

    def _overlap_count_map(self):
        count = np.zeros(self.mask.shape, dtype=np.uint16)
        for layer in self.label_layers.values():
            if layer is not None:
                count += np.asarray(layer, dtype=np.uint16)
        return count

    @staticmethod
    def _overlap_sidecar_path(mask_path):
        stem, _ = os.path.splitext(str(mask_path))
        return stem + '.overlap.npz'

    def _save_overlap_sidecar(self, mask_path):
        """Save exact multi-label membership next to the portable PNG."""
        sidecar = self._overlap_sidecar_path(mask_path)
        values = self._existing_layer_values()
        if values:
            stack = np.stack([self._label_layer(v) for v in values], axis=0).astype(np.uint8)
        else:
            stack = np.zeros((0,) + self.mask.shape, dtype=np.uint8)
        np.savez_compressed(
            sidecar,
            labels=np.asarray(values, dtype=np.uint16),
            masks=stack,
            shape=np.asarray(self.mask.shape, dtype=np.int32),
            format_version=np.asarray([1], dtype=np.int16),
        )
        return sidecar

    def _load_overlap_sidecar(self, mask_path):
        """Load overlap layers if a sidecar produced by Save mask exists."""
        sidecar = self._overlap_sidecar_path(mask_path)
        if not os.path.isfile(sidecar):
            return False
        try:
            with np.load(sidecar, allow_pickle=False) as z:
                labels = np.asarray(z['labels']).astype(int).ravel()
                masks = np.asarray(z['masks'])
            if masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(self.mask.shape):
                return False
            if masks.shape[0] != len(labels):
                return False
            layers = {}
            for value, layer in zip(labels, masks):
                value = int(value)
                if 1 <= value <= MAX_LABEL_VALUE and np.any(layer):
                    layers[value] = (np.asarray(layer) > 0)
            self.label_layers = layers
            self._sync_projection_from_layers()
            return True
        except Exception:
            return False

    def _resized_layer(self, value, size):
        layer = self._label_layer(value)
        rw, rh = size
        if (rw, rh) == (self.mask.shape[1], self.mask.shape[0]):
            return layer
        return cv2.resize(layer.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST) > 0

    def _current_label_max(self):
        """Largest label currently present in the mask.

        Values 0, 1 and 2 are the fixed semantic base classes.  Keeping at least
        2 available makes Leaf/Primary-root editing possible even if one of them
        is temporarily absent from the current mask.  New lateral labels are
        created only by the explicit New lateral button.
        """
        if self.mask is None or self.mask.size == 0:
            return 2
        values = self._existing_layer_values()
        return max(2, min(MAX_LABEL_VALUE, max(values) if values else 2))

    def _one_label_edit_target(self):
        """Return the protected label in One-label mode, otherwise None."""
        if getattr(self, "label_view_var", None) is None:
            return None
        if self.label_view_var.get() != "active":
            return None
        value = int(getattr(self, "view_label_value", 0))
        return value if value > 0 else None

    def _apply_draw_mask(self, draw_mask, value):
        """Apply brush/line pixels to independent label layers.

        Non-zero painting is additive: it NEVER removes another label from the
        same pixel.  This is essential at primary/lateral junctions, where one
        pixel may legitimately belong to both roots.  The 2-D ``self.mask`` is
        rebuilt only as a compatibility projection.

        Eraser semantics remain intuitive:
          * All-label view: erase every label under the brush.
          * One-label view: erase only the displayed label.
        """
        draw = np.asarray(draw_mask, dtype=bool)
        if not np.any(draw):
            return False

        value = int(value)
        target = self._one_label_edit_target()
        changed = False

        if value == 0:
            if target is not None and target > 0:
                layer = self._ensure_label_layer(target)
                hit = draw & layer
                if np.any(hit):
                    layer[hit] = False
                    changed = True
            else:
                for label, layer in list(self.label_layers.items()):
                    hit = draw & np.asarray(layer, dtype=bool)
                    if np.any(hit):
                        layer[hit] = False
                        changed = True
        else:
            layer = self._ensure_label_layer(value)
            add = draw & ~layer
            if np.any(add):
                layer[add] = True
                changed = True

        if changed:
            # Drop empty layers, except fixed semantic labels can be recreated on demand.
            self.label_layers = {
                int(v): np.asarray(m, dtype=bool)
                for v, m in self.label_layers.items() if m is not None and np.any(m)
            }
            self._sync_projection_from_layers()
        return changed

    def _existing_root_labels(self):
        """Return root label values currently available in this ROI.

        Leaf (1) is intentionally excluded. Primary root (2) is always available,
        while lateral roots are the actual values >=3 present in the mask. A newly
        created lateral label is also kept available before its first stroke.
        """
        values = {2}
        if self.mask is not None and self.mask.size:
            values.update(v for v in self._existing_layer_values() if int(v) >= 3)

        current = int(self.class_var.get()) if hasattr(self, "class_var") else 2
        if current >= 3:
            values.add(current)

        shown = int(getattr(self, "view_label_value", 2))
        if shown >= 3:
            values.add(shown)

        return sorted(v for v in values if 2 <= v <= MAX_LABEL_VALUE)

    @staticmethod
    def _contrast_text_color(rgb):
        r, g, b = (int(v) for v in rgb)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if luminance >= 150 else "#ffffff"

    def _select_root_brush(self, value):
        """Select one root label and switch to Brush without affecting the mask."""
        value = int(value)
        self.tool_var.set("brush")
        self._set_class(value, allow_new=(value > self._current_label_max()))
        self.status_var.set(self._loc(f"Selected root label {value}", f"根ラベル {value} を選択しました"))
        return "break"

    def _refresh_root_brushes(self):
        """Compatibility no-op: root selection is now indicated by the cursor."""
        return

    def _existing_display_labels(self):
        """Return foreground label values that actually have pixels."""
        if self.mask is None or self.mask.size == 0:
            return []
        return [v for v in self._existing_layer_values() if 1 <= int(v) <= MAX_LABEL_VALUE]

    def _step_root_label(self, delta):
        """W/S cycles the current label only while One label view is active."""
        if self.label_view_var.get() != "active":
            self.status_var.set(self._loc(
                "W/S is available in One label view.",
                "W/S は［1 ラベル］表示で使用できます。"
            ))
            return "break"

        labels = self._existing_display_labels()
        if not labels:
            self.status_var.set(self._loc(
                "No labeled pixels are available.",
                "表示できるラベル付きピクセルがありません。"
            ))
            return "break"

        current = int(getattr(self, "view_label_value", labels[0]))
        if current in labels:
            idx = labels.index(current)
        else:
            idx = min(range(len(labels)), key=lambda i: abs(labels[i] - current))

        new_idx = max(0, min(len(labels) - 1, idx + int(delta)))
        if new_idx == idx:
            direction = "first" if delta < 0 else "last"
            self.status_var.set(self._loc(
                f"Already at the {direction} displayed label ({labels[idx]}).",
                f"これ以上移動できません（表示ラベル {labels[idx]}）。"
            ))
            return "break"

        # Synchronize the selected label and the One-label display, but keep the
        # current drawing tool unchanged.
        self._set_class(labels[new_idx])
        self._request_redraw(immediate=True)
        return "break"

    def _step_roi(self, delta):
        """A/D navigation between YOLO ROIs through the integrated callback."""
        if self.on_roi_step is None:
            self.status_var.set(self._loc("ROI navigation is available only in the integrated editor", "ROI 移動は統合エディターでのみ使用できます"))
            return "break"
        try:
            moved = bool(self.on_roi_step(int(delta)))
        except Exception as exc:
            messagebox.showerror(self._tr("ROI navigation"), str(exc), parent=self.window)
            return "break"

        if not moved:
            edge = "first" if delta < 0 else "last"
            self.status_var.set(self._loc(f"Already at the {edge} YOLO ROI", "これ以上 YOLO ROI を移動できません"))
        return "break"

    def set_roi_position(self, roi_index=None, roi_count=None):
        """Update the compact ROI-position indicator used by A/D navigation."""
        self.roi_index = roi_index
        self.roi_count = roi_count
        if roi_index is None or roi_count is None or int(roi_count) <= 0:
            self.roi_text_var.set("")
        else:
            self.roi_text_var.set(f"ROI {int(roi_index) + 1}/{int(roi_count)}")

    def load_context(
        self,
        image,
        mask,
        reference_mask=None,
        title=None,
        history_path=None,
        roi_index=None,
        roi_count=None,
    ):
        """Load another ROI into the same editor window.

        The caller is responsible for applying/saving dirty edits before switching.
        Each ROI keeps an independent history file.
        """
        self.image = self._normalize_image(image)
        if self.image is None:
            self.image = np.zeros((720, 960, 3), dtype=np.uint8)

        self.mask = self._normalize_mask(mask, self.image.shape[:2])
        self.label_layers = self._layers_from_projection(self.mask)
        self._sync_projection_from_layers()
        self._project_mask_baseline = self.mask.copy()
        self._project_layers_baseline = self._copy_layers(self.label_layers)
        self.reference_mask = (
            self._normalize_mask(reference_mask, self.image.shape[:2])
            if reference_mask is not None else None
        )
        self.reference_path = None
        self.reference_is_ground_truth = False
        self._update_validation_button_state()

        self.history_path = history_path
        self._base_display_cache_key = None
        self._base_display_cache = None
        self._photo = None
        self._dragging = False
        self._stroke_changed = False
        self._line_start = None
        self._last_brush_point = None
        self._dirty = False
        self._redraw_pending = False
        self._summary_cache = None

        if title:
            self.window.title(self._tr(title) if title == "Root annotation editor" else title)
        self.set_roi_position(roi_index, roi_count)

        loaded = False
        if self.history_path and os.path.isfile(self.history_path):
            loaded = self._load_history(self.history_path)
            if loaded:
                self.persist_history_var.set(True)
        if not loaded:
            self._reset_history("Initial")

        # Do not carry a lateral-root label from the previous ROI into this one.
        # Start from Primary root (2); W/S or the colored buttons can then select
        # any lateral label that actually exists in the newly loaded mask.
        self.class_var.set(2)
        self.view_label_value = 2
        self._class_changed()
        self._refresh_root_brushes()
        self._update_apply_button_state()
        self._fit_to_window(after_idle=True)
        self._request_redraw(immediate=True)

    def _build_ui(self):
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)

        left = ttk.Frame(self.window, padding=(8, 8, 6, 8))
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        # History is the only vertically expanding area. Annotation classes and
        # tools stay visible even on a small laptop screen.
        left.rowconfigure(6, weight=1)

        label_box = ttk.LabelFrame(left, text=self._tr("Current brush label"))
        label_box.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        label_box.columnconfigure(1, weight=1)

        self.label_swatch = tk.Label(label_box, width=3, height=2, relief="sunken", bd=1)
        self.label_swatch.grid(row=0, column=0, rowspan=2, padx=(7, 8), pady=7, sticky="ns")
        self.label_value_text = tk.StringVar(value=self._loc("Value 2", "値 2"))
        self.label_type_text = tk.StringVar(value=self._tr("Primary root"))
        ttk.Label(label_box, textvariable=self.label_value_text, font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=1, sticky="sw", pady=(6, 0))
        ttk.Label(label_box, textvariable=self.label_type_text).grid(row=1, column=1, sticky="nw", pady=(0, 5))
        label_actions = ttk.Frame(label_box)
        label_actions.grid(row=0, column=2, rowspan=2, sticky="w", padx=(4, 6), pady=7)
        ttk.Button(label_actions, text=self._tr("New lateral  [N]"), command=self._new_lateral_label).pack(
            side="left", padx=(0, 4))
        ttk.Button(label_actions, text=self._tr("Delete[Del]"), command=self._delete_active_lateral_label).pack(
            side="left")
        ttk.Label(
            label_box,
            text=self._tr("Wheel: label value   Right-click: pick label   Ctrl+wheel: zoom   Shift+wheel: brush width"),
            foreground="#666",
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=7, pady=(0, 6))

        summary_box = ttk.LabelFrame(left, text=self._tr("Root / Leaf summary"))
        summary_box.grid(row=1, column=0, sticky="ew", pady=(0, 7))
        summary_box.columnconfigure(0, weight=1)
        ttk.Label(summary_box, textvariable=self.summary_leaf_var).grid(row=0, column=0, sticky="w", padx=7, pady=(5, 1))
        ttk.Label(summary_box, textvariable=self.summary_primary_var).grid(row=1, column=0, sticky="w", padx=7, pady=1)
        ttk.Label(summary_box, textvariable=self.summary_lateral_var).grid(row=2, column=0, sticky="w", padx=7, pady=1)
        ttk.Separator(summary_box, orient="horizontal").grid(row=3, column=0, sticky="ew", padx=7, pady=3)
        ttk.Label(summary_box, textvariable=self.summary_active_var, foreground="#555").grid(row=4, column=0, sticky="w", padx=7, pady=(1, 5))

        tool_box = ttk.LabelFrame(left, text=self._tr("Drawing tool"))
        tool_box.grid(row=2, column=0, sticky="ew", pady=(0, 7))
        tools = [("Brush [B]", "brush"), ("Line [L]", "line"),
                 ("Fill [F]", "fill"), ("Picker [I]", "picker")]
        for i, (name, val) in enumerate(tools):
            ttk.Radiobutton(tool_box, text=self._tr(name), value=val, variable=self.tool_var).grid(
                row=i // 2, column=i % 2, sticky="w", padx=6, pady=2)

        ttk.Separator(tool_box, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=(4, 3)
        )
        roi_nav = ttk.Frame(tool_box)
        roi_nav.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=(1, 5)
        )
        roi_nav.columnconfigure(0, weight=1)
        roi_nav.columnconfigure(1, weight=0)
        roi_nav.columnconfigure(2, weight=1)

        prev_roi_btn = ttk.Button(
            roi_nav, text=self._tr("◀ Previous ROI [A]"), command=lambda: self._step_roi(-1)
        )
        prev_roi_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        ttk.Label(
            roi_nav, textvariable=self.roi_text_var, width=9, anchor="center"
        ).grid(row=0, column=1, padx=2)

        next_roi_btn = ttk.Button(
            roi_nav, text=self._tr("Next ROI [D] ▶"), command=lambda: self._step_roi(1)
        )
        next_roi_btn.grid(row=0, column=2, sticky="ew", padx=(3, 0))

        if self.on_roi_step is None:
            prev_roi_btn.configure(state="disabled")
            next_roi_btn.configure(state="disabled")

        brush_box = ttk.Frame(left)
        brush_box.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(brush_box, text=self._tr("Brush width (image px)")).pack(side="left")
        ttk.Spinbox(brush_box, from_=1, to=200, textvariable=self.brush_var, width=6).pack(side="right")

        display_box = ttk.LabelFrame(left, text=self._tr("Display"))
        display_box.grid(row=4, column=0, sticky="ew", pady=(0, 7))
        ttk.Checkbutton(display_box, text=self._tr("Image"), variable=self.show_image_var,
                        command=self._request_redraw).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Checkbutton(display_box, text=self._tr("Labels [Space]"), variable=self.show_labels_var,
                        command=self._request_redraw).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Checkbutton(display_box, text=self._tr("Difference"), variable=self.show_diff_var,
                        command=self._request_redraw).grid(row=1, column=0, sticky="w", padx=5)
        ttk.Checkbutton(display_box, text=self._tr("Skeleton"), variable=self.show_skeleton_var,
                        command=self._request_redraw).grid(row=1, column=1, sticky="w", padx=5)

        view_row = ttk.Frame(display_box)
        view_row.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=(2, 1))
        ttk.Label(view_row, text=self._tr("Label view [V]:")).pack(side="left")
        ttk.Radiobutton(
            view_row, text=self._tr("All"), value="all", variable=self.label_view_var,
            command=self._request_redraw
        ).pack(side="left", padx=(6, 2))
        ttk.Radiobutton(
            view_row, text=self._tr("One label [W/S]"), value="active", variable=self.label_view_var,
            command=self._request_redraw
        ).pack(side="left", padx=(2, 0))

        ttk.Label(display_box, text=self._tr("TP=green   FP=red   FN=blue"), foreground="#666").grid(
            row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(2, 3))

        ttk.Label(left, text=self._tr("History"), font=("TkDefaultFont", 10, "bold")).grid(
            row=5, column=0, sticky="w")
        hist_frame = ttk.Frame(left)
        hist_frame.grid(row=6, column=0, sticky="nsew", pady=(4, 5))
        hist_frame.rowconfigure(0, weight=1)
        hist_frame.columnconfigure(0, weight=1)
        self.history_list = tk.Listbox(hist_frame, width=29, height=8, exportselection=False)
        self.history_list.grid(row=0, column=0, sticky="nsew")
        hs = ttk.Scrollbar(hist_frame, orient="vertical", command=self.history_list.yview)
        hs.grid(row=0, column=1, sticky="ns")
        self.history_list.configure(yscrollcommand=hs.set)
        self.history_list.bind("<<ListboxSelect>>", self._history_selected)

        hist_btns = ttk.Frame(left)
        hist_btns.grid(row=7, column=0, sticky="ew")
        ttk.Button(hist_btns, text=self._tr("Undo"), command=self.undo).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(hist_btns, text=self._tr("Redo"), command=self.redo).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Checkbutton(
            left, text=self._tr("Persist history locally"), variable=self.persist_history_var,
            command=self._persist_toggle,
        ).grid(row=8, column=0, sticky="w", pady=(6, 0))
        ttk.Label(left, text=self._tr("History is capped at 40 steps."), foreground="#666").grid(
            row=9, column=0, sticky="w")

        main = ttk.Frame(self.window)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(main, padding=(4, 6))
        toolbar.grid(row=0, column=0, sticky="ew")
        if self.allow_open_image:
            ttk.Button(toolbar, text=self._tr("Open image"), command=self.open_image).pack(side="left", padx=2)
        ttk.Button(toolbar, text=self._tr("Open mask"), command=self.open_mask).pack(side="left", padx=2)
        ttk.Button(toolbar, text=self._tr("Load GT mask"), command=self.load_reference).pack(side="left", padx=2)
        ttk.Button(toolbar, text=self._tr("Save mask"), command=self.save_mask).pack(side="left", padx=2)
        ttk.Button(toolbar, text=self._tr("Save color labels"), command=self.save_color_labels).pack(side="left", padx=2)

        if self.on_save_validation is not None:
            self.save_validation_button = ttk.Button(
                toolbar,
                text=self._tr("Save validation CSV"),
                command=self.save_validation_results,
                state="disabled",
            )
            self.save_validation_button.pack(side="left", padx=(8, 2))
        ttk.Button(toolbar, text=self._tr("Clear mask"), command=self.clear_mask).pack(side="left", padx=(10, 2))
        ttk.Button(toolbar, text=self._tr("Fit"), command=self.fit_to_window).pack(side="left", padx=(12, 2))
        ttk.Button(toolbar, text="100%", command=self.actual_size).pack(side="left", padx=2)

        if self.on_apply is not None:
            self.apply_button = ttk.Button(
                toolbar, text=self._tr("Apply to project [Ctrl+S]"),
                command=self.apply_to_project, state="disabled"
            )
            self.apply_button.pack(side="right", padx=4)

        canvas_frame = ttk.Frame(main)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame, background="#333", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        xbar.grid(row=1, column=0, sticky="ew")
        ybar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

        bottom = ttk.Frame(main, padding=(6, 3))
        bottom.grid(row=2, column=0, sticky="ew")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")
        ttk.Label(bottom, textvariable=self.metrics_var).pack(side="right")

        self.canvas.bind("<ButtonPress-1>", self._left_press)
        self.canvas.bind("<B1-Motion>", self._left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._left_release)
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<ButtonPress-3>", self._right_pick_label)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", self._wheel)
        self.canvas.bind("<Button-5>", self._wheel)
        self.canvas.bind("<Motion>", self._update_colored_cursor)
        self.canvas.bind("<Leave>", self._hide_colored_cursor)
        self.canvas.bind("<Configure>", lambda e: self._request_redraw())

        self.set_roi_position(self.roi_index, self.roi_count)
        self._class_changed()

    def _tr(self, text: str) -> str:
        return EDITOR_JA.get(text, text) if self.language == "ja" else text

    def _loc(self, en: str, ja: str) -> str:
        return ja if self.language == "ja" else en

    def _class_changed(self):
        """Refresh the compact current-label indicator."""
        value = max(0, min(MAX_LABEL_VALUE, int(self.class_var.get())))
        if value != int(self.class_var.get()):
            self.class_var.set(value)
        kind = label_name(value)
        color = label_color(value)
        self.label_value_text.set(self._loc(f"Value {value}", f"値 {value}"))
        self.label_type_text.set(self._tr(kind))
        self.label_swatch.configure(bg=rgb_hex(color))
        self.status_var.set(self._loc(f"Active label {value}: {kind}", f"選択ラベル {value}: {self._tr(kind)}"))
        # Nonzero selections also become the label shown in One-label mode.
        # Selecting the eraser (0) changes only the brush, not the displayed root.
        if value > 0:
            self.view_label_value = value
        self._update_annotation_summary(recompute=False)
        self._refresh_root_brushes()
        if hasattr(self, "canvas"):
            self._draw_colored_cursor()
        if getattr(self, "label_view_var", None) is not None and self.label_view_var.get() == "active":
            self._request_redraw()

    def _bind_shortcuts(self):
        self.window.bind("<Control-z>", lambda e: self.undo())
        self.window.bind("<Control-y>", lambda e: self.redo())
        self.window.bind("<Control-Shift-Z>", lambda e: self.redo())
        self.window.bind("<Control-s>", self._ctrl_s_shortcut)
        self.window.bind("<Key-b>", lambda e: self.tool_var.set("brush"))
        self.window.bind("<Key-l>", lambda e: self.tool_var.set("line"))
        self.window.bind("<Key-f>", lambda e: self.tool_var.set("fill"))
        self.window.bind("<Key-i>", lambda e: self.tool_var.set("picker"))
        self.window.bind("<Key-e>", lambda e: self._set_class(0))
        self.window.bind("<Key-0>", lambda e: self._set_class(0))
        self.window.bind("<Key-1>", lambda e: self._set_class(1))
        self.window.bind("<Key-2>", lambda e: self._set_class(2))
        self.window.bind("<Key-3>", lambda e: self._set_class(3))
        self.window.bind("<Key-n>", lambda e: self._new_lateral_label())
        self.window.bind("<Delete>", self._delete_shortcut)
        self.window.bind("<space>", self._toggle_labels_shortcut)
        self.window.bind("<Key-v>", self._toggle_label_view_shortcut)
        # ROI navigation: A = previous YOLO box, D = next YOLO box.
        self.window.bind("<Key-a>", lambda e: self._step_roi(-1))
        self.window.bind("<Key-d>", lambda e: self._step_roi(1))

        # In One label view: W = previous visible label, S = next visible label.
        self.window.bind("<Key-w>", lambda e: self._step_root_label(-1))
        self.window.bind("<Key-s>", lambda e: self._step_root_label(1))

    @staticmethod
    def _shortcut_in_text_input(event):
        """Do not steal shortcut keys from text-entry controls."""
        widget = getattr(event, "widget", None)
        return widget is not None and isinstance(
            widget, (tk.Entry, ttk.Entry, tk.Text, tk.Spinbox,
                     ttk.Spinbox, ttk.Combobox)
        )

    def _toggle_labels_shortcut(self, event=None):
        """Space toggles Display -> Labels."""
        if event is not None and self._shortcut_in_text_input(event):
            return None
        self.show_labels_var.set(not bool(self.show_labels_var.get()))
        self._request_redraw(immediate=True)
        return "break"

    def _toggle_label_view_shortcut(self, event=None):
        """V toggles Label view between All and One label."""
        if event is not None and self._shortcut_in_text_input(event):
            return None
        new_mode = "active" if self.label_view_var.get() != "active" else "all"
        if new_mode == "active":
            labels = self._existing_display_labels()
            shown = int(getattr(self, "view_label_value", -1))
            if labels and shown not in labels:
                current = int(self.class_var.get())
                self.view_label_value = current if current in labels else labels[0]
        self.label_view_var.set(new_mode)
        self._request_redraw(immediate=True)
        return "break"

    def _ctrl_s_shortcut(self, event=None):
        """Ctrl+S applies project edits; standalone mode keeps Save mask."""
        if event is not None and self._shortcut_in_text_input(event):
            return None
        if self.on_apply is not None:
            if self._project_has_pending_edits():
                self.apply_to_project()
            else:
                self.status_var.set(self._loc(
                    "No label changes to apply.",
                    "適用するラベル変更はありません。"
                ))
        else:
            self.save_mask()
        return "break"

    def _set_class(self, value, allow_new=False):
        upper = MAX_LABEL_VALUE if allow_new else self._current_label_max()
        self.class_var.set(max(0, min(upper, int(value))))
        self._class_changed()

    def _new_lateral_label(self):
        current_max = self._current_label_max()
        next_value = max(3, current_max + 1)
        if next_value > MAX_LABEL_VALUE:
            messagebox.showwarning(
                self._tr("Lateral root labels"),
                self._loc(f"The editor supports label values up to {MAX_LABEL_VALUE}.", f"このエディターで使用できるラベル値は {MAX_LABEL_VALUE} までです。"),
                parent=self.window,
            )
            next_value = MAX_LABEL_VALUE
        self._set_class(next_value, allow_new=True)
    def _delete_shortcut(self, event=None):
        """Delete the active lateral root without affecting text-entry controls."""
        if event is not None and self._shortcut_in_text_input(event):
            return None
        self._delete_active_lateral_label()
        return "break"
    def _delete_active_lateral_label(self):
        """Delete the active lateral layer and compact IDs to 3, 4, 5, ..."""
        value = int(self.class_var.get())
        if value < 3:
            messagebox.showinfo(
                self._tr("Delete lateral root"),
                self._loc("Select a lateral-root label (value 3 or higher) before deleting.", "削除する前に、値 3 以上の側根ラベルを選択してください。"),
                parent=self.window,
            )
            return

        layer = self.label_layers.get(value)
        count = int(np.count_nonzero(layer)) if layer is not None else 0
        if count <= 0:
            self.status_var.set(self._loc(f"Label {value} has no pixels to delete", f"ラベル {value} には削除するピクセルがありません"))
            return

        answer = messagebox.askyesno(
            self._tr("Delete lateral root"),
            self._loc(
                f"Delete lateral-root label {value} ({count} pixels)?\n\nHigher lateral-root labels will be renumbered automatically. The operation can be restored with Undo.",
                f"側根ラベル {value}（{count} ピクセル）を削除しますか？\n\n大きい側根ラベルは自動的に連番へ振り直されます。この操作は［元に戻す］で復元できます。"
            ),
            parent=self.window,
        )
        if not answer:
            return

        self.label_layers.pop(value, None)
        # Compact lateral labels while preserving every boolean layer, including overlaps.
        lateral_old = sorted(v for v in self.label_layers if int(v) >= 3 and np.any(self.label_layers[v]))
        compacted = {v: m for v, m in self.label_layers.items() if int(v) < 3 and np.any(m)}
        for new_value, old_value in enumerate(lateral_old, start=3):
            compacted[new_value] = np.asarray(self.label_layers[old_value], dtype=bool).copy()
        self.label_layers = compacted
        self._sync_projection_from_layers()

        lateral_values = sorted(v for v in self.label_layers if int(v) >= 3)
        new_active = min(max(value, 3), lateral_values[-1]) if lateral_values else 2
        self.class_var.set(new_active)
        self.view_label_value = new_active
        self._commit_history(f"Delete lateral {value} + compact labels")
        self._class_changed()
        self._request_redraw(immediate=True)

        last_value = lateral_values[-1] if lateral_values else 2
        self.status_var.set(self._loc(
            f"Deleted lateral-root label {value}; labels are now continuous through {last_value}",
            f"側根ラベル {value} を削除しました。ラベルは {last_value} まで連番に整理されました"
        ))

    # ---------------------------- history ----------------------------
    def _reset_history(self, action="Initial"):
        self.history = [self.mask.copy()]
        self.history_layers = [self._copy_layers(self.label_layers)]
        self.history_actions = [action]
        self.history_index = 0
        self._refresh_history_list()
        self._mark_summary_changed()

    def _commit_history(self, action):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
            self.history_layers = self.history_layers[: self.history_index + 1]
            self.history_actions = self.history_actions[: self.history_index + 1]
        same_projection = bool(self.history and np.array_equal(self.mask, self.history[-1]))
        same_layers = bool(self.history_layers and self._layers_equal(self.label_layers, self.history_layers[-1]))
        if same_projection and same_layers:
            return
        self.history.append(self.mask.copy())
        self.history_layers.append(self._copy_layers(self.label_layers))
        self.history_actions.append(action)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_layers.pop(0)
            self.history_actions.pop(0)
        self.history_index = len(self.history) - 1
        self._dirty = True
        self._refresh_history_list()
        self._mark_summary_changed()
        if self.persist_history_var.get():
            self._schedule_history_save()

    def _history_display_text(self, action):
        if self.language != "ja":
            return action
        exact = {
            "Initial": "初期状態",
            "Open image": "画像を開く",
            "Open mask": "マスクを開く",
            "Clear mask": "マスクを消去",
        }
        if action in exact:
            return exact[action]
        replacements = (
            ("Delete lateral ", "側根を削除 "),
            ("Fill ", "塗りつぶし "),
            ("Line ", "線 "),
            ("Brush ", "ブラシ "),
        )
        display = action
        for prefix, translated in replacements:
            if action.startswith(prefix):
                display = translated + action[len(prefix):]
                break
        for src, dst in (("Primary root", "主根"), ("Lateral root", "側根"),
                         ("Leaf", "葉"), ("Background / Eraser", "背景 / 消しゴム"),
                         ("compact labels", "ラベルを連番化")):
            display = display.replace(src, dst)
        return display

    def _refresh_history_list(self):
        self.history_list.delete(0, tk.END)
        for i, action in enumerate(self.history_actions):
            self.history_list.insert(tk.END, f"{i:02d}  {self._history_display_text(action)}")
        if self.history_index >= 0:
            self.history_list.selection_clear(0, tk.END)
            self.history_list.selection_set(self.history_index)
            self.history_list.see(self.history_index)

    def _history_selected(self, _event=None):
        sel = self.history_list.curselection()
        if not sel:
            return
        self._goto_history(sel[0])

    def _goto_history(self, index):
        if not (0 <= index < len(self.history)):
            return
        self.history_index = int(index)
        self.mask = self.history[self.history_index].copy()
        if self.history_index < len(self.history_layers):
            self.label_layers = self._copy_layers(self.history_layers[self.history_index])
        else:
            self.label_layers = self._layers_from_projection(self.mask)
        self._sync_projection_from_layers()
        self._dirty = True
        self._refresh_history_list()
        self._mark_summary_changed()
        self._request_redraw()

    def undo(self):
        if self.history_index > 0:
            self._goto_history(self.history_index - 1)

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self._goto_history(self.history_index + 1)

    def _persist_toggle(self):
        if self.persist_history_var.get():
            if not self.history_path:
                self.history_path = self._default_history_path()
            self._save_history_silent()
        self.status_var.set(self._loc(
            f"History persistence {'enabled' if self.persist_history_var.get() else 'disabled'}",
            f"履歴のローカル保存を{'有効' if self.persist_history_var.get() else '無効'}にしました"
        ))

    def _default_history_path(self):
        return os.path.join(os.getcwd(), "root_editor_history.npz")

    def _schedule_history_save(self):
        if self._history_save_job is not None:
            try:
                self.window.after_cancel(self._history_save_job)
            except Exception:
                pass
        self._history_save_job = self.window.after(1200, self._save_history_silent)

    def _save_history_silent(self):
        self._history_save_job = None
        if not self.history_path or not self.history:
            return False
        try:
            folder = os.path.dirname(os.path.abspath(self.history_path))
            os.makedirs(folder, exist_ok=True)
            payload = {
                'states': np.stack(self.history, axis=0).astype(np.uint8),
                'actions': np.asarray(self.history_actions, dtype='U80'),
                'index': np.asarray([self.history_index], dtype=np.int32),
                'overlap_history_version': np.asarray([1], dtype=np.int16),
            }
            for i, layers in enumerate(self.history_layers):
                values = sorted(v for v, m in layers.items() if np.any(m))
                payload[f'layer_labels_{i}'] = np.asarray(values, dtype=np.uint16)
                payload[f'layer_masks_{i}'] = (
                    np.stack([layers[v] for v in values], axis=0).astype(np.uint8)
                    if values else np.zeros((0,) + self.mask.shape, dtype=np.uint8)
                )
            np.savez_compressed(self.history_path, **payload)
            return True
        except Exception as exc:
            self.status_var.set(self._loc(f"History save failed: {exc}", f"履歴の保存に失敗しました: {exc}"))
            return False

    def _load_history(self, path):
        try:
            with np.load(path, allow_pickle=False) as z:
                states = z['states']
                actions = z['actions'].astype(str).tolist()
                idx = int(z['index'][0]) if 'index' in z else len(states) - 1
                if states.ndim != 3 or states.shape[1:] != self.mask.shape:
                    return False
                all_states = [st.astype(np.uint8).copy() for st in states]
                all_layers = []
                for i, st in enumerate(all_states):
                    lk = f'layer_labels_{i}'
                    mk = f'layer_masks_{i}'
                    if lk in z and mk in z:
                        labels = np.asarray(z[lk]).astype(int).ravel()
                        masks = np.asarray(z[mk])
                        layers = {}
                        if masks.ndim == 3 and masks.shape[0] == len(labels) and tuple(masks.shape[1:]) == tuple(self.mask.shape):
                            for value, layer in zip(labels, masks):
                                if 1 <= int(value) <= MAX_LABEL_VALUE and np.any(layer):
                                    layers[int(value)] = np.asarray(layer) > 0
                        all_layers.append(layers if layers else self._layers_from_projection(st))
                    else:
                        all_layers.append(self._layers_from_projection(st))

            start = max(0, len(all_states) - self.max_history)
            self.history = all_states[start:]
            self.history_layers = all_layers[start:]
            self.history_actions = actions[start:start + len(self.history)]
            self.history_index = max(0, min(idx - start, len(self.history) - 1))
            self.mask = self.history[self.history_index].copy()
            self.label_layers = self._copy_layers(self.history_layers[self.history_index])
            self._sync_projection_from_layers()
            self._refresh_history_list()
            self._mark_summary_changed()
            return True
        except Exception:
            return False

    # ---------------------------- drawing ----------------------------
    def _draw_colored_cursor(self):
        """Draw a small canvas cursor using the active label color.

        For Brush/Line, the circle approximately follows the current brush width.
        Eraser is shown as a white circle with an X. Other tools use a compact
        colored crosshair. This does not alter the mask.
        """
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("paint_cursor")

        pos = getattr(self, "_cursor_canvas_pos", None)
        if pos is None:
            return

        cx, cy = pos
        value = int(self.class_var.get())
        tool = self.tool_var.get()

        if value == 0:
            outline = "#ffffff"
        else:
            outline = rgb_hex(label_color(value))

        if tool in ("brush", "line"):
            radius = max(4.0, float(self.brush_var.get()) * float(self.scale) / 2.0)
        else:
            radius = 6.0

        # Outer ring shows active label color and approximate brush footprint.
        self.canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            outline=outline, width=2, tags=("paint_cursor",)
        )

        if value == 0:
            # Make the eraser unmistakable.
            d = max(3.0, radius * 0.55)
            self.canvas.create_line(
                cx - d, cy - d, cx + d, cy + d,
                fill=outline, width=2, tags=("paint_cursor",)
            )
            self.canvas.create_line(
                cx - d, cy + d, cx + d, cy - d,
                fill=outline, width=2, tags=("paint_cursor",)
            )
        else:
            # Small center cross improves precision at thin root junctions.
            d = min(4.0, max(2.0, radius * 0.35))
            self.canvas.create_line(
                cx - d, cy, cx + d, cy,
                fill=outline, width=1, tags=("paint_cursor",)
            )
            self.canvas.create_line(
                cx, cy - d, cx, cy + d,
                fill=outline, width=1, tags=("paint_cursor",)
            )

        self.canvas.tag_raise("paint_cursor")

    def _update_colored_cursor(self, event):
        """Move the colored cursor to the current image position."""
        if self._canvas_to_image(event) is None:
            self._cursor_canvas_pos = None
            if hasattr(self, "canvas"):
                self.canvas.delete("paint_cursor")
            return

        self._cursor_canvas_pos = (
            float(self.canvas.canvasx(event.x)),
            float(self.canvas.canvasy(event.y)),
        )
        self._draw_colored_cursor()

    def _hide_colored_cursor(self, _event=None):
        self._cursor_canvas_pos = None
        if hasattr(self, "canvas"):
            self.canvas.delete("paint_cursor")

    def _canvas_to_image(self, event):
        if self.scale <= 0:
            return None
        x = int(self.canvas.canvasx(event.x) / self.scale)
        y = int(self.canvas.canvasy(event.y) / self.scale)
        h, w = self.mask.shape
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return None

    def _left_press(self, event):
        self._update_colored_cursor(event)
        pt = self._canvas_to_image(event)
        if pt is None:
            return
        x, y = pt
        tool = self.tool_var.get()
        if tool == "picker":
            value = int(self.mask[y, x])
            self._set_class(value)
            self.status_var.set(self._loc(f"Picked label {value}: {label_name(value)}", f"ラベル {value} を取得: {self._tr(label_name(value))}"))
            return
        if tool == "fill":
            old = int(self.mask[y, x])
            new = int(self.class_var.get())
            # Flood the connected region visible in the compatibility projection,
            # then add/remove membership in the overlap-aware layers.
            same = (self.mask == old).astype(np.uint8)
            ffmask = np.zeros((same.shape[0] + 2, same.shape[1] + 2), np.uint8)
            cv2.floodFill(same, ffmask, (x, y), 2, flags=4)
            region = same == 2
            if self._apply_draw_mask(region, new):
                self._commit_history(f"Fill {new} → {label_name(new)}")
                self._request_redraw()
            return
        if tool == "line":
            self._line_start = (x, y)
            self._dragging = True
            return

        self._dragging = True
        self._stroke_changed = False
        self._last_brush_point = (x, y)
        self._paint_segment((x, y), (x, y))

    def _left_drag(self, event):
        self._update_colored_cursor(event)
        if not self._dragging or self.tool_var.get() != "brush":
            return
        pt = self._canvas_to_image(event)
        if pt is None:
            # Do not bridge across a trip outside the image/canvas.
            self._last_brush_point = None
            return
        if self._last_brush_point is None:
            self._paint_segment(pt, pt)
        elif pt != self._last_brush_point:
            # Crucial for fast mouse movement: connect event samples with a
            # rasterized segment, so missing Tk motion events cannot create gaps.
            self._paint_segment(self._last_brush_point, pt)
        self._last_brush_point = pt

    def _left_release(self, event):
        self._update_colored_cursor(event)
        if not self._dragging:
            return
        tool = self.tool_var.get()
        if tool == "line" and self._line_start is not None:
            pt = self._canvas_to_image(event)
            if pt is not None:
                value = int(self.class_var.get())
                thickness = max(1, int(self.brush_var.get()))
                draw = np.zeros(self.mask.shape, dtype=np.uint8)
                cv2.line(draw, self._line_start, pt, 1,
                         thickness=thickness, lineType=cv2.LINE_8)
                # Round end caps make Line and Brush visually/structurally match.
                radius = max(0, thickness // 2)
                if radius > 0:
                    cv2.circle(draw, self._line_start, radius, 1, -1, lineType=cv2.LINE_8)
                    cv2.circle(draw, pt, radius, 1, -1, lineType=cv2.LINE_8)
                if self._apply_draw_mask(draw > 0, value):
                    self._commit_history(f"Line {value} → {label_name(value)}")
        elif tool == "brush":
            pt = self._canvas_to_image(event)
            if pt is not None and self._last_brush_point is not None and pt != self._last_brush_point:
                self._paint_segment(self._last_brush_point, pt, request_redraw=False)
            if self._stroke_changed:
                self._commit_history(f"Brush {int(self.class_var.get())} → {label_name(int(self.class_var.get()))}")
        self._dragging = False
        self._line_start = None
        self._last_brush_point = None
        self._stroke_changed = False
        self._request_redraw(immediate=True)

    def _paint_segment(self, p0, p1, request_redraw=True):
        """Paint one continuous semantic segment in image coordinates.

        Tk may emit sparse <B1-Motion> events when the mouse moves quickly.
        Connecting consecutive samples with cv2.line makes the stored mask
        continuous regardless of UI event rate.
        """
        value = int(self.class_var.get())
        thickness = max(1, int(self.brush_var.get()))
        draw = np.zeros(self.mask.shape, dtype=np.uint8)
        if p0 == p1:
            radius = max(0, thickness // 2)
            if radius <= 0:
                draw[p0[1], p0[0]] = 1
            else:
                cv2.circle(draw, p0, radius, 1, thickness=-1, lineType=cv2.LINE_8)
        else:
            cv2.line(draw, p0, p1, 1, thickness=thickness, lineType=cv2.LINE_8)
            # Round cap at the current sample to avoid tiny diagonal pinholes.
            radius = max(0, thickness // 2)
            if radius > 0:
                cv2.circle(draw, p1, radius, 1, thickness=-1, lineType=cv2.LINE_8)
        if self._apply_draw_mask(draw > 0, value):
            self._stroke_changed = True
        if request_redraw:
            self._request_redraw()

    def _right_pick_label(self, event):
        """Pick the semantic label under the cursor with one right-click.

        This is a one-shot picker: the current drawing tool is preserved, so a
        user can right-click a lateral root and immediately continue painting
        with that exact instance value.  Middle-mouse drag remains available for
        panning.
        """
        pt = self._canvas_to_image(event)
        if pt is None:
            return "break"
        x, y = pt
        value = int(self.mask[y, x])
        self._set_class(value)
        self.status_var.set(self._loc(
            f"Picked label {value}: {label_name(value)} (right-click)",
            f"ラベル {value} を取得: {self._tr(label_name(value))}（右クリック）"
        ))
        return "break"

    def _pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _wheel(self, event):
        direction = 0
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            direction = 1
        elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            direction = -1
        if direction == 0:
            return "break"

        state = int(getattr(event, "state", 0) or 0)
        ctrl = bool(state & 0x0004)
        shift = bool(state & 0x0001)

        if ctrl:
            old_scale = self.scale
            self.scale *= 1.2 if direction > 0 else (1 / 1.2)
            self.scale = max(0.05, min(self.scale, 8.0))
            if abs(self.scale - old_scale) >= 1e-9:
                self._request_redraw()
                self._update_colored_cursor(event)
            return "break"

        if shift:
            self.brush_var.set(max(1, min(200, int(self.brush_var.get()) + direction)))
            self._request_redraw()
            self._update_colored_cursor(event)
            return "break"

        # Normal wheel changes annotation identity, but cannot scroll beyond
        # the largest label that actually exists in the current mask.
        old_value = int(self.class_var.get())
        self._set_class(old_value + direction)
        if direction > 0 and int(self.class_var.get()) == old_value:
            self.status_var.set(self._loc(
                f"Maximum existing label is {self._current_label_max()}; use New lateral to create a new instance",
                f"現在の最大ラベルは {self._current_label_max()} です。［新しい側根］で新しいインスタンスを作成してください"
            ))
        self._request_redraw()
        self._update_colored_cursor(event)
        return "break"

    # ---------------------------- annotation summary ----------------------------
    @staticmethod
    def _skeleton_length_px(binary):
        """Approximate 8-connected skeleton length without a skan dependency."""
        skel = morphology.skeletonize(np.asarray(binary) > 0)
        if not np.any(skel):
            return 0.0
        h, w = skel.shape
        length = 0.0
        # Count each undirected edge once.  Diagonal shortcuts across an already
        # connected orthogonal corner are ignored to avoid corner over-counting.
        for r, c in np.argwhere(skel):
            if c + 1 < w and skel[r, c + 1]:
                length += 1.0
            if r + 1 < h and skel[r + 1, c]:
                length += 1.0
            if r + 1 < h and c + 1 < w and skel[r + 1, c + 1]:
                if not (skel[r + 1, c] or skel[r, c + 1]):
                    length += 2.0 ** 0.5
            if r + 1 < h and c - 1 >= 0 and skel[r + 1, c - 1]:
                if not (skel[r + 1, c] or skel[r, c - 1]):
                    length += 2.0 ** 0.5
        return float(length)

    def _format_length(self, pixels):
        pixels = float(pixels)
        if self.unit == "px":
            return f"{pixels:.1f} px"
        return f"{pixels * self.resolution:.3f} {self.unit}"

    def _format_area(self, pixels):
        pixels = float(pixels)
        if self.unit == "px":
            return f"{pixels:.0f} px²"
        return f"{pixels * self.resolution * self.resolution:.3f} {self.unit}²"

    def _recompute_annotation_summary(self):
        leaf_mask = self._label_layer(1)
        primary_mask = self._label_layer(2)
        lateral_values = [v for v in self._existing_layer_values() if int(v) >= 3]
        leaf_groups = max(0, cv2.connectedComponents(leaf_mask.astype(np.uint8), connectivity=8)[0] - 1)
        leaf_n = estimate_leaf_count_from_mask(leaf_mask, self.image)
        cache = {
            "leaf_count": int(leaf_n),
            "leaf_groups": int(leaf_groups),
            "leaf_area": int(leaf_mask.sum()),
            "primary_length": self._skeleton_length_px(primary_mask),
            "lateral_values": lateral_values,
            "lateral_lengths": {},
            "label_pixels": {},
        }
        for value in [1, 2] + lateral_values:
            cache["label_pixels"][value] = int(self._label_layer(value).sum())
        for value in lateral_values:
            cache["lateral_lengths"][value] = self._skeleton_length_px(self._label_layer(value))
        cache["lateral_total"] = float(sum(cache["lateral_lengths"].values()))
        self._summary_cache = cache

    def _update_annotation_summary(self, recompute=False):
        if recompute or self._summary_cache is None:
            self._recompute_annotation_summary()
        c = self._summary_cache
        if self.language == "ja":
            group_text = (f"、連結領域 {c['leaf_groups']} 個"
                          if c['leaf_count'] != c['leaf_groups'] else "")
            self.summary_leaf_var.set(
                f"葉: 推定 {c['leaf_count']} 枚{group_text}、面積 {self._format_area(c['leaf_area'])}"
            )
            self.summary_primary_var.set(
                f"主根: 長さ {self._format_length(c['primary_length'])}"
            )
            self.summary_lateral_var.set(
                f"側根: {len(c['lateral_values'])} 本、合計 {self._format_length(c['lateral_total'])}"
            )
        else:
            group_text = f", {c['leaf_groups']} connected group(s)" if c['leaf_count'] != c['leaf_groups'] else ""
            self.summary_leaf_var.set(
                f"Leaf: {c['leaf_count']} estimated leaf(s){group_text}, area {self._format_area(c['leaf_area'])}"
            )
            self.summary_primary_var.set(
                f"Primary root: length {self._format_length(c['primary_length'])}"
            )
            self.summary_lateral_var.set(
                f"Lateral roots: {len(c['lateral_values'])}, total {self._format_length(c['lateral_total'])}"
            )
        value = int(self.class_var.get())
        if self.language == "ja":
            if value == 0:
                detail = "背景 / 消しゴム"
            elif value == 1:
                detail = f"葉ラベル、面積 {self._format_area(c['label_pixels'].get(1, 0))}"
            elif value == 2:
                detail = f"主根、長さ {self._format_length(c['primary_length'])}"
            else:
                detail = f"側根 #{value - 2}、値 {value}、長さ {self._format_length(c['lateral_lengths'].get(value, 0.0))}"
            self.summary_active_var.set(f"選択中: {detail}")
        else:
            if value == 0:
                detail = "Background / Eraser"
            elif value == 1:
                detail = f"Leaf label, area {self._format_area(c['label_pixels'].get(1, 0))}"
            elif value == 2:
                detail = f"Primary root, length {self._format_length(c['primary_length'])}"
            else:
                detail = (f"Lateral #{value - 2}, value {value}, length "
                          f"{self._format_length(c['lateral_lengths'].get(value, 0.0))}")
            self.summary_active_var.set(f"Active: {detail}")

    def _project_has_pending_edits(self):
        """Return True only when mask pixels differ from the last project apply."""
        if self.on_apply is None:
            return False
        baseline = getattr(self, "_project_mask_baseline", None)
        baseline_layers = getattr(self, "_project_layers_baseline", None)
        if baseline is None or baseline.shape != self.mask.shape:
            return True
        if not np.array_equal(self.mask, baseline):
            return True
        return not self._layers_equal(self.label_layers, baseline_layers)

    def _update_apply_button_state(self):
        """Enable Apply only while there are unapplied label-pixel changes."""
        pending = self._project_has_pending_edits()
        button = getattr(self, "apply_button", None)
        if button is not None:
            button.configure(state=("normal" if pending else "disabled"))
        # In integrated mode, closing should prompt only for changes that have
        # not yet been applied to the project.
        if self.on_apply is not None:
            self._dirty = pending
        return pending

    def _mark_summary_changed(self):
        self._summary_cache = None
        self._update_annotation_summary(recompute=True)
        self._refresh_root_brushes()
        self._update_apply_button_state()

    # ---------------------------- display / metrics ----------------------------
    def _compose(self, output_size=None):
        """Compose the editor view from independent overlap-aware layers."""
        h, w = self.mask.shape
        if output_size is None:
            rw, rh = w, h
        else:
            rw, rh = map(int, output_size)
            rw, rh = max(1, rw), max(1, rh)

        if self.show_image_var.get():
            key = (id(self.image), rw, rh)
            if self._base_display_cache_key != key or self._base_display_cache is None:
                if (rw, rh) == (w, h):
                    base = self.image.copy()
                else:
                    interp = cv2.INTER_AREA if (rw < w or rh < h) else cv2.INTER_LINEAR
                    base = cv2.resize(self.image, (rw, rh), interpolation=interp)
                self._base_display_cache_key = key
                self._base_display_cache = base
            out = self._base_display_cache.copy()
        else:
            out = np.zeros((rh, rw, 3), dtype=np.uint8)

        active_only = self.label_view_var.get() == 'active'
        active_value = int(getattr(self, 'view_label_value', self.class_var.get()))
        visible_values = self._existing_layer_values()

        # Resize label layers independently.  This is important: resizing only a
        # single semantic projection would lose overlap membership at junctions.
        display_layers = {v: self._resized_layer(v, (rw, rh)) for v in visible_values}

        if self.show_labels_var.get():
            if active_only:
                if active_value > 0:
                    selected = display_layers.get(active_value)
                    if selected is not None and np.any(selected):
                        out[selected] = label_color(active_value)
            else:
                # Average colors at shared pixels instead of letting one instance
                # visually punch a hole through another root at the junction.
                accum = np.zeros((rh, rw, 3), dtype=np.float32)
                count = np.zeros((rh, rw), dtype=np.float32)
                for value in visible_values:
                    layer = display_layers[value]
                    if not np.any(layer):
                        continue
                    accum[layer] += label_color(value).astype(np.float32)
                    count[layer] += 1.0
                fg = count > 0
                if np.any(fg):
                    out[fg] = np.clip(accum[fg] / count[fg, None], 0, 255).astype(np.uint8)

        if self.show_skeleton_var.get():
            if active_only:
                root = display_layers.get(active_value, np.zeros((rh, rw), dtype=bool)) if active_value >= 2 else np.zeros((rh, rw), dtype=bool)
            else:
                root = np.zeros((rh, rw), dtype=bool)
                for value, layer in display_layers.items():
                    if int(value) >= 2:
                        root |= layer
            if np.any(root):
                skel = morphology.skeletonize(root)
                out[skel] = np.array([255, 255, 255], dtype=np.uint8)

        if self.show_diff_var.get() and self.reference_mask is not None:
            if (rw, rh) == (w, h):
                ref_mask = self.reference_mask
            else:
                ref_mask = cv2.resize(self.reference_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)

            if active_only:
                cur = display_layers.get(active_value, np.zeros((rh, rw), dtype=bool))
                ref = ref_mask == active_value
            else:
                cur = np.zeros((rh, rw), dtype=bool)
                for layer in display_layers.values():
                    cur |= layer
                ref = ref_mask > 0

            tp = cur & ref
            fp = cur & ~ref
            fn = ~cur & ref
            out[tp] = np.array([0, 210, 0], dtype=np.uint8)
            out[fp] = np.array([255, 40, 40], dtype=np.uint8)
            out[fn] = np.array([40, 100, 255], dtype=np.uint8)
        return out

    def _update_metrics(self):
        if self.reference_mask is None:
            self.metrics_var.set("")
            return
        if self.label_view_var.get() == "active":
            value = int(getattr(self, "view_label_value", self.class_var.get()))
            cur = self._label_layer(value)
            ref = self.reference_mask == value
        else:
            cur = self._foreground_union_layer()
            ref = self.reference_mask > 0
        tp = int(np.logical_and(cur, ref).sum())
        fp = int(np.logical_and(cur, ~ref).sum())
        fn = int(np.logical_and(~cur, ref).sum())
        tn = int(np.logical_and(~cur, ~ref).sum())
        dice_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn
        precision_den = tp + fp
        recall_den = tp + fn
        dice = (2 * tp / dice_den) if dice_den else 1.0
        iou = (tp / iou_den) if iou_den else 1.0
        precision = (tp / precision_den) if precision_den else 1.0
        recall = (tp / recall_den) if recall_den else 1.0
        self.metrics_var.set(
            f"TP {tp}  FP {fp}  FN {fn}  TN {tn}   Dice {dice:.4f}  IoU {iou:.4f}  P {precision:.4f}  R {recall:.4f}"
        )

    def _request_redraw(self, immediate=False):
        """Coalesce expensive image redraws while preserving every mask sample."""
        if immediate:
            if self._redraw_pending:
                # A scheduled callback may still fire, but _redraw is idempotent.
                self._redraw_pending = False
            self._redraw()
            return
        if not self._redraw_pending:
            self._redraw_pending = True
            # ~40 FPS is visually smooth while leaving the Tk event loop enough
            # time to collect fast mouse motion events.
            self.window.after(25, self._redraw)

    def _redraw(self):
        self._redraw_pending = False
        h, w = self.mask.shape
        rw = max(1, int(round(w * self.scale)))
        rh = max(1, int(round(h * self.scale)))
        comp = self._compose((rw, rh))
        self._photo = ImageTk.PhotoImage(Image.fromarray(comp))
        self.canvas.delete("image")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="image")
        self.canvas.tag_lower("image")
        self.canvas.configure(scrollregion=(0, 0, rw, rh))
        self._draw_colored_cursor()
        self._update_metrics()
        value = int(self.class_var.get())
        tool_name = {"brush":"ブラシ", "line":"線", "fill":"塗りつぶし", "picker":"スポイト"}.get(self.tool_var.get(), self.tool_var.get())
        self.status_var.set(self._loc(
            f"label {value} ({label_name(value)}) | {self.tool_var.get()} | brush {int(self.brush_var.get())} px | zoom {self.scale * 100:.0f}% | {w}×{h}",
            f"ラベル {value}（{self._tr(label_name(value))}） | {tool_name} | ブラシ {int(self.brush_var.get())} px | ズーム {self.scale * 100:.0f}% | {w}×{h}"
        ))

    def _fit_to_window(self, after_idle=False):
        if after_idle:
            self.window.after(100, self.fit_to_window)
        else:
            self.fit_to_window()

    def fit_to_window(self):
        self.window.update_idletasks()
        cw = max(100, self.canvas.winfo_width() - 10)
        ch = max(100, self.canvas.winfo_height() - 10)
        h, w = self.mask.shape
        self.scale = max(0.05, min(cw / w, ch / h, 4.0))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._request_redraw()

    def actual_size(self):
        self.scale = 1.0
        self._request_redraw()

    def clear_mask(self):
        if not self._existing_layer_values():
            return
        if messagebox.askyesno(self._tr("Clear mask"), self._loc("Clear all annotations?", "すべてのアノテーションを消去しますか？"), parent=self.window):
            self.label_layers = {}
            self._sync_projection_from_layers()
            self._commit_history("Clear mask")
            self._request_redraw()

    # ---------------------------- files ----------------------------
    @staticmethod
    def _read_image_unicode(path, flags=cv2.IMREAD_UNCHANGED):
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)

    @staticmethod
    def _write_png_unicode(path, arr):
        ext = os.path.splitext(path)[1].lower() or ".png"
        ok, buf = cv2.imencode(ext, arr)
        if not ok:
            raise IOError(f"Could not encode {path}")
        buf.tofile(path)

    def open_image(self):
        path = filedialog.askopenfilename(
            parent=self.window,
            filetypes=[(self._tr("Images"), "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), (self._tr("All files"), "*.*")],
        )
        if not path:
            return
        arr = self._read_image_unicode(path, cv2.IMREAD_COLOR)
        if arr is None:
            messagebox.showerror(self._tr("Open image"), self._loc("Could not open image.", "画像を開けませんでした。"), parent=self.window)
            return
        self.image = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.label_layers = {}
        self._sync_projection_from_layers()
        self._project_mask_baseline = self.mask.copy()
        self._project_layers_baseline = self._copy_layers(self.label_layers)
        self.reference_mask = None
        self.reference_path = None
        self.reference_is_ground_truth = False
        self._update_validation_button_state()
        self.history_path = os.path.splitext(path)[0] + ".rootedit_history.npz"
        self.persist_history_var.set(os.path.isfile(self.history_path))
        if self.persist_history_var.get() and self._load_history(self.history_path):
            pass
        else:
            self._reset_history("Open image")
        if self.on_apply is None:
            self._dirty = False
        self._update_apply_button_state()
        self.fit_to_window()

    def open_mask(self):
        path = filedialog.askopenfilename(
            parent=self.window,
            filetypes=[(self._tr("Mask"), "*.png *.tif *.tiff *.bmp"), (self._tr("All files"), "*.*")],
        )
        if not path:
            return
        arr = self._read_image_unicode(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            return
        self.mask = self._normalize_mask(arr, self.image.shape[:2])
        self.label_layers = self._layers_from_projection(self.mask)
        overlap_loaded = self._load_overlap_sidecar(path)
        if not overlap_loaded:
            self._sync_projection_from_layers()
        self._reset_history("Open mask")
        self._dirty = True
        suffix = self._loc(" + overlap data", " + 重複データ") if overlap_loaded else ""
        self.status_var.set(self._loc(f"Opened mask: {path}", f"マスクを開きました: {path}") + suffix)
        self._request_redraw()

    def _update_validation_button_state(self):
        button = getattr(self, "save_validation_button", None)
        if button is None:
            return
        enabled = (
            self.on_save_validation is not None
            and self.reference_is_ground_truth
            and self.reference_mask is not None
        )
        button.configure(state=("normal" if enabled else "disabled"))

    def save_validation_results(self):
        """Export current prediction-vs-GT metrics through the integrated callback."""
        if self.on_save_validation is None:
            messagebox.showinfo(
                self._tr("Validation export"),
                self._loc("Validation export is available only in the integrated project editor.", "検証結果の出力は統合プロジェクトエディターでのみ使用できます。"),
                parent=self.window,
            )
            return
        if not self.reference_is_ground_truth or self.reference_mask is None:
            messagebox.showinfo(
                self._tr("Validation export"),
                self._loc("Load a ground-truth mask for the current ROI first.", "先に現在の ROI の正解（GT）マスクを読み込んでください。"),
                parent=self.window,
            )
            return
        if self.reference_mask.shape != self.mask.shape:
            messagebox.showerror(
                self._tr("Validation export"),
                self._loc("The GT mask size does not exactly match the current ROI.", "GT マスクのサイズが現在の ROI と一致しません。"),
                parent=self.window,
            )
            return
        try:
            result = self.on_save_validation(
                self.mask.copy(),
                self.reference_mask.copy(),
                self.reference_path,
            )
        except Exception as exc:
            messagebox.showerror(
                self._tr("Validation export"),
                self._loc(f"Could not save validation results:\n{exc}", f"検証結果を保存できませんでした:\n{exc}"),
                parent=self.window,
            )
            return

        if isinstance(result, (list, tuple)):
            paths = [str(p) for p in result if p]
            message = self._loc("Saved validation results:\n", "検証結果を保存しました:\n") + "\n".join(paths)
        elif result:
            message = self._loc(f"Saved validation results:\n{result}", f"検証結果を保存しました:\n{result}")
        else:
            message = self._loc("Validation results saved.", "検証結果を保存しました。")
        self.status_var.set(self._loc("Validation results saved", "検証結果を保存しました"))
        messagebox.showinfo(self._tr("Validation export"), message, parent=self.window)

    def load_reference(self):
        """Load an exact-shape ground-truth semantic mask for this ROI."""
        path = filedialog.askopenfilename(
            parent=self.window,
            title=self._tr("Load GT mask for current ROI"),
            filetypes=[
                (self._tr("Ground-truth mask"), "*.png *.tif *.tiff *.bmp"),
                (self._tr("All files"), "*.*"),
            ],
        )
        if not path:
            return

        arr = self._read_image_unicode(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            messagebox.showerror(
                self._tr("Load GT mask"), self._loc("Could not open the selected GT mask.", "選択した GT マスクを開けませんでした。"), parent=self.window
            )
            return
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if tuple(arr.shape[:2]) != tuple(self.image.shape[:2]):
            messagebox.showerror(
                self._tr("Load GT mask"),
                self._loc(
                    f"GT mask size must exactly match the current ROI.\n\nCurrent ROI: {self.image.shape[1]} x {self.image.shape[0]}\nGT mask: {arr.shape[1]} x {arr.shape[0]}\n\nNo resizing or alignment is applied.",
                    f"GT マスクのサイズは現在の ROI と完全に一致する必要があります。\n\n現在の ROI: {self.image.shape[1]} x {self.image.shape[0]}\nGT マスク: {arr.shape[1]} x {arr.shape[0]}\n\nリサイズや位置合わせは行いません。"
                ),
                parent=self.window,
            )
            return

        self.reference_mask = np.asarray(arr, dtype=np.uint8).copy()
        self.reference_path = path
        self.reference_is_ground_truth = True
        self.show_diff_var.set(True)
        self._update_validation_button_state()
        self._request_redraw(immediate=True)
        self.status_var.set(self._loc(f"GT mask loaded: {os.path.basename(path)}", f"GT マスクを読み込みました: {os.path.basename(path)}"))

    def save_mask(self, path=None):
        if path is None:
            path = filedialog.asksaveasfilename(
                parent=self.window,
                defaultextension=".png",
                filetypes=[(self._tr("PNG semantic mask"), "*.png")],
            )
        if not path:
            return False
        try:
            # PNG remains a conventional semantic projection for interoperability.
            # The sidecar stores the exact multi-label membership at junctions.
            self._sync_projection_from_layers()
            self._write_png_unicode(path, self.mask.astype(np.uint8))
            sidecar = self._save_overlap_sidecar(path)
            if self.persist_history_var.get():
                if not self.history_path:
                    self.history_path = os.path.splitext(path)[0] + ".rootedit_history.npz"
                self._save_history_silent()
            if self.on_apply is None:
                self._dirty = False
            else:
                self._update_apply_button_state()
            self.status_var.set(self._loc(
                f"Saved full-width mask (not skeletonized): {path}; overlap: {sidecar}",
                f"全幅マスクを保存しました（スケルトン化なし）: {path}; 重複: {sidecar}"
            ))
            return True
        except Exception as exc:
            messagebox.showerror(self._tr("Save mask"), str(exc), parent=self.window)
            return False

    def save_color_labels(self):
        path = filedialog.asksaveasfilename(
            parent=self.window,
            defaultextension=".png",
            filetypes=[(self._tr("PNG color labels"), "*.png")],
        )
        if not path:
            return
        color = np.zeros((*self.mask.shape, 3), dtype=np.uint8)
        accum = np.zeros((*self.mask.shape, 3), dtype=np.float32)
        count = np.zeros(self.mask.shape, dtype=np.float32)
        for value in self._existing_layer_values():
            layer = self._label_layer(value)
            accum[layer] += label_color(value).astype(np.float32)
            count[layer] += 1.0
        fg = count > 0
        if np.any(fg):
            color[fg] = np.clip(accum[fg] / count[fg, None], 0, 255).astype(np.uint8)
        try:
            self._write_png_unicode(path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            self.status_var.set(self._loc(f"Saved color labels: {path}", f"カラーラベルを保存しました: {path}"))
        except Exception as exc:
            messagebox.showerror(self._tr("Save labels"), str(exc), parent=self.window)

    def apply_to_project(self):
        if self.on_apply is None:
            return
        try:
            if not self._project_has_pending_edits():
                self._update_apply_button_state()
                return
            self._sync_projection_from_layers()
            overlap_payload = {
                int(v): self._label_layer(v).astype(np.uint8).copy()
                for v in self._existing_layer_values()
            }

            # Backward-compatible callback protocol.  New integrations may accept
            # ``overlap_masks=...`` and preserve shared primary/lateral pixels all
            # the way into RootProp masks.  Older one-argument callbacks continue
            # to receive the semantic projection exactly as before.
            callback = self.on_apply
            used_overlap = False
            try:
                sig = inspect.signature(callback)
            except (TypeError, ValueError):
                sig = None

            if sig is None:
                callback(self.mask.copy())
            else:
                params = list(sig.parameters.values())
                has_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
                if 'overlap_masks' in sig.parameters or has_kw:
                    callback(self.mask.copy(), overlap_masks=overlap_payload)
                    used_overlap = True
                else:
                    positional = [p for p in params if p.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )]
                    if has_varargs or len(positional) >= 2:
                        callback(self.mask.copy(), overlap_payload)
                        used_overlap = True
                    else:
                        callback(self.mask.copy())

            if self.persist_history_var.get():
                self._save_history_silent()
            self._project_mask_baseline = self.mask.copy()
            self._project_layers_baseline = self._copy_layers(self.label_layers)
            self._dirty = False
            self._update_apply_button_state()
            note = self._loc(
                'Applied to project with overlap masks' if used_overlap else 'Applied projection to project (overlap callback not available)',
                '重複マスク付きでプロジェクトへ適用しました' if used_overlap else '投影マスクをプロジェクトへ適用しました（重複対応コールバックなし）'
            )
            self.status_var.set(note)
        except Exception as exc:
            messagebox.showerror(self._tr("Apply"), str(exc), parent=self.window)

    def _on_close(self):
        if self.persist_history_var.get():
            self._save_history_silent()
        if self._dirty:
            if self.on_apply is not None:
                answer = messagebox.askyesnocancel(
                    self._tr("Unsaved edits"),
                    self._loc("Apply the current edits to the project before closing?", "閉じる前に現在の編集内容をプロジェクトへ適用しますか？"),
                    parent=self.window,
                )
                if answer is None:
                    return
                if answer:
                    self.apply_to_project()
            else:
                answer = messagebox.askyesnocancel(
                    self._tr("Unsaved edits"),
                    self._loc("Save the semantic mask before closing?", "閉じる前にセマンティックマスクを保存しますか？"),
                    parent=self.window,
                )
                if answer is None:
                    return
                if answer and not self.save_mask():
                    return
        self.window.destroy()


def launch_standalone(master=None, language="en"):
    """Open the editor independently. Can be called from the main GUI menu."""
    title = "RAPID - 根アノテーションエディター" if language == "ja" else "RAPID - Root annotation editor"
    if master is None:
        editor = SemanticMaskEditor(None, allow_open_image=True, language=language, title=title)
        editor.open_image()
        editor.window.mainloop()
        return editor
    return SemanticMaskEditor(master, allow_open_image=True, language=language, title=title)


if __name__ == "__main__":
    launch_standalone()
