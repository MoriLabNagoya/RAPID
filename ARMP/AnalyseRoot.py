# -*- coding: utf-8 -*-
import time
"""
Created on Tue Apr 26 15:46:22 2022

@author: chwang
"""
import cv2
from skimage import morphology
from skimage.segmentation import watershed

from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import itertools
from skimage.filters import threshold_multiotsu
#import glob 
#import os
#import sys
#from scipy import misc
#from scipy.sparse.dok import dok_matrix
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra
#import skimage.filters.thresholding
import math
import heapq
IMAGE_SIZE = 256
ROOTTYPES =['None','Leaf','Primary root','Lateral root','External root']
from scipy.ndimage import convolve

def find_endpoints(skeleton):
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    endpoints = np.argwhere(neighbors == 11)  # 自身+1邻居
    return endpoints
def dehaze(img):
    arr = np.asarray(img, dtype=np.float32)
    low_in = float(np.min(arr))
    high_in = float(np.max(arr))
    if high_in <= low_in:
        return np.zeros_like(arr, dtype=np.uint8)
    new_img = 255.0 * ((arr - low_in) / (high_in - low_in))
    return np.clip(new_img, 0, 255).astype(np.uint8)
class RootProp:
    def __init__(self,leng,distan,path,t,pt,image,lr,root_class=None):
        self.length =leng
        self.distance =distan
        self.path =path
        self.type = t
        # ``type`` historically doubled as the lateral-instance id (3, 4, 5, ...).
        # Keep that field for backward compatibility, but store the biological
        # class separately so an external/disconnected root can be represented
        # without breaking old projects.
        if root_class is None:
            self.root_class = 3 if int(t) >= 3 else int(t)
        else:
            self.root_class = int(root_class)
        
        self.pt = pt
        self.lr = lr
        if len(image.shape)==2:
            self.mask = image.copy()
        else:
            self.mask = np.zeros_like(image[:,:,0])
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f
class AnalyseRoot:
    def __init__(self,im,method,bsave,Show,Unet):
        #print("Init")
        self._image = im
        self._method = method
        self._bSave = bsave
        self._bShowImg = Show
        self.Unet= Unet
        self.mRootProp =list()
    

    def _primary_path_from_crown(self, skeleton0, leaf_mask=None, root_mask=None):
        """Select a primary-root centreline using crown evidence first.

        A component is allowed to compete for the primary root only when it is
        physically close to the lower leaf/crown region.  This prevents a long
        Petri-dish rim or neighbouring object, separated from the shoot, from
        winning merely because it has a long skeleton.  After the component is
        selected, the distal path is chosen by geodesic length together with a
        mild preference for downward progression.
        """
        sk = (np.asarray(skeleton0) > 0).astype(np.uint8)
        h, w = sk.shape
        pts_all = np.argwhere(sk > 0)
        if pts_all.size == 0:
            self._primary_component_mask = np.zeros_like(sk, dtype=np.uint8)
            return [], None, None

        # Estimate a typical root radius from the binary root mask.  This is
        # used only to scale proximity tolerances; fixed 5-pixel thresholds are
        # too sensitive to plant/image size.
        root_radius = 2.0
        if root_mask is not None and np.any(root_mask):
            rb = (np.asarray(root_mask) > 0).astype(np.uint8)
            dt_root = cv2.distanceTransform(rb, cv2.DIST_L2, 5)
            vals = dt_root[sk > 0]
            vals = vals[vals > 0]
            if vals.size:
                root_radius = float(np.clip(np.median(vals), 1.0, 12.0))
        self._root_radius_est = root_radius

        crown_target = (0, int(round((w - 1) / 2.0)))
        leaf_center = (0.0, float((w - 1) / 2.0))
        leaf_radius = 0.0
        leaf_dist = None
        lower_leaf = None
        if leaf_mask is not None and np.any(leaf_mask):
            lm = (np.asarray(leaf_mask) > 0).astype(np.uint8)
            lp = np.argwhere(lm > 0)

            # Use the centre of the complete leaf mass as an independent cue.
            # A real primary root normally starts close to this centre, whereas a
            # Petri-dish rim / neighbouring object can be very long but starts far
            # away horizontally.  The 90th-percentile radius describes the normal
            # extent of the leaf mass, so a root immediately below a large leaf is
            # not penalised simply because it is far from the centroid in pixels.
            leaf_center = (float(np.median(lp[:, 0])), float(np.median(lp[:, 1])))
            leaf_r = np.sqrt((lp[:, 0] - leaf_center[0]) ** 2 +
                             (lp[:, 1] - leaf_center[1]) ** 2)
            if leaf_r.size:
                leaf_radius = float(np.percentile(leaf_r, 90))

            y_cut = np.percentile(lp[:, 0], 72)
            lower_leaf = lp[lp[:, 0] >= y_cut]
            if len(lower_leaf) == 0:
                lower_leaf = lp
            crown_target = (int(round(np.median(lower_leaf[:, 0]))),
                            int(round(np.median(lower_leaf[:, 1]))))
            # Distance to the leaf boundary/region for every image pixel.
            leaf_dist = cv2.distanceTransform((lm == 0).astype(np.uint8),
                                              cv2.DIST_L2, 5)
        self._leaf_distance_map = leaf_dist
        self._primary_leaf_center = leaf_center
        self._primary_leaf_radius = float(leaf_radius)

        ncc, labels = cv2.connectedComponents(sk, connectivity=8)
        components = []
        max_crown_gap = max(
            12.0,
            min(60.0, 0.030 * float(h)),
            6.0 * root_radius,
        )

        for cid in range(1, ncc):
            cp = np.argwhere(labels == cid)
            if len(cp) < 2:
                continue
            y_span = float(cp[:, 0].max() - cp[:, 0].min())
            if leaf_dist is not None:
                ld = leaf_dist[cp[:, 0], cp[:, 1]]
                j = int(np.argmin(ld))
                min_leaf = float(ld[j])
                near_pt = (int(cp[j, 0]), int(cp[j, 1]))
            else:
                d2 = ((cp[:, 0] - crown_target[0]) ** 2 +
                      (cp[:, 1] - crown_target[1]) ** 2)
                j = int(np.argmin(d2))
                min_leaf = float(np.sqrt(d2[j]))
                near_pt = (int(cp[j, 0]), int(cp[j, 1]))

            crown_dx = abs(float(near_pt[1] - crown_target[1]))
            start_center_dist = math.hypot(float(near_pt[0]) - leaf_center[0],
                                           float(near_pt[1]) - leaf_center[1])
            start_center_excess = max(0.0, start_center_dist - float(leaf_radius))

            # Keep a diagnostic score, but DO NOT allow a very long component to
            # compensate arbitrarily for starting far from the leaf centre.  The
            # actual component choice below is lexicographic: leaf/crown proximity
            # first, root extent second.
            score = (1.2 * y_span + 0.05 * math.sqrt(float(len(cp)))
                     - 5.0 * min_leaf - 2.5 * start_center_excess
                     - 0.25 * crown_dx)
            components.append((score, cid, min_leaf, near_pt, y_span, len(cp),
                               start_center_dist, start_center_excess, crown_dx))

        if not components:
            self._primary_component_mask = np.zeros_like(sk, dtype=np.uint8)
            return [], None, crown_target

        eligible = [c for c in components if c[2] <= max_crown_gap]
        if eligible:
            # The candidate start distance to the leaf centre has priority over
            # total length.  First create a narrow band around the best start, then
            # let vertical/geodesic extent choose within that biologically plausible
            # band.  This prevents a remote dish rim from winning merely by length.
            min_start_excess = min(c[7] for c in eligible)
            start_band = max(10.0, 3.0 * root_radius, 0.045 * float(w))
            near_start = [c for c in eligible
                          if c[7] <= min_start_excess + start_band]
            chosen_comp = max(near_start, key=lambda c: (c[4], c[5], -c[2], c[0]))
            selection_mode = 'leaf-center-first'
        else:
            # If leaf occlusion creates a larger crown gap, still keep the leaf
            # centre as the first cue.  Length is only a tie-breaker among roots
            # whose candidate starts are comparably close to the leaf centre.
            min_start_excess = min(c[7] for c in components)
            start_band = max(12.0, 3.5 * root_radius, 0.055 * float(w))
            near_start = [c for c in components
                          if c[7] <= min_start_excess + start_band]
            min_gap = min(c[2] for c in near_start)
            relaxed_gap = max(8.0, 2.5 * root_radius, 0.012 * float(h))
            near_components = [c for c in near_start if c[2] <= min_gap + relaxed_gap]
            chosen_comp = max(near_components, key=lambda c: (c[4], c[5], -c[7], c[0]))
            selection_mode = 'leaf-center-relaxed'
            if getattr(self, 'primary_debug', False):
                print('[PrimaryPath] strict crown gap unavailable; using leaf-center-first relaxed selection',
                      flush=True)

        _, comp_id, comp_gap, near_pt, _, _, comp_center_dist, comp_center_excess, comp_crown_dx = chosen_comp
        comp = (labels == comp_id).astype(np.uint8)
        self._primary_component_mask = comp.copy()
        comp_pts = np.argwhere(comp > 0)

        # Pick a crown anchor within the selected component.  Among points close
        # to the leaf, favour the horizontal vicinity of the lower-leaf centre.
        if leaf_dist is not None:
            ld = leaf_dist[comp_pts[:, 0], comp_pts[:, 1]]
            best_ld = float(ld.min())
            pool = comp_pts[ld <= best_ld + max(2.0, 1.5 * root_radius)]
            # Among equally leaf-adjacent pixels, choose the one closest to the
            # complete leaf centre rather than simply the lower-leaf x coordinate.
            metric = np.sqrt((pool[:, 0] - leaf_center[0]) ** 2 +
                             (pool[:, 1] - leaf_center[1]) ** 2)
            p = pool[int(np.argmin(metric))]
            anchor = (int(p[0]), int(p[1]))
        else:
            d2 = ((comp_pts[:, 0] - crown_target[0]) ** 2 +
                  (comp_pts[:, 1] - crown_target[1]) ** 2)
            p = comp_pts[int(np.argmin(d2))]
            anchor = (int(p[0]), int(p[1]))

        neigh = cv2.filter2D(comp, cv2.CV_16S, np.ones((3, 3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT) - comp.astype(np.int16)
        ep_pts = np.argwhere((comp > 0) & (neigh == 1))
        if len(ep_pts) == 0:
            ep_pts = comp_pts
        below = ep_pts[ep_pts[:, 0] >= anchor[0] - int(round(0.03 * h))]
        if len(below) > 0:
            ep_pts = below

        nbrs = [(-1,-1,math.sqrt(2.0)), (-1,0,1.0), (-1,1,math.sqrt(2.0)),
                (0,-1,1.0),                         (0,1,1.0),
                (1,-1,math.sqrt(2.0)),  (1,0,1.0),  (1,1,math.sqrt(2.0))]
        dist = {anchor: 0.0}
        parent = {anchor: None}
        pq = [(0.0, anchor)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u):
                continue
            uy, ux = u
            for dy, dx, cost in nbrs:
                v = (uy + dy, ux + dx)
                vy, vx = v
                if vy < 0 or vy >= h or vx < 0 or vx >= w or comp[vy, vx] == 0:
                    continue
                nd = d + cost
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))

        def reconstruct(end):
            path = []
            cur = end
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            path.reverse()
            return path

        # Candidate ranking is intentionally LENGTH-FIRST.
        #
        # The previous score strongly rewarded vertical_eff.  In practice this can
        # make a shorter, almost-vertical lateral/ROI-edge path outrank a longer
        # primary root.  Here verticality is no longer a positive cue by itself.
        # Instead it is used together with border occupancy + straightness to detect
        # likely image/Petri-dish edges.  Among non-edge candidates, geodesic length
        # is the dominant cue; downward progress and horizontal displacement are only
        # small tie-breakers.
        side_margin = max(5, int(round(0.04 * w)))
        candidate_records = []
        candidate_debug = []

        for p in ep_pts:
            end = (int(p[0]), int(p[1]))
            if end == anchor or end not in dist:
                continue
            path = reconstruct(end)
            if len(path) < 2:
                continue

            arr = np.asarray(path, dtype=np.int32)
            geo = float(dist[end])
            dy = float(end[0] - anchor[0])
            dx = float(end[1] - anchor[1])
            downward = max(0.0, dy)
            horiz = abs(dx)
            euclid = math.hypot(dy, dx)
            straightness = euclid / max(geo, 1e-9)
            verticality = abs(dy) / max(euclid, 1e-9)  # 1.0 => globally vertical

            border_dist = np.minimum(arr[:, 1], (w - 1) - arr[:, 1])
            side_frac = float(np.mean(border_dist <= side_margin))

            # A dish/ROI edge usually shows up most clearly in the distal portion:
            # after leaving the crown neighbourhood, a long, straight path runs
            # almost vertically along x≈0 or x≈w.  Requiring all three properties
            # avoids rejecting a genuine vertical primary root near the image centre.
            distal_start = max(0, int(round(0.40 * len(arr))))
            distal_border = border_dist[distal_start:]
            distal_side_frac = (float(np.mean(distal_border <= side_margin))
                                if distal_border.size else side_frac)
            end_on_side = min(end[1], (w - 1) - end[1]) <= side_margin
            edge_like = bool(
                verticality >= 0.90 and
                straightness >= 0.88 and
                (
                    distal_side_frac >= 0.60 or
                    (end_on_side and side_frac >= 0.45)
                )
            )

            # Length is deliberately dominant.  The correction terms are small:
            # they resolve near-ties rather than allowing a short vertical lateral
            # to beat a substantially longer primary root.
            score = (
                geo
                + 0.10 * downward
                - 0.08 * horiz
                - 0.20 * geo * max(0.0, side_frac - 0.20)
            )

            rec = {
                'score': score,
                'end': end,
                'geo': geo,
                'downward': downward,
                'horiz': horiz,
                'verticality': verticality,
                'straightness': straightness,
                'side_frac': side_frac,
                'distal_side_frac': distal_side_frac,
                'edge_like': edge_like,
                'path': path,
            }
            candidate_records.append(rec)

        # Stage 1: reject obvious border-edge paths if at least one biological-looking
        # alternative exists.  Never return empty solely because every candidate was
        # edge-like; the caller has additional crown/line checks and a conservative
        # fallback.
        usable = [r for r in candidate_records if not r['edge_like']]
        if not usable:
            usable = candidate_records

        # Stage 2: longest path first.  Keep candidates in a narrow length band near
        # the longest one, then use the mild score above to break close calls.  This
        # makes the selection stable against one or two skeleton pixels of noise.
        if usable:
            max_geo = max(r['geo'] for r in usable)
            length_band = max(3.0, 0.05 * max_geo)
            near_longest = [r for r in usable if r['geo'] >= max_geo - length_band]
            best_rec = max(near_longest, key=lambda r: (
                r['score'], r['geo'], r['downward'], -r['horiz']
            ))
            best_score = float(best_rec['score'])
            best_path = best_rec['path']
            best_end = best_rec['end']
        else:
            best_score = -float('inf')
            best_path = None
            best_end = None

        candidate_debug = [
            (r['score'], r['end'], r['geo'], r['verticality'], r['straightness'],
             r['side_frac'], r['distal_side_frac'], r['edge_like'])
            for r in candidate_records
        ]

        if best_path is None:
            reachable = [(d, p) for p, d in dist.items() if p != anchor]
            if not reachable:
                return [anchor], anchor, crown_target
            _, best_end = max(reachable)
            best_path = reconstruct(best_end)

        anchor_center_dist = math.hypot(float(anchor[0]) - leaf_center[0],
                                        float(anchor[1]) - leaf_center[1])
        anchor_center_excess = max(0.0, anchor_center_dist - float(leaf_radius))

        if getattr(self, 'primary_debug', False):
            print('[PrimaryPath] mode=%s crown=%s leaf_center=(%.1f,%.1f) leaf_radius=%.1f '
                  'selected_component=%d gap=%.1f start_center_dist=%.1f excess=%.1f '
                  'anchor=%s endpoints=%d chosen=%s path_points=%d root_radius=%.2f' %
                  (selection_mode, str(crown_target), leaf_center[0], leaf_center[1], leaf_radius,
                   comp_id, comp_gap, anchor_center_dist, anchor_center_excess, str(anchor),
                   len(ep_pts), str(best_end), len(best_path), root_radius), flush=True)
            for score, end, geo, vert, straight, sf, dsf, edge_like in sorted(candidate_debug, reverse=True)[:8]:
                print('  candidate=%s score=%.1f geodesic=%.1f verticality=%.2f '
                      'straightness=%.2f side=%.2f distal_side=%.2f edge_like=%s' %
                      (str(end), score, geo, vert, straight, sf, dsf, str(edge_like)),
                      flush=True)

        # Save diagnostics for the post-selection plausibility check.
        self._primary_selection_mode = selection_mode
        self._primary_comp_gap = float(comp_gap)
        self._primary_crown_target = crown_target
        self._primary_anchor = anchor
        self._primary_component_id = int(comp_id)
        self._primary_start_leaf_center_dist = float(anchor_center_dist)
        self._primary_start_leaf_center_excess = float(anchor_center_excess)

        return best_path, anchor, crown_target

    def _reject_line_like_remote_primary(self, path, image_shape, root_radius=None):
        """Reject a line-like primary only when its start is also remote from leaves.

        A genuine primary root can be straight, therefore straightness alone is
        never sufficient.  The candidate is suspicious when (1) its centreline is
        well explained by one fitted line and (2) its crown-side start lies well
        outside the normal leaf extent.  This combination targets Petri-dish rims,
        scratches and neighbouring vertical structures without deleting a straight
        root that actually originates under the shoot.
        """
        h, w = image_shape[:2]
        arr = np.asarray(path, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 20:
            return False, {'reason': 'too-short'}

        xy = np.column_stack((arr[:, 1], arr[:, 0])).astype(np.float32)
        xy0 = xy.astype(np.float64) - xy.astype(np.float64).mean(axis=0, keepdims=True)
        cov = np.cov(xy0, rowvar=False)
        try:
            eig = np.linalg.eigvalsh(cov)
            eig = np.sort(np.maximum(eig, 0.0))[::-1]
            linearity = float(eig[0] / max(eig.sum(), 1e-9))
        except Exception:
            linearity = 0.0

        # Robust geometric line fit.  fit_rmse is the perpendicular distance of
        # path points to the best line; a long dish edge usually remains small
        # even when it is locally a little wavy.
        fit_rmse = float('inf')
        fit_p90 = float('inf')
        direction = np.array([0.0, 0.0], dtype=float)
        try:
            vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(
                xy.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01
            ).ravel()]
            nrm = math.hypot(vx, vy)
            if nrm > 1e-9:
                vx, vy = vx / nrm, vy / nrm
                dx = xy[:, 0].astype(np.float64) - x0
                dy = xy[:, 1].astype(np.float64) - y0
                perp = np.abs(dx * vy - dy * vx)
                fit_rmse = float(np.sqrt(np.mean(perp * perp)))
                fit_p90 = float(np.percentile(perp, 90))
                direction = np.array([vy, vx], dtype=float)  # image dy,dx
                if direction[0] < 0:
                    direction *= -1.0
        except Exception:
            pass

        dif = np.diff(arr, axis=0)
        geo = float(np.sum(np.linalg.norm(dif, axis=1)))
        euclid = float(np.linalg.norm(arr[-1] - arr[0]))
        straightness = euclid / max(geo, 1e-9)

        rr = float(root_radius if root_radius is not None else getattr(self, '_root_radius_est', 2.0))
        comp_gap = float(getattr(self, '_primary_comp_gap', 0.0))
        crown = getattr(self, '_primary_crown_target', None)
        anchor = getattr(self, '_primary_anchor', None)
        leaf_center = getattr(self, '_primary_leaf_center', None)
        leaf_radius = float(getattr(self, '_primary_leaf_radius', 0.0))

        if crown is not None and anchor is not None:
            crown_dx = abs(float(anchor[1] - crown[1]))
        else:
            crown_dx = 0.0

        if leaf_center is not None and anchor is not None:
            start_center_dist = math.hypot(float(anchor[0]) - float(leaf_center[0]),
                                           float(anchor[1]) - float(leaf_center[1]))
        else:
            start_center_dist = float(getattr(self, '_primary_start_leaf_center_dist', 0.0))
        start_center_excess = max(0.0, start_center_dist - leaf_radius)

        # Scale-adaptive line criterion.  Both global PCA linearity and fitted-line
        # residual are used because a dish rim may be slightly wavy rather than a
        # mathematically perfect line.
        fit_limit = max(3.0, 1.8 * rr, 0.008 * max(euclid, 1.0))
        line_like = bool(
            linearity >= 0.985 and
            straightness >= 0.70 and
            fit_rmse <= fit_limit and
            fit_p90 <= 1.7 * fit_limit
        )

        # Distance from the candidate START to the whole leaf centre is the second
        # independent cue requested for remote/outlier roots.  Subtracting the leaf
        # radius makes this tolerant of large overlapping leaf masses.
        start_far_limit = max(18.0, 7.0 * rr, 0.10 * float(w))
        start_far = bool(start_center_excess > start_far_limit)
        gap_bad = comp_gap > max(18.0, 7.0 * rr, 0.035 * float(h))
        dx_bad = crown_dx > max(20.0, 8.0 * rr, 0.16 * float(w))

        # Primary rule: line-like + remote start => false-root candidate.
        # The old crown-gap/dx pair is retained only as supporting evidence.
        reject = bool(line_like and start_far)

        diag = {
            'linearity': linearity,
            'fit_rmse': fit_rmse,
            'fit_p90': fit_p90,
            'fit_limit': fit_limit,
            'straightness': straightness,
            'direction_dy': float(direction[0]),
            'direction_dx': float(direction[1]),
            'crown_gap': comp_gap,
            'crown_dx': crown_dx,
            'start_center_dist': start_center_dist,
            'start_center_excess': start_center_excess,
            'start_far_limit': start_far_limit,
            'start_far': start_far,
            'gap_bad': gap_bad,
            'dx_bad': dx_bad,
            'line_like': line_like,
        }
        return reject, diag

    def _primary_path_with_line_rejection(self, skeleton0, leaf_mask=None, root_mask=None, max_retries=3):
        """Select a primary path while never turning a root-containing ROI into leaf-only.

        A line-like candidate is rejected only when it is BOTH far from the crown
        and strongly displaced horizontally from the crown.  Rejected components
        are removed only from a temporary search skeleton.  If every alternative
        is rejected, the most crown-compatible rejected path is restored as a
        conservative fallback.  This preserves root output while still allowing
        obvious remote Petri-dish structures to be skipped when a better candidate
        exists.
        """
        work = (np.asarray(skeleton0) > 0).astype(np.uint8)
        rejected_components = []
        rejected_candidates = []

        for attempt in range(max(1, int(max_retries) + 1)):
            path, anchor, crown = self._primary_path_from_crown(work, leaf_mask, root_mask)
            if len(path) < 2:
                break

            reject, diag = self._reject_line_like_remote_primary(
                path, work.shape, getattr(self, '_root_radius_est', 2.0)
            )
            if getattr(self, 'primary_debug', False):
                print('[PrimaryLineCheck] attempt=%d reject=%s line_like=%s linearity=%.4f '
                      'fit_rmse=%.2f/%.2f straightness=%.3f dir=(%.3f,%.3f) '
                      'start_leaf_center=%.1f excess=%.1f/%.1f start_far=%s crown_gap=%.1f crown_dx=%.1f' %
                      (attempt + 1, str(reject), str(diag.get('line_like', False)),
                       diag.get('linearity', 0.0), diag.get('fit_rmse', 0.0),
                       diag.get('fit_limit', 0.0), diag.get('straightness', 0.0),
                       diag.get('direction_dy', 0.0), diag.get('direction_dx', 0.0),
                       diag.get('start_center_dist', 0.0), diag.get('start_center_excess', 0.0),
                       diag.get('start_far_limit', 0.0), str(diag.get('start_far', False)),
                       diag.get('crown_gap', 0.0), diag.get('crown_dx', 0.0)), flush=True)

            if not reject:
                return path, anchor, crown, rejected_components

            # Save the candidate before suppressing it.  If no acceptable
            # alternative exists, we restore the rejected candidate that has the
            # strongest crown evidence instead of returning an empty root result.
            fallback_score = (
                float(diag.get('start_center_excess', float('inf'))),
                float(diag.get('start_center_dist', float('inf'))),
                float(diag.get('crown_gap', float('inf'))),
                -float(len(path))
            )
            rejected_candidates.append((fallback_score, list(path), anchor, crown))

            comp = getattr(self, '_primary_component_mask', None)
            if comp is None or not np.any(comp):
                break
            rejected_components.append(comp.copy())
            work[np.asarray(comp) > 0] = 0
            if getattr(self, 'primary_debug', False):
                print('[PrimaryLineCheck] rejected remote line-like component; retry primary search',
                      flush=True)
            if not np.any(work):
                break

        if rejected_candidates:
            rejected_candidates.sort(key=lambda item: item[0])
            _score, path, anchor, crown = rejected_candidates[0]
            if getattr(self, 'primary_debug', False):
                print('[PrimaryLineCheck] no better alternative found; restore best crown-related '
                      'rejected candidate so ROI does not become leaf-only', flush=True)
            return path, anchor, crown, rejected_components

        # Last conservative attempt: use the crown selector without line rejection.
        # This is still safer than the historical global-lowest A* fallback.
        path, anchor, crown = self._primary_path_from_crown(skeleton0, leaf_mask, root_mask)
        if len(path) >= 2:
            if getattr(self, 'primary_debug', False):
                print('[PrimaryLineCheck] fallback=crown-selector-without-line-rejection', flush=True)
            return path, anchor, crown, rejected_components

        return path, anchor, crown, rejected_components

    def astar_closest(self,mask, start, end):
            def distance(pos1, pos2):
                return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

            # Initialize start and end node
            start_node = Node(None, start)
            start_node.g = start_node.h = start_node.f = 0
            end_node = Node(None, end)
            end_node.g = end_node.h = end_node.f = 0

            # Initialize both open and closed list
            open_list = []
            closed_list = []

            # Add the start node
            open_list.append(start_node)

            # Loop until you find the end or explore all nodes
            closest_node = start_node  # Keep track of the closest node to the end
            while len(open_list) > 0:
                current_node = open_list[0]
                current_index = 0
                for index, item in enumerate(open_list):
                    if item.f < current_node.f:
                        current_node = item
                        current_index = index

                # Remove current node from open list and add to closed list
                open_list.pop(current_index)
                closed_list.append(current_node)

                # If it's closer to the end than any we've seen before, update closest_node
                if distance(current_node.position, end_node.position) < distance(closest_node.position, end_node.position):
                    closest_node = current_node

                # Found the goal
                if current_node == end_node:
                    path = []
                    while current_node is not None:
                        path.append(current_node.position)
                        current_node = current_node.parent
                    return path[::-1]  # Return reversed path

                # Generate children
                children = []
                for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0),(1, 1),(1, -1),(-1, 1),(-1, -1)]:  # Adjacent squares
                    node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                    if node_position[0] > (len(mask) - 1) or node_position[0] < 0 or node_position[1] > (len(mask[0]) - 1) or node_position[1] < 0:
                        continue
                    if mask[node_position[0]][node_position[1]] != 1:
                        continue

                    new_node = Node(current_node, node_position)
                    children.append(new_node)

                for child in children:
                    if child in closed_list:
                        continue

                    child.g = current_node.g + 1
                    child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                    child.f = child.g + child.h

                    if any(open_node for open_node in open_list if child == open_node and child.g > open_node.g):
                        continue

                    open_list.append(child)

            # If we're here, it means we've explored all nodes but didn't reach the end; return the path to the closest node
            path = []
            while closest_node is not None:
                path.append(closest_node.position)
                closest_node = closest_node.parent
            return path[::-1]
    #def ProcessManual(self,mask,ind):
    def checkMore(self,mainpath,mask, start,tim):#tim is true, is insert, tim is false, is append, reverse time.
         stack = [start]

    # 只要栈不为空，就继续处理
         while stack:
        # 弹出栈顶元素作为当前点
             current = stack.pop()
            # 定义8个可能的移动方向（上，下，左，右以及对角线）
             directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]

            # 检查所有可能的邻域点
             for d in directions:
                 neighbor = (current[0] + d[0], current[1] + d[1])

                     # 确保邻域点在 mask 的范围内
                 if 0 <= neighbor[0] < len(mask) and 0 <= neighbor[1] < len(mask[0]):
                     # 检查邻域点是否有效（mask为1）且尚未被访问过
                      if mask[neighbor[0]][neighbor[1]] == 1 and neighbor not in mainpath:
                             # 将有效的邻域点推入栈中
                         stack.append(neighbor)
                         if neighbor not in mainpath:
                             if tim:
                                mainpath.insert(0,neighbor)
                             else:
                                mainpath.append(neighbor)
         return mainpath
        
    def ProcessLoad(self,mask,imgtype,mm):
       
       self.mlabel = mask.copy()
       if 1:
           mRootProp = None
           if 1:                
                mainPath= list()
                if np.sum(mask)==0:
                   mRootProp=RootProp(0, 0, mainPath, mm,(0,0),self.Pt2Mask(mainPath, mm),0)
                else:
                    curMask = mask
                    if mm ==1 and imgtype==2:# 
                        #print("leaf")
                        prop = self.GetLeafFromMask(curMask)
                        mRootProp = prop
                    else:
                        mRootProp = self.ProcessManual(mask, mm)
       self.mRootProp = mRootProp
       return mRootProp
    
    def find_lowest_point(self,binary_image):
        lowest_point = None
        for y in range(binary_image.shape[0]):
            row = binary_image[y, :]
            indices = np.where(row == 1)[0]
            if len(indices) > 0:
                x = indices[0]
                if lowest_point is None or y > lowest_point[1]:
                    lowest_point = (x, y)
        return lowest_point
    def PostProcesLable(self,img):
        kernel = np.ones((10,10), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        #angle,rect = self.GetAppromRT(img.copy())
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        if nlabels <= 1:
            return np.zeros_like(img, dtype=np.uint8)
        tmp = [stats[i, cv2.CC_STAT_AREA] for i in range(1, nlabels)]
        target = int(np.argmax(tmp)) + 1
        result = np.where(labels == target, 1, 0)
        return result.astype(np.uint8)

    def PostProcesLableLeafAware(self, img, leaf=None):
        """Root cleanup for method 5 without discarding the biological root.

        The historical PostProcesLable() keeps only the *largest* connected
        component.  That is unsafe when a Petri-dish/ruler edge is larger than
        the plant root: the true root can be deleted before primary-root
        selection even starts.

        This method keeps all components that have plausible shoot/leaf
        evidence and a few substantial fallback components.  Primary-root
        selection is then responsible for deciding which retained component is
        the actual root.  Tiny isolated speckles are still removed.
        """
        # A 10x10 rectangular closing is too aggressive for thin lateral roots:
        # it can merge nearby branches into the primary-root body and change the
        # skeleton topology before lateral-root classification.  Weak-root
        # reconstruction already repairs small gaps, so keep this cleanup local.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        work = cv2.morphologyEx((np.asarray(img) > 0).astype(np.uint8),
                                cv2.MORPH_CLOSE, kernel)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            work, connectivity=8
        )
        if nlabels <= 1:
            return np.zeros_like(work, dtype=np.uint8)

        h, w = work.shape
        image_area = float(h * w)
        min_area = max(20, int(round(0.00002 * image_area)))

        leaf_bin = None
        leaf_dist = None
        leaf_center = None
        leaf_radius = 0.0
        if leaf is not None and np.any(leaf):
            leaf_bin = (np.asarray(leaf) > 0).astype(np.uint8)
            ys, xs = np.nonzero(leaf_bin)
            if len(xs):
                leaf_center = (float(np.mean(ys)), float(np.mean(xs)))
                leaf_radius = float(np.sqrt(len(xs) / np.pi))
                # distance from every pixel to the nearest leaf pixel
                leaf_dist = cv2.distanceTransform(1 - leaf_bin, cv2.DIST_L2, 5)

        # A component close to the shoot must never be thrown away merely
        # because a remote rim/ruler component has a larger area.
        near_leaf_limit = max(
            25.0,
            1.60 * leaf_radius,
            0.055 * float(h),
            0.10 * float(w),
        )

        records = []
        for cid in range(1, nlabels):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[cid, cv2.CC_STAT_LEFT])
            y = int(stats[cid, cv2.CC_STAT_TOP])
            cw = int(stats[cid, cv2.CC_STAT_WIDTH])
            ch = int(stats[cid, cv2.CC_STAT_HEIGHT])
            comp = (labels == cid)

            if leaf_dist is not None:
                vals = leaf_dist[comp]
                min_leaf = float(vals.min()) if vals.size else float('inf')
            else:
                min_leaf = float('inf')

            cy, cx = float(centroids[cid][1]), float(centroids[cid][0])
            if leaf_center is not None:
                center_dx = abs(cx - leaf_center[1])
            else:
                center_dx = abs(cx - 0.5 * w)

            # Ranking is only used for fallback retention.  Leaf proximity is
            # intentionally much stronger than raw area.
            score = (
                -6.0 * min_leaf
                -0.35 * center_dx
                +0.35 * float(ch)
                +0.02 * float(area)
            )
            records.append({
                'cid': cid, 'area': area, 'bbox': (x, y, cw, ch),
                'min_leaf': min_leaf, 'center_dx': center_dx, 'score': score,
            })

        if not records:
            return np.zeros_like(work, dtype=np.uint8)

        keep = set()
        if leaf_dist is not None:
            for r in records:
                if r['min_leaf'] <= near_leaf_limit:
                    keep.add(r['cid'])

        # Keep a small number of substantial alternatives as a conservative
        # fallback for leaf occlusion / a short segmentation gap.  They remain
        # candidates only; they are NOT automatically called primary roots.
        by_score = sorted(records, key=lambda r: r['score'], reverse=True)
        for r in by_score[:3]:
            keep.add(r['cid'])

        # Also keep very large components so later line/leaf-center guards can
        # explicitly reject them instead of silently altering the segmentation.
        largest_area = max(r['area'] for r in records)
        for r in records:
            if r['area'] >= 0.35 * largest_area:
                keep.add(r['cid'])

        result = np.isin(labels, list(keep)).astype(np.uint8)

        if getattr(self, 'primary_debug', False):
            print('[RootCC] total=%d kept=%d min_area=%d near_leaf_limit=%.1f leaf_center=%s' %
                  (len(records), len(keep), min_area, near_leaf_limit,
                   str(leaf_center)), flush=True)
            for r in sorted(records, key=lambda rr: (rr['cid'] not in keep, rr['min_leaf'], -rr['area']))[:12]:
                print('  cc=%d keep=%s area=%d bbox=%s min_leaf=%.1f center_dx=%.1f score=%.1f' %
                      (r['cid'], str(r['cid'] in keep), r['area'], str(r['bbox']),
                       r['min_leaf'], r['center_dx'], r['score']), flush=True)

        return result

    def Dice(self, image_true, image_pred):
         # np.bool was removed in NumPy >= 1.24; use np.bool_ instead.
         image_true = np.asarray(image_true).astype(np.bool_)
         image_pred = np.asarray(image_pred).astype(np.bool_)

         # Standard Dice coefficient: 2|A∩B| / (|A| + |B|).
         intersection = np.logical_and(image_true, image_pred).sum()
         denominator = image_true.sum() + image_pred.sum()
         if denominator == 0:
             return 1.0
         return 2.0 * intersection / denominator
    def threshold_optimization(self, gray_image, binary_image):
        best_threshold = None
        best_dice_score = 0.0
        best_binary_image = None
        #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for threshold in range(256):
            _, thresholded_image = cv2.threshold(gray_image, threshold, 1, cv2.THRESH_BINARY)
            current_dice_score = self.Dice(thresholded_image, binary_image)
            #cv2.imwrite("ROIs/%d.bmp"%threshold, thresholded_image*255)
            if current_dice_score > best_dice_score:
                best_dice_score = current_dice_score
                best_threshold = threshold
                best_binary_image = thresholded_image

        return best_threshold, best_binary_image
    def Processleaf(self,leaf):
        num_labels, labels, stats, centroids  = cv2.connectedComponentsWithStats(leaf, connectivity=8)
        for label in range(1, num_labels):  
               area = stats[label, cv2.CC_STAT_AREA]  # 获取面积信
               if area < 200:
                   leaf[labels == label] = 0   
        return leaf
    def _filter_leaf_horizontal_center(self, leaf, edge_fraction=0.18):
        """Remove likely neighbouring-plant leaves using horizontal position only.

        The YOLO ROI is centred on the target plant, therefore its leaf centroid is
        expected to lie around x = w/2.  Leaf components whose centroids fall close
        to the left or right ROI boundary are treated as likely neighbouring-plant
        outliers.  No vertical-position prior is used.  If every component would be
        rejected, the original mask is returned as a conservative fallback.
        """
        leaf = (np.asarray(leaf) > 0).astype(np.uint8)
        if not np.any(leaf):
            return leaf
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(leaf, connectivity=8)
        if n <= 2:
            # A single remaining leaf component is not rejected solely by position.
            return leaf

        _h, w = leaf.shape
        margin = max(2.0, float(edge_fraction) * float(w))
        kept = []
        for i in range(1, n):
            cx = float(centroids[i][0])
            # Equivalent to requiring sufficient distance from both x=0 and x=w.
            if min(cx, float(w) - cx) >= margin:
                kept.append(i)

        if not kept:
            return leaf

        out = np.zeros_like(leaf)
        for i in kept:
            out[labels == i] = 1
        return out

    # Backward-compatible name for older calls/projects.  ICML-improve now uses
    # horizontal distance only; the previous upper-region rule is intentionally
    # no longer applied.
    def _filter_leaf_upper_center(self, leaf):
        return self._filter_leaf_horizontal_center(leaf)

    @staticmethod
    def _neighbors8(point, shape):
        r, c = point
        h, w = shape
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    yield (rr, cc), (math.sqrt(2.0) if dr and dc else 1.0)

    def _extend_from_lowest_neighborhood(self, mainpath, skeleton0, lowest_point):
        """Extend a primary path beyond its lowest point through non-zero neighbours.

        The lowest skeleton pixel is used only as the starting point for a local
        connectivity search, not as an assumed biological root tip.  Pixels already
        contained in ``mainpath`` are blocked (except the lowest point), and the
        farthest reachable skeleton pixel is appended/prepended through its shortest
        connected path.  This recovers a terminal segment that bends upward after
        reaching the lowest image position without introducing a global root filter.
        """
        if not mainpath or lowest_point is None:
            return list(mainpath)
        path = [(int(p[0]), int(p[1])) for p in mainpath]
        low = (int(lowest_point[0]), int(lowest_point[1]))
        if not (0 <= low[0] < skeleton0.shape[0] and 0 <= low[1] < skeleton0.shape[1]):
            return path
        existing = set(path)
        if low not in existing:
            return path

        dist = {low: 0.0}
        prev = {}
        pq = [(0.0, low)]
        while pq:
            d, cur = heapq.heappop(pq)
            if d != dist.get(cur):
                continue
            for nb, step in self._neighbors8(cur, skeleton0.shape):
                if not skeleton0[nb]:
                    continue
                if nb in existing and nb != low:
                    continue
                nd = d + step
                if nd < dist.get(nb, float('inf')):
                    dist[nb] = nd
                    prev[nb] = cur
                    heapq.heappush(pq, (nd, nb))

        tip = max(dist, key=dist.get)
        # A one-pixel neighbour is often only skeleton thickness/noise; require
        # a small but real continuation before modifying the primary path.
        if tip == low or dist[tip] < 2.0:
            return path
        extension = [tip]
        while extension[-1] != low:
            extension.append(prev[extension[-1]])
        extension.reverse()  # lowest -> recovered terminal tip

        if path[0] == low:
            return list(reversed(extension[1:])) + path
        if path[-1] == low:
            return path + extension[1:]
        return path

    def Pt2Mask(self,paths,target):
        mask = np.zeros_like(self.mlabel)
        for pt in paths:
            mask = cv2.circle(mask, center=(pt[1],pt[0]),radius=2,color=1,thickness=1, lineType=cv2.LINE_4,shift=0)
        return mask

    @staticmethod
    def _dijkstra_pixel_graph(graph, start):
        """Shortest paths on a small weighted skeleton-pixel graph."""
        start = (int(start[0]), int(start[1]))
        dist = {start: 0.0}
        prev = {}
        heap = [(0.0, start)]
        while heap:
            cur_d, cur = heapq.heappop(heap)
            if cur_d != dist.get(cur):
                continue
            for nb, weight in graph.get(cur, {}).items():
                nd = cur_d + float(weight)
                if nd < dist.get(nb, float('inf')):
                    dist[nb] = nd
                    prev[nb] = cur
                    heapq.heappush(heap, (nd, nb))
        return dist, prev

    def _representative_network_path(self, branch_paths, main_dist=None):
        """Return one representative path for a possibly branched root network.

        ``RootProp.length`` stores the TOTAL skeleton length of the merged network.
        ``RootProp.path`` cannot encode branching, so for direction/insertion-angle
        calculations it stores the geodesic from the primary attachment to the
        farthest descendant tip.  External roots use an approximate network
        diameter instead.
        """
        graph = {}
        all_points = set()

        def add_edge(a, b):
            a = (int(a[0]), int(a[1]))
            b = (int(b[0]), int(b[1]))
            if a == b:
                return
            w = math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))
            graph.setdefault(a, {})[b] = min(w, graph.setdefault(a, {}).get(b, w))
            graph.setdefault(b, {})[a] = min(w, graph.setdefault(b, {}).get(a, w))
            all_points.add(a)
            all_points.add(b)

        for subpath in branch_paths:
            coords = [(int(p[0]), int(p[1])) for p in np.asarray(subpath)]
            if not coords:
                continue
            all_points.update(coords)
            graph.setdefault(coords[0], {})
            for a, b in zip(coords[:-1], coords[1:]):
                add_edge(a, b)

        if not all_points:
            return [], (0, 0), (0, 0), 0.0
        if len(all_points) == 1:
            only = next(iter(all_points))
            return [only], only, only, 0.0

        endpoints = [p for p in all_points if len(graph.get(p, {})) <= 1]

        def rebuild(prev, start, end):
            path = [end]
            cur = end
            while cur != start and cur in prev:
                cur = prev[cur]
                path.append(cur)
            if path[-1] != start:
                return [start, end]
            path.reverse()
            return path

        if main_dist is not None:
            h, w = main_dist.shape
            def md(p):
                y, x = p
                if 0 <= y < h and 0 <= x < w:
                    return float(main_dist[y, x])
                return float('inf')
            # The first-order junction is the network point closest to the main root.
            start = min(all_points, key=lambda p: (md(p), p[0], p[1]))
            dist, prev = self._dijkstra_pixel_graph(graph, start)
            candidates = [p for p in endpoints if p in dist and p != start]
            if not candidates:
                candidates = [p for p in dist if p != start]
            end = max(candidates, key=lambda p: dist[p]) if candidates else start
            return rebuild(prev, start, end), start, end, float(dist.get(end, 0.0))

        # External root: use a two-sweep weighted-graph diameter approximation.
        seed = next(iter(all_points))
        dist0, _ = self._dijkstra_pixel_graph(graph, seed)
        a = max(dist0, key=dist0.get)
        dist1, prev1 = self._dijkstra_pixel_graph(graph, a)
        b = max(dist1, key=dist1.get)
        return rebuild(prev1, a, b), a, b, float(dist1.get(b, 0.0))

    def _merge_nonprimary_branch_networks(self, skeleton, branch_data, mainpath,
                                          main_dist, leaf_mask=None,
                                          target_start=3):
        """Classify non-primary branches without merging independent laterals.

        RAPID lateral-root rule:
          * Each branch that DIRECTLY attaches to the selected primary root is an
            independent first-order lateral-root seed.
          * A root that directly reaches the selected plant leaf/crown is also an
            independent lateral-root seed.  It must not be merged with a nearby
            primary-attached lateral merely because the downstream skeletons touch.
          * Second-/higher-order branches are merged into a first-order lateral only
            when their topology gives one clear parent seed.
          * When a downstream branch is similarly attributable to two different
            first-order seeds, it is kept as a separate display-only lateral instead
            of being used as a bridge that merges those roots.
          * A skeleton component with no connection to the primary root, selected
            leaf/crown, or any confirmed lateral is an External root.

        This differs from the previous implementation, which removed the primary
        path and then merged every connected non-primary component.  That could
        collapse two biologically independent lateral roots into one instance when
        they happened to touch farther away from the primary root.
        """
        if branch_data is None or len(branch_data) == 0:
            return [], target_start

        src_ids = branch_data["node-id-src"]
        dst_ids = branch_data["node-id-dst"]
        branch_lengths = branch_data["branch-distance"]
        main_set = set((int(p[0]), int(p[1])) for p in mainpath)

        branch_paths = {}
        primary_branches = set()
        primary_nodes = set()
        node_to_nonprimary = {}

        # First separate the selected primary-root edges from all other skeleton
        # edges.  A junction pixel may be shared by primary and lateral paths; that
        # shared endpoint is exactly what we use to identify first-order roots.
        for i in range(len(branch_data)):
            sp = np.asarray(skeleton.path_coordinates(i), dtype=np.int32)
            branch_paths[i] = sp
            coords = [(int(p[0]), int(p[1])) for p in sp]
            overlap = (sum(p in main_set for p in coords) / float(len(coords))) if coords else 0.0
            s = int(src_ids[i])
            d = int(dst_ids[i])
            if overlap >= 0.80:
                primary_branches.add(i)
                primary_nodes.add(s)
                primary_nodes.add(d)
                continue
            node_to_nonprimary.setdefault(s, set()).add(i)
            node_to_nonprimary.setdefault(d, set()).add(i)

        nonprimary = set(range(len(branch_data))) - primary_branches
        if not nonprimary:
            return [], target_start

        # Branch-to-branch topology after the primary edges have been removed.
        adjacency = {i: set() for i in nonprimary}
        for node, branches in node_to_nonprimary.items():
            bs = list(branches)
            for a in range(len(bs)):
                for b in range(a + 1, len(bs)):
                    i, j = bs[a], bs[b]
                    if i in nonprimary and j in nonprimary:
                        adjacency[i].add(j)
                        adjacency[j].add(i)

        # Distance to the selected leaf/crown.  The root mask intentionally removes
        # leaf pixels, so a real root-to-leaf attachment can end one or two pixels
        # outside the leaf mask.  Only branch ENDPOINTS use this small tolerance;
        # a lateral merely passing near a leaf is not promoted to a new seed.
        leaf_dist = None
        if leaf_mask is not None and np.any(leaf_mask):
            leaf_bin = (np.asarray(leaf_mask) > 0).astype(np.uint8)
            leaf_dist = cv2.distanceTransform((leaf_bin == 0).astype(np.uint8),
                                              cv2.DIST_L2, 5)

        root_radius = float(getattr(self, '_root_radius_est', 2.0))
        leaf_contact_limit = float(max(1.0, min(2.0, 0.50 * root_radius + 0.75)))
        min_network_length = float(max(8.0, min(15.0, 4.0 * root_radius)))
        min_display_length = float(max(3.0, min(7.0, 2.0 * root_radius)))
        h, w = main_dist.shape

        def md(p):
            y, x = int(p[0]), int(p[1])
            if 0 <= y < h and 0 <= x < w:
                return float(main_dist[y, x])
            return float('inf')

        def ld(p):
            if leaf_dist is None:
                return float('inf')
            y, x = int(p[0]), int(p[1])
            if 0 <= y < leaf_dist.shape[0] and 0 <= x < leaf_dist.shape[1]:
                return float(leaf_dist[y, x])
            return float('inf')

        # A seed is a NON-primary Skan branch that itself attaches directly to
        # primary or leaf.  Every such seed remains an independent lateral root.
        seed_kind = {}
        for bi in sorted(nonprimary):
            sp = branch_paths[bi]
            coords = [(int(p[0]), int(p[1])) for p in sp]
            if not coords:
                continue
            s = int(src_ids[bi])
            d = int(dst_ids[bi])
            touches_primary = bool({s, d} & primary_nodes) or any(p in main_set for p in coords)

            endpoints = [coords[0], coords[-1]]
            touches_leaf = bool(leaf_dist is not None and
                                min(ld(p) for p in endpoints) <= leaf_contact_limit)

            if touches_primary and touches_leaf:
                seed_kind[bi] = 'primary+leaf'
            elif touches_primary:
                seed_kind[bi] = 'primary'
            elif touches_leaf:
                seed_kind[bi] = 'leaf'

        # Connected components are still useful, but they are no longer themselves
        # lateral instances.  Components with several direct-attachment seeds are
        # PARTITIONED, not merged.
        pending = set(nonprimary)
        components = []
        while pending:
            seed = pending.pop()
            comp = {seed}
            stack = [seed]
            while stack:
                bi = stack.pop()
                for nb in adjacency.get(bi, ()):
                    if nb in pending:
                        pending.remove(nb)
                        comp.add(nb)
                        stack.append(nb)
            components.append(sorted(comp))

        def branch_step_cost(a, b):
            # Approximate centreline distance from the centre of branch a to b.
            # This gives physical branch length more weight than raw branch count.
            la = max(1.0, float(branch_lengths[a]))
            lb = max(1.0, float(branch_lengths[b]))
            return 0.5 * (la + lb)

        def distances_from_seed(comp_set, seed):
            """Dijkstra on the branch graph, never traversing another seed."""
            dist = {seed: 0.0}
            heap = [(0.0, seed)]
            while heap:
                cur_d, cur = heapq.heappop(heap)
                if cur_d != dist.get(cur):
                    continue
                for nb in adjacency.get(cur, ()):
                    if nb not in comp_set:
                        continue
                    # Direct-primary/direct-leaf roots are hard boundaries.  A seed
                    # may not propagate ownership THROUGH a different seed.
                    if nb in seed_kind and nb != seed:
                        continue
                    nd = cur_d + branch_step_cost(cur, nb)
                    if nd < dist.get(nb, float('inf')):
                        dist[nb] = nd
                        heapq.heappush(heap, (nd, nb))
            return dist

        def connected_subgroups(branches):
            """Return topology-connected subgroups within an arbitrary branch set."""
            todo = set(branches)
            out = []
            while todo:
                s = todo.pop()
                g = {s}
                stack = [s]
                while stack:
                    cur = stack.pop()
                    for nb in adjacency.get(cur, ()):
                        if nb in todo:
                            todo.remove(nb)
                            g.add(nb)
                            stack.append(nb)
                out.append(sorted(g))
            return out

        # (class, branches, anchor kind, display_only, seed branch)
        classified_groups = []

        for comp in components:
            comp_set = set(comp)
            seeds = [b for b in comp if b in seed_kind]

            if not seeds:
                # No primary/leaf attachment anywhere in this component: it is an
                # actual disconnected/external root network.
                classified_groups.append((4, comp, 'external', False, None))
                continue

            if len(seeds) == 1:
                # Unambiguous tree: first-order seed + every higher-order descendant.
                s = seeds[0]
                classified_groups.append((3, comp, seed_kind[s], False, s))
                continue

            # Several independent first-order roots happen to share one downstream
            # non-primary component.  Keep every direct-attachment seed independent
            # and assign descendants only when one parent is clearly closer.
            dist_maps = {s: distances_from_seed(comp_set, s) for s in seeds}
            owned = {s: {s} for s in seeds}
            ambiguous = set()

            for bi in comp:
                if bi in seed_kind:
                    continue
                candidates = []
                for s in seeds:
                    d = dist_maps[s].get(bi, float('inf'))
                    if np.isfinite(d):
                        candidates.append((float(d), s))
                candidates.sort()
                if not candidates:
                    ambiguous.add(bi)
                    continue
                if len(candidates) == 1:
                    owned[candidates[0][1]].add(bi)
                    continue

                best_d, best_seed = candidates[0]
                second_d = candidates[1][0]
                # Near-equidistant means the topology does not give a trustworthy
                # biological parent.  Do not let such a bridge merge two laterals.
                ambiguity_margin = max(1.5, 0.10 * max(best_d, 1.0))
                if (second_d - best_d) <= ambiguity_margin:
                    ambiguous.add(bi)
                else:
                    owned[best_seed].add(bi)

            for s in seeds:
                # Ownership can be topologically split around an ambiguous bridge;
                # only keep the subgroup containing the direct-attachment seed.
                groups = connected_subgroups(owned[s])
                seed_group = next((g for g in groups if s in g), [s])
                classified_groups.append((3, seed_group, seed_kind[s], False, s))
                # Any detached owned islands are uncertain rather than force-merged.
                for g in groups:
                    if g is not seed_group and s not in g:
                        classified_groups.append((3, g, 'ambiguous', True, None))

            # Ambiguous pieces are drawn as their own lateral instance and marked
            # display-only.  They cannot bridge/merge the confirmed first-order roots.
            for g in connected_subgroups(ambiguous):
                classified_groups.append((3, g, 'ambiguous', True, None))

        results = []
        target = int(target_start)
        debug = getattr(self, 'lateral_debug', False)

        for root_class, group, anchor_kind, display_only, seed_branch in classified_groups:
            paths = [branch_paths[i] for i in group if len(branch_paths[i])]
            if not paths:
                continue
            total_length = float(sum(float(branch_lengths[i]) for i in group))

            # Confirmed direct-attachment laterals are allowed to be somewhat shorter
            # than the historical threshold.  Very short/ambiguous fragments are
            # retained only for display and excluded from quantitative summaries.
            if total_length <= min_display_length:
                if debug:
                    print(f"[RootNetwork] branches={group} reject=tiny total={total_length:.1f} min={min_display_length:.1f}", flush=True)
                continue
            if root_class == 3 and total_length <= min_network_length:
                display_only = True
            if root_class == 4 and total_length <= min_network_length:
                # Tiny disconnected components are much more likely segmentation noise.
                if debug:
                    print(f"[RootNetwork] external branches={group} reject=short total={total_length:.1f}", flush=True)
                continue

            all_coords = []
            for sp in paths:
                all_coords.extend((int(p[0]), int(p[1])) for p in sp)
            if not all_coords:
                continue

            # For primary-attached roots use primary distance to orient the
            # representative path.  For leaf-only roots use leaf distance instead.
            anchor_dist = None
            if root_class == 3:
                if 'primary' in anchor_kind:
                    anchor_dist = main_dist
                elif anchor_kind == 'leaf' and leaf_dist is not None:
                    anchor_dist = leaf_dist

            rep_path, attach, tip, rep_distance = self._representative_network_path(
                paths, anchor_dist
            )
            if len(rep_path) < 2:
                rep_path = list(dict.fromkeys(all_coords))
            if not rep_path:
                continue

            unique_coords = list(dict.fromkeys(all_coords))
            prop_mask = self.Pt2Mask(unique_coords, target)
            lr = 0
            if root_class == 3 and anchor_kind != 'ambiguous' and attach != tip:
                lr = 1 if int(tip[1]) >= int(attach[1]) else -1

            endpoint_distance = math.hypot(float(tip[0] - attach[0]),
                                             float(tip[1] - attach[1]))
            prop = RootProp(total_length, endpoint_distance, rep_path, target, tip,
                            prop_mask, lr, root_class=root_class)
            prop.representative_path_length = float(rep_distance)
            prop.label_value = target
            prop.branch_indices = tuple(group)
            prop.merged_branch_levels = (len(group) > 1)
            prop.attachment_kind = anchor_kind
            prop.seed_branch_index = seed_branch
            prop.direct_primary_attachment = bool('primary' in anchor_kind)
            prop.direct_leaf_attachment = bool('leaf' in anchor_kind)
            prop.ambiguous_attachment = bool(anchor_kind == 'ambiguous')
            prop.analysis_excluded = bool(display_only)
            prop.display_only = bool(display_only)
            prop.attachment_point = attach if (root_class == 3 and anchor_kind != 'ambiguous') else None
            results.append(prop)

            if debug:
                cname = 'lateral' if root_class == 3 else 'external'
                print(f"[RootNetwork] {cname} label={target} anchor={anchor_kind} "
                      f"branches={group} total={total_length:.1f} display_only={display_only}",
                      flush=True)
            target += 1

        return results, target

    def _leaf_count_legacy(self, leaf):
        """Legacy leaf-count estimator used by the original ICML method."""
        leaf = (np.asarray(leaf) > 0).astype(np.uint8)
        if not np.any(leaf):
            return 0
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(leaf, kernel, iterations=3)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(eroded_mask, connectivity=8)
        return max(num_labels - 1, 1)

    def _split_leaf_instances_distance_core(self, leaf):
        """Estimate leaf instances with an over-segmentation correction pass.

        The first pass preserves sensitivity to genuinely small leaves.  If a
        connected leaf group is split into too many candidates (>5), a stricter
        second pass is triggered automatically.  The second pass requires
        stronger/thicker cores and larger inter-marker spacing before watershed.
        This is intended to prevent one broad leaf from being counted several
        times while avoiding a hard-coded final leaf count of 3 or 4.
        """
        leaf = (np.asarray(leaf) > 0).astype(np.uint8)
        labels_out = np.zeros(leaf.shape, dtype=np.uint16)
        if not np.any(leaf):
            return labels_out, 0, False

        if getattr(self._image, 'ndim', 0) == 3:
            hsv = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].astype(np.float32)
        else:
            saturation = np.full(leaf.shape, 255.0, dtype=np.float32)

        legacy_count = self._leaf_count_legacy(leaf)
        n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            leaf, connectivity=8)
        next_label = 1
        any_split = False
        component_debug = []

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
                labels_out[y:y+h, x:x+w][comp > 0] = next_label
                component_debug.append((cc, 1, 1, 1, False))
                next_label += 1
                continue

            dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            dist_s = cv2.GaussianBlur(dist, (0, 0), 1.0)
            max_dist = float(dist_s.max())
            if max_dist <= 1.0:
                labels_out[y:y+h, x:x+w][comp > 0] = next_label
                component_debug.append((cc, 1, 1, 1, False))
                next_label += 1
                continue

            win = int(np.clip(round(max_dist * 0.8), 7, 25))
            if win % 2 == 0:
                win += 1
            peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win))
            local_max = dist_s >= (cv2.dilate(dist_s, peak_kernel) - 1e-6)
            peak_mask = (local_max & (dist_s >= max(2.0, 0.22 * max_dist))
                         & (comp > 0)).astype(np.uint8)

            n_peak, peak_labels, _, _ = cv2.connectedComponentsWithStats(
                peak_mask, connectivity=8)
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
                local_sat = (float(np.mean(sat_crop[disk]))
                             if np.any(disk) else float(sat_crop[rr, cc2]))

                annulus_r1 = max(3.0, 1.10 * radius)
                annulus_r2 = max(5.0, 2.00 * radius)
                d2 = (yy - rr) ** 2 + (xx - cc2) ** 2
                annulus = ((d2 >= annulus_r1 ** 2) &
                           (d2 <= annulus_r2 ** 2) & (comp > 0))
                if np.any(annulus):
                    local_background = float(np.percentile(dist_s[annulus], 75))
                else:
                    local_background = 0.0
                prominence = radius - local_background
                prominent_peak = prominence >= max(1.0, 0.14 * radius)

                strong_core = radius >= 0.60 * max_dist
                smaller_leaf_core = (
                    radius >= 0.42 * max_dist and
                    local_sat >= max(20.0, 0.65 * sat_reference) and
                    prominent_peak
                )
                if strong_core or smaller_leaf_core:
                    # radius, row, col, saturation, prominence
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

            # Pass 1: sensitive enough to retain small true leaves.
            selected_first = suppress_peaks(candidates, 0.60)
            selected = selected_first
            strict_used = False

            # Pass 2: only when pass 1 clearly suggests over-segmentation.
            # It uses stronger cores and considerably larger marker separation.
            if len(selected_first) > 5:
                strict_pool = []
                for cand in candidates:
                    radius, rr, cc2, local_sat, prominence = cand
                    strict_core = (
                        radius >= 0.50 * max_dist and
                        local_sat >= max(20.0, 0.72 * sat_reference) and
                        prominence >= max(1.2, 0.18 * radius)
                    )
                    # Very strong broad cores are kept even if saturation is weak.
                    very_strong_core = radius >= 0.68 * max_dist
                    if strict_core or very_strong_core:
                        strict_pool.append(cand)
                strict_selected = suppress_peaks(strict_pool, 0.82)
                # Use the strict result when it still contains enough structure
                # to represent multiple leaves.  Otherwise keep pass 1 rather
                # than collapsing the whole crown to a single leaf.
                if 2 <= len(strict_selected) < len(selected_first):
                    selected = strict_selected
                    strict_used = True

            if len(selected) <= 1:
                labels_out[y:y+h, x:x+w][comp > 0] = next_label
                component_debug.append(
                    (cc, len(candidates), len(selected_first), 1, strict_used))
                next_label += 1
                continue

            # Safety cap only for watershed marker count.  It is not a forced
            # final biological leaf count; abnormal cases are reduced by ranking
            # marker strength rather than truncating the reported count afterward.
            if len(selected) > 6:
                selected = selected[:6]
                strict_used = True

            markers = np.zeros(comp.shape, dtype=np.int32)
            for marker_id, (_, rr, cc2, _, _) in enumerate(selected, start=1):
                markers[rr, cc2] = marker_id

            ws = watershed(-dist_s, markers=markers, mask=(comp > 0), connectivity=2)
            ws_ids, ws_counts = np.unique(ws[ws > 0], return_counts=True)

            min_region = max(40, int(round(area * 0.025)))
            valid_pairs = [(int(i), int(a)) for i, a in zip(ws_ids, ws_counts)
                           if int(a) >= min_region]

            # Stronger post-watershed area consistency.  Tiny fragments are
            # commonly caused by a secondary core on one leaf blade.
            if len(valid_pairs) >= 3:
                region_areas = np.asarray([a for _, a in valid_pairs], dtype=np.float32)
                median_area = float(np.median(region_areas))
                tiny_cut = max(float(min_region), 0.25 * median_area)
                filtered_pairs = [(i, a) for i, a in valid_pairs if a >= tiny_cut]
                if len(filtered_pairs) >= 2:
                    valid_pairs = filtered_pairs

            valid_ids = [i for i, _ in valid_pairs]
            if len(valid_ids) <= 1:
                labels_out[y:y+h, x:x+w][comp > 0] = next_label
                component_debug.append(
                    (cc, len(candidates), len(selected_first), 1, strict_used))
                next_label += 1
                continue

            if len(valid_ids) > 6:
                valid_pairs.sort(key=lambda t: t[1], reverse=True)
                valid_ids = [i for i, _ in valid_pairs[:6]]

            any_split = True
            crop_out = labels_out[y:y+h, x:x+w]
            for wid in valid_ids:
                crop_out[ws == wid] = next_label
                next_label += 1

            missed = (comp > 0) & (crop_out == 0)
            if np.any(missed):
                fill = crop_out.copy()
                for _ in range(max(h, w)):
                    if not np.any(missed):
                        break
                    dil = cv2.dilate(fill, np.ones((3, 3), np.uint16), iterations=1)
                    take = missed & (dil > 0)
                    if not np.any(take):
                        break
                    fill[take] = dil[take]
                    missed[take] = False
                crop_out[:] = fill

            component_debug.append(
                (cc, len(candidates), len(selected_first), len(valid_ids), strict_used))

        count = int(next_label - 1)

        # Only fall back in truly pathological cases.  A second-pass result of
        # 3--5 leaves is therefore preserved instead of silently reverting to the
        # old erosion-based count.
        fallback = count <= 0 or count > max(8, legacy_count * 3)
        final_count = legacy_count if fallback else count

        # Temporary diagnostic output for tuning.  Set self.leaf_debug = False
        # anywhere after constructing AnalyseRoot if console output is unwanted.
        if getattr(self, 'leaf_debug', False):
            print('[LeafCount] legacy=%d components=%d final=%d fallback=%s' %
                  (legacy_count, max(n_cc - 1, 0), final_count, str(fallback)))
            for cc, cand_n, first_n, result_n, strict_used in component_debug:
                print('  component=%d candidates=%d first_pass=%d second_pass=%s result=%d' %
                      (cc, cand_n, first_n, 'yes' if strict_used else 'no', result_n))

        if fallback:
            return np.zeros_like(labels_out), legacy_count, False
        return labels_out, count, any_split

    def GetLeafFromMaskImproved(self, leaf):
        """Leaf RootProp for ICML-improve with distance-core instance count."""
        leaf = (np.asarray(leaf) > 0).astype(np.uint8)
        area = int(np.sum(leaf))
        if area <= 0:
            return RootProp(0, 0, [], 1, (0, 0), leaf, 0)
        coord = np.nonzero(leaf)
        cx, cy = np.mean(coord[1]), np.mean(coord[0])
        instance_labels, leaf_count, used_core = self._split_leaf_instances_distance_core(leaf)
        leaf_count = max(int(leaf_count), 1)
        leaf_L = np.nonzero(leaf)
        leaf_path = list(zip(leaf_L[0], leaf_L[1]))
        prop = RootProp(area, area / leaf_count, leaf_path, 1,
                        (int(cy), int(cx)), leaf, 0)
        # Preserve the instance result for future visualisation/export without
        # changing the existing RootProp/CSV interface.
        prop.leaf_count = leaf_count
        prop.leaf_instance_labels = instance_labels
        prop.leaf_count_method = 'distance-core' if used_core else 'legacy-fallback'
        return prop

    def GetLeafFromMask(self,leaf):
        area = np.sum(leaf)
        coord = np.nonzero(leaf)
        cx,cy = np.mean(coord[1]),np.mean(coord[0])
        #navigate = (cy,cx)
        kernel = np.ones((5,5), np.uint8)
        eroded_mask = cv2.erode(leaf, kernel, iterations=3)
        num_labels, labels, stats, centroids  = cv2.connectedComponentsWithStats(eroded_mask, connectivity=8)

        # connectedComponentsWithStats counts the background as label 0.
        # Store the mean leaf area in ``distance`` so the existing GUI can
        # recover the leaf count as area / mean_area without an off-by-one.
        leaf_count = max(num_labels - 1, 1)
        leaf_L = np.nonzero(leaf)
        Leafpath = list(zip(leaf_L[0], leaf_L[1]))
        return RootProp(area, area / leaf_count, Leafpath, 1,
                        (int(cy), int(cx)), leaf, 0)
    def segment_leaves(self,image):
     # 加载图像
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 设置颜色阈值（绿色部分）
        lower_green = np.array([30, 30, 30])  # 最低颜色阈值
        upper_green = np.array([80, 255, 255])  # 最高颜色阈值

        # 创建掩膜
        mask = cv2.inRange(image_hsv, lower_green, upper_green)
    
        # 进行形态学操作，以去除小的噪点并填充区域
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
        # 根据掩膜提取树叶区域
        result = cv2.bitwise_and(image, image, mask=mask)
        MAX_V =np.max(mask)
        if MAX_V>0:
            mask = mask/ MAX_V
        return result,mask.astype(np.uint8)

    def segment_leaves_improved(self, image):
        """Slightly broader HSV leaf candidate extraction for ICML-improve only.

        The legacy ICML thresholds are intentionally left unchanged.  The improved
        branch widens hue/saturation/value acceptance modestly to recover pale,
        yellow-green, darker, or less saturated leaf pixels.
        """
        if image.ndim != 3:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 20, 20], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(image_hsv, lower_green, upper_green)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return (mask > 0).astype(np.uint8)

    def _filter_leaf_by_astar_endpoint(self, leaf, astar_endpoint):
        """Remove at most one leaf outlier using the A* crown/root endpoint.

        Assumption used by ICML-improve: the YOLO ROI contains the target plant and
        at most one neighbouring leaf group.  A single leaf group is therefore
        always retained.  With two or more groups, only the group whose centroid is
        farthest from the A* endpoint is removed; all other groups are preserved.

        ``astar_endpoint`` is in (row, col) coordinates.
        """
        leaf = (np.asarray(leaf) > 0).astype(np.uint8)
        if not np.any(leaf) or astar_endpoint is None:
            return leaf

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(leaf, connectivity=8)
        component_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0]
        if len(component_ids) <= 1:
            return leaf

        ey, ex = float(astar_endpoint[0]), float(astar_endpoint[1])
        distances = {}
        for i in component_ids:
            cx, cy = centroids[i]
            distances[i] = math.hypot(float(cx) - ex, float(cy) - ey)

        outlier_id = max(component_ids, key=lambda i: distances[i])
        filtered = leaf.copy()
        filtered[labels == outlier_id] = 0

        # Conservative fallback: never let the outlier rule erase all leaf pixels.
        return filtered if np.any(filtered) else leaf
   
    def _filter_leaf_components_by_primary_crown(self, leaf, crown_endpoint):
        """Keep leaf components spatially associated with the selected plant crown.

        IMPORTANT: this function is called only *after* the primary-root path has
        already been selected.  It therefore changes the displayed/measured leaf
        mask only and does not feed back into primary-root selection.

        The leaf candidate mask may contain green leaves from neighbouring plants
        when a YOLO ROI is large.  We choose the connected component that comes
        closest to the primary-root crown endpoint as the seed component, then keep
        other leaf components only when they are also within an adaptive crown
        neighbourhood.  This allows several disconnected leaves of the same plant
        to survive while removing distant upper/right or lower neighbouring leaves.

        ``crown_endpoint`` uses (row, col) coordinates.
        """
        leaf = (np.asarray(leaf) > 0).astype(np.uint8)
        if not np.any(leaf) or crown_endpoint is None:
            return leaf

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            leaf, connectivity=8
        )
        ids = [i for i in range(1, n)
               if int(stats[i, cv2.CC_STAT_AREA]) >= 20]
        if len(ids) <= 1:
            return leaf

        h, w = leaf.shape
        cy0, cx0 = float(crown_endpoint[0]), float(crown_endpoint[1])

        # Distance from the crown endpoint to the nearest pixel of each component.
        info = []
        for i in ids:
            ys, xs = np.where(labels == i)
            if len(xs) == 0:
                continue
            d2 = (ys.astype(np.float64) - cy0) ** 2 + \
                 (xs.astype(np.float64) - cx0) ** 2
            min_dist = float(np.sqrt(np.min(d2)))
            area = int(stats[i, cv2.CC_STAT_AREA])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            ccx, ccy = centroids[i]
            info.append({
                'id': i, 'min_dist': min_dist, 'area': area,
                'bbox': (x, y, bw, bh),
                'centroid': (float(ccx), float(ccy)),
            })

        if not info:
            return leaf

        # The component physically closest to the selected primary crown is the
        # anchor leaf group for this plant.  Crown evidence outranks component area.
        seed = min(info, key=lambda z: (z['min_dist'], -z['area']))
        sx, sy, sbw, sbh = seed['bbox']
        eq_radius = math.sqrt(max(float(seed['area']), 1.0) / math.pi)
        seed_diag = math.hypot(float(sbw), float(sbh))

        # Adaptive but deliberately bounded.  Large outlier-separated masks must
        # not inflate this threshold (unlike the previous whole-mask leaf radius).
        keep_dist = max(45.0,
                        2.4 * eq_radius,
                        0.55 * seed_diag,
                        0.08 * float(min(h, w)))
        keep_dist = min(keep_dist, 0.24 * float(min(h, w)))

        # A second, slightly looser threshold is allowed for a substantial leaf
        # component whose centroid is still in the crown neighbourhood.  This helps
        # retain a disconnected true leaf while rejecting tiny distant green noise.
        large_area = max(120, int(round(0.12 * seed['area'])))
        loose_dist = min(1.35 * keep_dist, 0.30 * float(min(h, w)))

        kept = []
        debug_rows = []
        for z in info:
            i = z['id']
            d = z['min_dist']
            ccx, ccy = z['centroid']
            centroid_dist = math.hypot(ccx - cx0, ccy - cy0)

            keep = (i == seed['id']) or (d <= keep_dist)
            reason = 'seed' if i == seed['id'] else ('crown-near' if keep else 'far')

            if (not keep and z['area'] >= large_area and d <= loose_dist
                    and centroid_dist <= 1.6 * loose_dist):
                keep = True
                reason = 'large-near'

            if keep:
                kept.append(i)
            debug_rows.append((i, keep, reason, z['area'], d,
                               centroid_dist, z['bbox']))

        # Conservative safety: seed always survives, therefore this normally cannot
        # be empty.  Keep the original mask rather than erase all leaves if an
        # unexpected numerical/component issue occurs.
        if not kept:
            return leaf

        filtered = np.zeros_like(leaf, dtype=np.uint8)
        for i in kept:
            filtered[labels == i] = 1

        if getattr(self, 'leaf_debug', False):
            print('[LeafPlantFilter] crown=%s components=%d kept=%d seed=%d '
                  'seed_area=%d keep_dist=%.1f loose_dist=%.1f' %
                  (str((int(round(cy0)), int(round(cx0)))), len(info), len(kept),
                   seed['id'], seed['area'], keep_dist, loose_dist), flush=True)
            for i, keep, reason, area, d, dc, bbox in sorted(
                    debug_rows, key=lambda r: r[4]):
                print('  leaf_cc=%d keep=%s reason=%s area=%d crown_min=%.1f '
                      'centroid_dist=%.1f bbox=%s' %
                      (i, str(keep), reason, area, d, dc, str(bbox)), flush=True)

        return filtered

    def Predict(self,out):
        skeleton0 = morphology.skeletonize(out)
        g0, c0 = skeleton_to_csgraph(skeleton0)
        skeleton =Skeleton(skeleton0, spacing=1)
        #branch_data = summarize(skeleton)
        #srcIdxSet =branch_data["node-id-src"]
        #desIdxSet = branch_data["node-id-dst"]
        #orig  = np.zeros_like(out)
        extend  = np.zeros_like(out)   

        for i in range(skeleton.n_paths):
            coords_to_wipe = skeleton.path_coordinates(i)

            ext = self.extend_curve(coords_to_wipe,3)
            for pt in ext:
                extend = cv2.circle(extend, (int(pt[1]),int(pt[0])), 1, 1, 1)
        return extend
    def GetAppromRT(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        center, size, angle = rect
        if angle < -45:
            angle += 90
        return angle, rect
    
    
    def PostProcess(self, image,
                min_area=300,
                use_relative_area=False,
                relative_ratio=0.01,
                rect_dilate_iter=3):
        img = (image > 0).astype(np.uint8).copy()

        # 只为估计矩形做适度膨胀
        if rect_dilate_iter > 0:
            kernel_rect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            img_for_rect = cv2.dilate(img, kernel_rect, iterations=rect_dilate_iter)
        else:
            img_for_rect = img.copy()
    
        # 用膨胀图算矩形
        _, rect = self.GetAppromRT(img_for_rect)
        if rect is None:
            return np.zeros_like(img)
        box = cv2.boxPoints(rect).astype(np.float32)
    
        # 连通域在“原图”上做
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    
        if use_relative_area and nlabels > 1:
            maxA = stats[1:, cv2.CC_STAT_AREA].max()
            min_area = max(min_area, maxA * relative_ratio)
    
        mask = np.zeros_like(img)
        for i in range(1, nlabels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            pt = tuple(centroids[i])
            if cv2.pointPolygonTest(box, pt, False) < 0:
                continue
            mask[labels == i] = 1
    
        # ★ 整个流程里不再对 mask 做腐蚀
        return mask


    def Get(self):
        #print("Get")
        return self.mRootProp
    def GetLabel(self):
        """Return a single-channel DISPLAY projection of the organ masks.

        The RootProp masks themselves are the source of truth and may overlap.
        Therefore a pixel may simultaneously belong to the primary root and one
        lateral root.  A uint8 display image cannot encode two instance ids at
        one pixel, so this function only chooses which colour/id is visible.

        Display order:
            Leaf -> Primary root -> Lateral roots

        Thus a primary/lateral junction is shown with the lateral-root colour,
        while the same junction pixel remains present in BOTH RootProp masks.
        No masks are added numerically, so overlap can never create a false
        instance value such as 1 + 2 = 3.
        """
        label = np.zeros_like(self.mlabel, dtype=np.uint8)

        # Paint leaf first, then primary, then lateral instances.  This changes
        # only the display projection; prop.mask is never modified here.
        ordered_indices = []

        ordered_indices.extend(
            i for i, prop in enumerate(self.mRootProp)
            if int(getattr(prop, "type", 0)) == 1
        )
        ordered_indices.extend(
            i for i, prop in enumerate(self.mRootProp)
            if int(getattr(prop, "type", 0)) == 2
        )
        ordered_indices.extend(
            i for i, prop in enumerate(self.mRootProp)
            if int(getattr(prop, "type", 0)) >= 3
        )

        # Keep any uncommon/unclassified entries visible as a final fallback.
        ordered_indices.extend(
            i for i, prop in enumerate(self.mRootProp)
            if int(getattr(prop, "type", 0)) <= 0
        )

        for i in ordered_indices:
            prop = self.mRootProp[i]
            mask = np.asarray(prop.mask) > 0
            if np.any(mask):
                label[mask] = i + 1

        return label

    def GetInstanceMasks(self):
        """Return independent binary masks; overlaps are intentionally preserved."""
        return [((np.asarray(prop.mask) > 0).astype(np.uint8))
                for prop in self.mRootProp]

    def GetLabelLoad(self):
        return self.mlabel
        #return self.mlabel
    def Process(self,skeleton0,st,ed):
        # A sparse adjacency matrix.
        # Two pixels are adjacent in the graph if both are painted.
        adjacency = dok_matrix((skeleton0.shape[0] * skeleton0.shape[1],
                    skeleton0.shape[0] * skeleton0.shape[1]), dtype=bool)

        # The following lines fills the adjacency matrix by
        directions = list(itertools.product([0, 1, -1], [0, 1, -1]))
        for i in range(1, skeleton0.shape[0] - 1):
            for j in range(1, skeleton0.shape[1] - 1):
                if not skeleton0[i, j]:
                    continue
        
                for y_diff, x_diff in directions:
                    if skeleton0[i + y_diff, j + x_diff]:
                        adjacency[self.to_index(skeleton0,i, j),
                          self.to_index(skeleton0,i + y_diff, j + x_diff)] = True

        # We chose two arbitrary points, which we know are connected
        #st,ed  = (160, 74) ,(99, 779)
        
        source = self.to_index(skeleton0,st[1],st[0]) 
        target = self.to_index(skeleton0,ed[1],ed[0])
        
        # Compute the shortest path between the source and all other points in the image
        _, predecessors = dijkstra(adjacency, directed=False, indices=[source],
                               unweighted=True, return_predecessors=True)
        
        # Constructs the path between source and target
        pixel_index = target
        pixels_path = []
        
        while pixel_index != source:
            if len(pixels_path) > skeleton0.shape[0]:
                return False,[]
            pixels_path.append(pixel_index)
            pixel_index = predecessors[0, pixel_index]
        return True, pixels_path
    def to_index(self,img, y, x):
        return y * img.shape[1] + x
    def to_coordinates(self,img,index):
        return int(index / img.shape[1]), int(index % img.shape[1])
    def JudgeMain(self,one,bronchi):
        distmax = 5
        #print(one,bronchi)
        for cur  in bronchi:
            dist = np.fabs(one[0]- cur[0])+np.fabs(one[1]- cur[1])
            #print(dist)
            if dist < distmax:
                return True
        return False
    def JudgeMainPath(self,B,A):
        overlap_count = 0
        half_B_length = len(B) // 2

        for point in A:
            if point in B:
                overlap_count += 1

        if overlap_count < half_B_length:
            return False
        else:
            return True
    def ProcessManual(self,mask,target):
        g0, c0 = skeleton_to_csgraph(mask)
        #branch_data = summarize(Skeleton(skeleton0))
        st,ed  = (int(c0[1][1]),int(c0[0][1])),(int(c0[1][-1]),int(c0[0][-1]))  
        mainPath = list()
        #mainlabels.append((ed[1],ed[0]))
        ret,pixels_path = self.Process(mask,st,ed)
        
        for pixel_index in pixels_path:
            i, j = self.to_coordinates(mask,pixel_index)
            mainPath.append((i,j))
        mainPath.append((st[1],st[0]))
        
        mainPath = self.checkMore(mainPath,mask,st,0)
        mainPath = self.checkMore(mainPath,mask,ed,1)
        sumLong=0
        for k in range(0,len(mainPath)-1):
            x,y = mainPath[k]
            m,n =mainPath[k+1]
            dist = (x-m)*(x-m)+(y-n)*(y-n)
            sumLong = sumLong+np.sqrt(dist)
        st = mainPath[-1]
        ed = mainPath[0]
        distSE = (st[0]-ed[0])*(st[0]-ed[0])+(st[1]-ed[1])*(st[1]-ed[1])
        mRootProp = RootProp(sumLong, np.sqrt(distSE), mainPath, target,st, mask,0)#self.Pt2Mask(mainPath, target)
        return mRootProp
       
    def JudgeOverirde(self,subBronchi,pixels_path):
        leng = len(subBronchi)
        count = 0
        for one in subBronchi:
            if self.JudgeMain(one,pixels_path):
                count = count +1
        return count > leng//2


    def compute_main_direction(self, mainpath, tail_len=5, sType="tail"):
        """
        估计 mainpath 某一端的“外延方向”。
    
        约定：
            - mainpath[0] 纵坐标最大（几何“尾部”）
            - sType="tail": 用最前面的 tail_len 段，从内部指向 mainpath[0] 外侧
            - sType="head": 用最后面的 tail_len 段，从内部指向 mainpath[-1] 外侧
        """
        mp = np.asarray(mainpath, dtype=float)
        if len(mp) < 2:
            return None
    
        k = min(tail_len, len(mp) - 1)
    
        if sType == "tail":
            # 用最前 k+1 个点，方向：mp[k] -> mp[0]
            seg = mp[:k+1]
            v = seg[0] - seg[-1]
        else:  # "head"
            # 用最后 k+1 个点，方向：mp[-(k+1)] -> mp[-1]
            seg = mp[-(k+1):]
            v = seg[-1] - seg[0]
    
        n = np.linalg.norm(v)
        if n == 0:
            return None
        return v / n


    def extension_score(self, mainpath, subpath, sType="tail",
                        max_gap=30,
                        max_angle_deg=45,
                        tail_len=5,
                        overlap_eps=1.5,
                        max_full_overlap_ratio=0.8):
        """
        判断一条 subpath 是否可以接到 mainpath 的某一端（由 sType 决定）。
    
        约定：
            - mainpath[0] 纵坐标最大（几何“尾部”在 0 下标）
            - sType="tail"  -> 以 mainpath[0] 为基准
            - sType="head"  -> 以 mainpath[-1] 为基准
    
        返回:
            (gap, angle, info) 或 None
    
        info:
            {
                "gap": gap,
                "angle": angle,
                "idx_near": 被选为连接点的 subpath 索引,
                "sType": sType,
            }
        """
        mp = np.asarray(mainpath, dtype=float)
        sp = np.asarray(subpath, dtype=float)
    
        if len(mp) < 2 or len(sp) < 2:
            return None
    
        # 1. mainpath 指定端的主方向（外延方向）
        dir_main = self.compute_main_direction(mp, tail_len=tail_len, sType=sType)
        if dir_main is None:
            return None
    
        # 2. 基准点：注意和你的约定对应
        if sType == "tail":
            # 几何尾部在 mainpath[0]
            p_base = mp[0]
        else:  # "head"
            p_base = mp[-1]
    
        # 3. 整体重叠比例，过滤几乎完全重复的 subpath
        diff_all = mp[:, None, :] - sp[None, :, :]       # (len(mp), len(sp), 2)
        dists_all = np.linalg.norm(diff_all, axis=2)     # (len(mp), len(sp))
        min_dist_each_sp = dists_all.min(axis=0)         # 每个 subpath 点到 mainpath 的最近距离
        overlap_ratio = np.mean(min_dist_each_sp < overlap_eps)
    
        if overlap_ratio > max_full_overlap_ratio:
            # 绝大部分点都贴在 mainpath 上，当作重复 path，直接忽略
            return None
    
        # 4. 先找出所有离基准点不太远的点（<= max_gap），再加上两个端点兜底
        dists_to_base = np.linalg.norm(sp - p_base, axis=1)  # (len(sp),)
        candidate_indices = np.where(dists_to_base <= max_gap)[0]
    
        extra = np.array([0, len(sp) - 1], dtype=int)
        candidate_indices = np.unique(np.concatenate([candidate_indices, extra]))
    
        if candidate_indices.size == 0:
            return None
    
        # 5. 任意 idx 的切线估计（用邻域线段平均）
        def tangent_at(sp_arr, idx, k=3):
            """
            在路径 sp_arr 上任意索引 idx 处，用邻域线段平均估计切线方向。
            k 是向两边看的最多线段数。
            """
            sp_arr = np.asarray(sp_arr, dtype=float)
            n = len(sp_arr)
            if n < 2:
                return None
    
            start = max(0, idx - k)
            end = min(n - 2, idx + k)  # 线段 i -> i+1
    
            vectors = []
            for i in range(start, end + 1):
                v = sp_arr[i + 1] - sp_arr[i]
                nv = np.linalg.norm(v)
                if nv > 0:
                    vectors.append(v)
    
            if not vectors:
                return None
    
            v_sum = np.sum(vectors, axis=0)
            nv = np.linalg.norm(v_sum)
            if nv == 0:
                return None
            return v_sum / nv
    
        # 6. 遍历所有候选点，选 gap 最小，其次 angle 最小的
        best_candidate = None  # (gap, angle, idx)
    
        for idx in candidate_indices:
            q = sp[idx]
            v_base = q - p_base
            gap = float(np.linalg.norm(v_base))
            if gap > max_gap:
                continue
    
            # ★ 只考虑“在 dir_main 前方”的点，避免把后面的东西接过来
            proj = float(np.dot(v_base, dir_main))  # dir_main 是单位向量
            if proj <= 0:
                # 在基准点的“背后”，不应接在这一端
                continue
    
            # --- 1) 连线方向角（p_base -> q）---
            nv_base = np.linalg.norm(v_base)
            if nv_base == 0:
                angle_base = 0.0
            else:
                cos1_raw = float(np.clip(np.dot(v_base / nv_base, dir_main), -1.0, 1.0))
                # 有了 proj>0，cos1_raw 已经不会是负的了，一般不用 abs
                angle_base = math.degrees(math.acos(cos1_raw))
    
            # --- 2) 局部切线方向角 ---
            t_dir = tangent_at(sp, idx, k=3)
            if t_dir is None:
                angle_tan = angle_base
            else:
                cos2_raw = float(np.clip(np.dot(t_dir, dir_main), -1.0, 1.0))
                # 局部切线方向容易“前后反”，这里只看平行程度
                cos2 = abs(cos2_raw)
                angle_tan = math.degrees(math.acos(cos2))
    
            angle = max(angle_base, angle_tan)
            if angle > max_angle_deg:
                continue
    
            if best_candidate is None:
                best_candidate = (gap, angle, idx)
            else:
                bg, ba, _ = best_candidate
                if gap < bg or (gap == bg and angle < ba):
                    best_candidate = (gap, angle, idx)
    
        if best_candidate is None:
            return None
    
        gap, angle, idx_near = best_candidate
        info = {
            "gap": gap,
            "angle": angle,
            "idx_near": idx_near,
            "sType": sType,
        }
        return gap, angle, info


    def bresenham_line(self, p0, p1):
        """
        Bresenham 直线：返回两个整数点之间所有像素坐标。
        """
        x0, y0 = map(int, p0)
        x1, y1 = map(int, p1)
        points = []
    
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
    
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
    
        return points
    
    def merge_one(self, mainpath, subpath, idx_near, tail_len=5, sType="tail",
                  overlap_eps=1.5):
        """
        把一条通过筛选的 subpath 接到 mainpath 上，
        用 idx_near 指示的连接端点（0 或 len-1），自动判断 subpath 的遍历方向。
    
        - sType: "tail" 接在 mainpath[-1] 后面; "head" 接在 mainpath[0] 前面
        - 支持 mainpath 和 subpath 存在重合：会自动去掉重合的那一段，只保留延伸部分
        """
        mp = np.asarray(mainpath, dtype=float)
        sp = np.asarray(subpath, dtype=float)
    
        if sType == "tail":
            p_base = mp[0]
        else:
            p_base = mp[-1]
    
        dir_main = self.compute_main_direction(mp, tail_len=tail_len, sType=sType)
        if dir_main is None:
            dir_main = np.array([1.0, 0.0])
    
        # ---- 先确定 subpath 的遍历顺序（向 mainpath 外延方向）----
        if len(sp) == 1:
            ordered = sp
        else:
            candidates = []
            if idx_near + 1 < len(sp):
                v_fwd = sp[idx_near + 1] - sp[idx_near]
                nv = np.linalg.norm(v_fwd)
                if nv > 0:
                    candidates.append((np.dot(v_fwd / nv, dir_main), +1))
            if idx_near - 1 >= 0:
                v_bwd = sp[idx_near - 1] - sp[idx_near]
                nv = np.linalg.norm(v_bwd)
                if nv > 0:
                    candidates.append((np.dot(v_bwd / nv, dir_main), -1))
    
            if candidates:
                _, direction = max(candidates, key=lambda x: x[0])
            else:
                direction = +1   # 实在判断不出来就默认正方向
    
            if direction == +1:
                inds = range(idx_near, len(sp))
            else:
                inds = range(idx_near, -1, -1)
            ordered = sp[list(inds)]
    
        # ---- 去掉 ordered 里与 mainpath 重叠的前缀，只保留真正延伸部分 ----
        mp_arr = mp
        ordered_arr = np.asarray(ordered, dtype=float)
    
        # 对 ordered 中每个点，计算到 mainpath 的最近距离
        diff = mp_arr[:, None, :] - ordered_arr[None, :, :]
        dists = np.linalg.norm(diff, axis=2)  # (len(mp), len(ordered))
        min_dist_each = dists.min(axis=0)     # (len(ordered),)
    
        # 找出从前往后，第一个“明显离开 mainpath”的点
        first_non_overlap_idx = 0
        for i, d in enumerate(min_dist_each):
            if d > overlap_eps:
                first_non_overlap_idx = i
                break
        else:
            # 整个 ordered 都紧贴 mainpath，说明基本没延伸，直接返回 mainpath 不变
            return [(int(round(p[0])), int(round(p[1]))) for p in mp_arr]
    
        ordered_trim = ordered_arr[first_non_overlap_idx:]
    
        if len(ordered_trim) == 0:
            # 没有真正延伸部分
            return [(int(round(p[0])), int(round(p[1]))) for p in mp_arr]
    
        # ---- 根据 sType 拼接 + 补线 ----
        if sType == "tail":
            q_start = ordered_trim[0]
            bridge = self.bresenham_line(p_base, q_start)
            if bridge:
                bridge = bridge[1:-1]   # 去掉两端，避免重复点
    
            merged = [(int(round(p[0])), int(round(p[1]))) for p in mp_arr]
            merged.extend(bridge)
            for p in ordered_trim:
                merged.append((int(round(p[0])), int(round(p[1]))))
        else:
            # 头部：我们希望最终顺序是：延伸外侧 -> ... -> 原 mainpath
            # ordered_trim 当前是从靠 mainpath 的点向外；接在前面时要反过来
            ordered_connect = ordered_trim[::-1]
            q_near_for_bridge = ordered_connect[-1]  # 靠 mainpath 的那个点
    
            bridge = self.bresenham_line(q_near_for_bridge, p_base)
            if bridge:
                bridge = bridge[1:-1]
    
            merged = []
            for p in ordered_connect:
                merged.append((int(round(p[0])), int(round(p[1]))))
            merged.extend(bridge)
            for p in mp_arr:
                merged.append((int(round(p[0])), int(round(p[1]))))
        return merged
    def extend_with_candidates(self, mainpath, subpaths, sType,
                               max_gap=30,
                               max_angle_deg=60,
                               max_perp_dist=8,  # 现在不用了，可以删掉
                               tail_len=5,
                               overlap_eps=1.5):
        """
        从多条 subpath 中，反复挑出“最可能是 mainpath 指定一端延长”的一条，接上去。
        sType 决定是接头("head")还是接尾("tail")。
    
        返回：
            merged_mainpath : 最终合并后的主路径
            used_subpaths   : 被合并进来的那些 subpath（按合并顺序）
            remaining       : 剩下没用上的 subpath
        """
        mp = list(mainpath)
        remaining = list(subpaths)
        used = []
    
        while True:
            best_j = None
            best_info = None
            best_gap = None
            best_angle = None
    
            for j, sp in enumerate(remaining):
                res = self.extension_score(
                    mp, sp,
                    sType=sType,
                    max_gap=max_gap,
                    max_angle_deg=max_angle_deg,
                    tail_len=tail_len,
                    overlap_eps=overlap_eps,
                )
                if res is None:
                    continue
    
                gap, angle, info = res
    
                if best_j is None:
                    best_j = j
                    best_info = info
                    best_gap = gap
                    best_angle = angle
                else:
                    if gap < best_gap or (gap == best_gap and angle < best_angle):
                        best_j = j
                        best_info = info
                        best_gap = gap
                        best_angle = angle
    
            # 没有任何合格的候选了
            if best_j is None:
                break
    
            # 把当前最好的这一条接上去
            mp = self.merge_one(
                mp,
                remaining[best_j],
                idx_near=best_info["idx_near"],
                tail_len=tail_len,
                sType=sType,
                overlap_eps=overlap_eps,
            )
            used.append(remaining.pop(best_j))
    
        return mp, used, remaining



    
    def ValidateAndCombine(self, leaf, mainpath,start_point, navigate):
        alpha = 1.25 # Contrast control (1.0-3.0)
        beta = 40 # Brightness control (0-100)
        gamma = dehaze(self._image)
        image0 = cv2.convertScaleAbs(gamma, alpha=alpha, beta=beta)

        gray = cv2.cvtColor(image0,cv2.COLOR_BGR2GRAY)
        gray =cv2.medianBlur(gray,3)
        th,out = cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow("out",out*255)
        out[leaf==1] =0
        #th,out = cv2.threshold(gray,th+10,1,cv2.THRESH_BINARY)
        #cv2.imshow("out-leaf",out*255)
        #cv2.waitKey()
        if self._bShowImg:
            # print("out")
            cv2.imshow("out",out*255)
            cv2.waitKey()
        #out = self.PostProcesLable(out)
        
        skeleton0 = morphology.skeletonize(out)
        g0, c0 = skeleton_to_csgraph(skeleton0)
        skeleton =Skeleton(skeleton0, spacing=1)
        branch_data = summarize(skeleton)
        subpaths = [skeleton.path_coordinates(i) for i in range(len(branch_data) - 1)]

        det = math.dist(navigate, mainpath[-1])
        if det > 100:
            mainpath, used_head, remaining_subpaths = self.extend_with_candidates(
                mainpath, subpaths, "head",
                max_gap=20, max_angle_deg=60, max_perp_dist=5, tail_len=5
            )
        else:
            remaining_subpaths = subpaths
        
        height, width = gray.shape
        bottom_det = math.dist((width // 2, height - 1), mainpath[0])
        if bottom_det > 100 and remaining_subpaths:
            mainpath, used_tail, remaining_subpaths = self.extend_with_candidates(
                mainpath, remaining_subpaths, "tail",
                max_gap=20, max_angle_deg=60, max_perp_dist=5, tail_len=5
            )
        
        return mainpath, branch_data, g0, c0, skeleton
    def _recover_weak_root_mask(self, gray, leaf_mask=None):
        """Recover faint root pixels without globally lowering the Otsu threshold.

        ICML-improve only.  Otsu pixels are treated as high-confidence root seeds.
        A second, slightly relaxed threshold plus a local-contrast test defines a
        weak candidate mask.  Morphological reconstruction grows only from the
        strong seeds through those weak candidates, so unrelated weak background
        regions are not introduced.  A very small closing step is used only to
        bridge residual 1--2 pixel gaps.

        Returns
        -------
        recovered : uint8 array (0/1)
            Root mask after weak-pixel recovery.
        otsu_threshold : float
            Otsu threshold used for the high-confidence seed mask.
        """
        g = np.asarray(gray, dtype=np.uint8)
        otsu_threshold, strong255 = cv2.threshold(
            g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        strong = (strong255 > 0).astype(np.uint8)

        if leaf_mask is not None and np.shape(leaf_mask) == np.shape(strong):
            leaf_bool = np.asarray(leaf_mask) > 0
            strong[leaf_bool] = 0
        else:
            leaf_bool = None

        if not np.any(strong):
            return strong, otsu_threshold

        # Relax the global threshold only moderately.  The amount scales with the
        # Otsu level but remains bounded so that dark background is not admitted.
        delta = int(np.clip(round(float(otsu_threshold) * 0.16), 12, 30))
        weak_threshold = max(0, int(round(float(otsu_threshold))) - delta)
        weak_global = g >= weak_threshold

        # A local threshold helps recover a faint root section that is still
        # slightly brighter than its immediate background, even when its absolute
        # intensity falls below the global Otsu threshold.
        min_dim = min(g.shape[:2])
        block = 31
        if min_dim < block:
            block = max(3, int(min_dim // 2) * 2 - 1)
        if block < 3:
            block = 3
        local255 = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, -2
        )
        local_bright = local255 > 0

        # Strong pixels are always retained.  New pixels must satisfy both the
        # relaxed global threshold and local brightness criterion.
        allowed = (strong > 0) | (weak_global & local_bright)
        if leaf_bool is not None:
            allowed[leaf_bool] = False

        allowed_u8 = allowed.astype(np.uint8)

        # Fill only very short interruptions in the weak support mask.  This is
        # intentionally much smaller than the old 10x10 post-processing kernel.
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        allowed_u8 = cv2.morphologyEx(allowed_u8, cv2.MORPH_CLOSE, small_kernel,
                                      iterations=1)
        if leaf_bool is not None:
            allowed_u8[leaf_bool] = 0

        # Binary morphological reconstruction: repeatedly dilate the strong root
        # seeds, but constrain every step to pixels supported by the weak mask.
        recovered = strong.copy()
        grow_kernel = np.ones((3, 3), dtype=np.uint8)
        max_iterations = max(g.shape[:2])
        for _ in range(max_iterations):
            grown = cv2.dilate(recovered, grow_kernel, iterations=1)
            grown = ((grown > 0) & (allowed_u8 > 0)).astype(np.uint8)
            if np.array_equal(grown, recovered):
                break
            recovered = grown

        # One final conservative close repairs single-pixel holes introduced by
        # colour variation.  No large-kernel connection is performed here.
        recovered = cv2.morphologyEx(recovered, cv2.MORPH_CLOSE, small_kernel,
                                     iterations=1)
        if leaf_bool is not None:
            recovered[leaf_bool] = 0
        return (recovered > 0).astype(np.uint8), otsu_threshold

    def _recover_lateral_mask_from_primary_otsu(self, gray, base_mask, mainpath,
                                                  leaf_mask=None):
        """Refine the root mask using an Otsu threshold learned around the primary root.

        The first ICML-improve pass still uses a whole-ROI Otsu threshold so that a
        reliable primary root can be found.  Once the primary centreline is known,
        this second pass samples the gray values in a tube around that centreline and
        computes another Otsu threshold there.  That local threshold is a much better
        reference for thin/faint lateral roots than the whole-ROI threshold.

        The threshold is deliberately relaxed a little for laterals, but candidate
        pixels are admitted only when they are connected back to the primary root by
        morphological reconstruction.  This is important: simply lowering Otsu over
        the entire ROI tends to introduce Petri-dish edges, ruler marks and bright
        background texture.

        Returns
        -------
        recovered : uint8 array (0/1)
            Primary-connected root mask with faint lateral-root candidates restored.
        diagnostics : dict
            Threshold/intensity information retained for optional debugging.
        """
        g = np.asarray(gray, dtype=np.uint8)
        base = (np.asarray(base_mask) > 0).astype(np.uint8)
        if g.ndim != 2 or base.shape != g.shape or mainpath is None or len(mainpath) < 2:
            return base, {}

        h, w = g.shape
        path_mask = np.zeros((h, w), dtype=np.uint8)
        valid_points = []
        for pt in mainpath:
            yy, xx = int(round(pt[0])), int(round(pt[1]))
            if 0 <= yy < h and 0 <= xx < w:
                path_mask[yy, xx] = 1
                valid_points.append((yy, xx))
        if len(valid_points) < 2:
            return base, {}

        root_radius = float(getattr(self, '_root_radius_est', 2.0))
        root_radius = float(np.clip(root_radius, 1.0, 12.0))
        core_r = int(np.clip(round(1.6 * root_radius), 2, 8))
        local_r = int(np.clip(round(5.0 * root_radius), 8, 32))

        core_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * core_r + 1, 2 * core_r + 1))
        local_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * local_r + 1, 2 * local_r + 1))
        core = cv2.dilate(path_mask, core_kernel, iterations=1) > 0
        local = cv2.dilate(path_mask, local_kernel, iterations=1) > 0

        if leaf_mask is not None and np.shape(leaf_mask) == g.shape:
            leaf_bool = np.asarray(leaf_mask) > 0
        else:
            leaf_bool = np.zeros(g.shape, dtype=bool)
        core &= ~leaf_bool
        local &= ~leaf_bool

        # Otsu is computed only around the selected primary root, not on the full
        # ROI.  The centreline median determines whether the root is brighter or
        # darker than its local background, although RAPID images are normally the
        # bright-root case.
        sample = g[local]
        path_vals = g[path_mask.astype(bool) & ~leaf_bool]
        ring = local & ~core
        ring_vals = g[ring]
        if sample.size < 64 or path_vals.size < 8:
            return base, {}

        primary_otsu, _ = cv2.threshold(
            sample.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        root_median = float(np.median(path_vals))
        bg_median = float(np.median(ring_vals)) if ring_vals.size else float(np.median(sample))
        bright_root = root_median >= bg_median

        # Lateral roots are commonly thinner and slightly darker than the primary
        # root.  Relax the primary-local Otsu threshold by a bounded amount rather
        # than applying an arbitrary global threshold.
        contrast = abs(root_median - bg_median)
        relax = int(np.clip(round(max(4.0, 0.18 * contrast, 0.055 * float(primary_otsu))),
                            4, 18))
        if bright_root:
            lateral_threshold = max(0, int(round(float(primary_otsu))) - relax)
            global_candidate = g >= lateral_threshold
            local255 = cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31 if min(h, w) >= 31 else max(3, (min(h, w)//2)*2-1), -2)
            local_support = local255 > 0
        else:
            lateral_threshold = min(255, int(round(float(primary_otsu))) + relax)
            global_candidate = g <= lateral_threshold
            local255 = cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31 if min(h, w) >= 31 else max(3, (min(h, w)//2)*2-1), -2)
            local_support = local255 > 0

        # Keep the existing reliable mask unconditionally; new pixels need both
        # primary-root-like intensity and local contrast support.
        allowed = (base > 0) | (global_candidate & local_support)
        allowed[leaf_bool] = False

        # Repair only tiny one/two-pixel interruptions.  Larger gaps should not be
        # bridged because they are more likely to be unrelated objects.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        allowed_u8 = cv2.morphologyEx(allowed.astype(np.uint8), cv2.MORPH_CLOSE,
                                      close_kernel, iterations=1)
        allowed_u8[leaf_bool] = 0

        # Seed the reconstruction on the known primary centreline.  A small dilation
        # makes the seed robust to a one-pixel skeleton shift after re-thresholding.
        seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seed = (cv2.dilate(path_mask, seed_kernel, iterations=1) > 0) & (allowed_u8 > 0)
        seed = seed.astype(np.uint8)
        if not np.any(seed):
            return base, {}

        recovered = seed.copy()
        grow_kernel = np.ones((3, 3), dtype=np.uint8)
        for _ in range(max(h, w)):
            grown = cv2.dilate(recovered, grow_kernel, iterations=1)
            grown = ((grown > 0) & (allowed_u8 > 0)).astype(np.uint8)
            if np.array_equal(grown, recovered):
                break
            recovered = grown

        # Keep all pixels that were already accepted by the first-pass mask.
        # The primary-guided Otsu stage is allowed to ADD faint connected lateral
        # pixels, but it must not erase a disconnected root that may later be
        # classified as an External root.  Newly thresholded remote background is
        # still excluded because only ``base`` is reintroduced here.
        recovered = cv2.morphologyEx(recovered, cv2.MORPH_CLOSE, close_kernel,
                                     iterations=1)
        recovered = np.maximum(recovered, base).astype(np.uint8)
        recovered[leaf_bool] = 0

        diagnostics = {
            'primary_otsu': float(primary_otsu),
            'lateral_threshold': int(lateral_threshold),
            'root_median': root_median,
            'background_median': bg_median,
            'bright_root': bool(bright_root),
            'added_pixels': int(np.count_nonzero((recovered > 0) & (base == 0))),
            'removed_remote_pixels': int(np.count_nonzero((base > 0) & (recovered == 0))),
        }
        self._primary_reference_otsu = diagnostics['primary_otsu']
        self._lateral_reference_threshold = diagnostics['lateral_threshold']
        self._lateral_mask_added_pixels = diagnostics['added_pixels']

        if getattr(self, 'lateral_debug', False):
            print('[LateralMask] primary_otsu=%.1f lateral_th=%d root_med=%.1f '
                  'bg_med=%.1f bright=%s added=%d removed_remote=%d' %
                  (diagnostics['primary_otsu'], diagnostics['lateral_threshold'],
                   diagnostics['root_median'], diagnostics['background_median'],
                   str(diagnostics['bright_root']), diagnostics['added_pixels'],
                   diagnostics['removed_remote_pixels']), flush=True)

        return (recovered > 0).astype(np.uint8), diagnostics

    def Processing(self):#_image is opencv type
        _dbg_total_t0 = time.perf_counter()
        # print(f"[DEBUG][Processing] start method={self._method} shape={self._image.shape}", flush=True)
        
        leaf = np.zeros(self._image.shape[:2], dtype=np.uint8)
        if len(self._image.shape)==2:
             gray = self._image.copy()#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif len(self._image.shape)==3:
             gray = self._image[:,:,0]#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
             result,leaf = self.segment_leaves(self._image)
        height,width = self._image.shape[0:2]
        navigate = (0, int(width /2))

        # ICML-NoAI detects leaves independently from the root using the
        # legacy HSV green-region method.  Keep the leaf as the first
        # RootProp so GetLabel() preserves the semantic convention:
        # 1=Leaf, 2=Primary root, 3+=Lateral roots.
        if self._method == 1 and np.any(leaf):
            leaf = self.Processleaf(leaf)
            if np.any(leaf):
                leaf_prop = self.GetLeafFromMask(leaf)
                self.mRootProp.append(leaf_prop)
                navigate = leaf_prop.pt

        if self._method ==1: #otsu for single root
            alpha = 1.25 # Contrast control (1.0-3.0)
            beta = 40 # Brightness control (0-100)
            gamma = dehaze(self._image)
            image0 = cv2.convertScaleAbs(gamma, alpha=alpha, beta=beta)

            if self._bShowImg:
                # print("image0")
                cv2.imshow("image0",image0)
            if image0.ndim == 3:
                gray = cv2.cvtColor(image0,cv2.COLOR_BGR2GRAY)
            else:
                gray = np.asarray(image0, dtype=np.uint8)
            gray = cv2.medianBlur(gray,3)
            th,out = cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #cv2.imshow("out",out*255)
            out[leaf==1] =0

            if self._bShowImg:
                # print("out")
                cv2.imshow("out",out*255)
                cv2.waitKey()
            out = self.PostProcesLable(out)

            self.mlabel = out.copy()
            skeleton0 = morphology.skeletonize(out)
            skeleton =Skeleton(skeleton0, spacing=1)
            g0, c0 = skeleton_to_csgraph(skeleton0)
            branch_data = summarize(Skeleton(skeleton0))
            #branch_type =  branch_data["branch-type"] 
            srcIdxSet =branch_data["node-id-src"]
            desIdxSet = branch_data["node-id-dst"]
            # Skan terminology: branch-distance is the geodesic/path length,
            # while euclidean-distance is the straight endpoint distance.
            branchPath = branch_data["branch-distance"]
            branchDistance = branch_data["euclidean-distance"]
            st,ed  = (int(c0[1][1]),int(c0[0][1])),(int(c0[1][-1]),int(c0[0][-1]))
            #print(st,ed)
            ret,pixels_path = self.Process(skeleton0,st,ed)
            #imMainR = self._image.copy()
            mainlabels = list()#mainlabels is used to save all points on main root
            mainlabels.append((ed[1],ed[0]))
            for pixel_index in pixels_path:
                i, j = self.to_coordinates(image0,pixel_index)
                mainlabels.append((i,j))
            mainlabels.append((st[1],st[0]))
            sumLong=0
            for k in range(0,len(mainlabels)-1):
                x,y = mainlabels[k]
                m,n =mainlabels[k+1]
                dist = (x-m)*(x-m)+(y-n)*(y-n)
                sumLong = sumLong+np.sqrt(dist)
            dist = (st[0]-ed[0])*(st[0]-ed[0])+(st[1]-ed[1])*(st[1]-ed[1])
            # Type 1 is reserved for leaves.  A root-only Otsu image therefore
            # contains one primary root (type 2) followed by lateral roots (>=3).
            self.mRootProp.append(RootProp(sumLong, np.sqrt(dist),
                                           mainlabels.copy(), 2, st,
                                           self.Pt2Mask(mainlabels, 2), 1))
            target = 3
            for i in range(0,len(branch_data)-1):# does not consider start and end point
                    if branchPath[i]>15:
                       srcIdx = srcIdxSet[i]
                       desIdx = desIdxSet[i]
                       
                       coordss = (int(c0[0][srcIdx]),int(c0[1][srcIdx]))#point coordinate
                       coorddd = (int(c0[0][desIdx]),int(c0[1][desIdx]))#point coordinate
                       bss = self.JudgeMain(coordss,mainlabels)
                       bdd = self.JudgeMain(coorddd,mainlabels)
                       subpath  = skeleton.path_coordinates(i)
                       if self.JudgeOverirde(subpath,mainlabels):
                           #print("Not a lateral root")
                           continue
                       elif bss or bdd:
                           pt, direct =0,(0,0)
                           #direct = 1 if bss else (-1 if bdd else 0)
                           direct = 1 if bss else -1
                           pt = coorddd if bss else coordss
                           self.mRootProp.append(RootProp(branchPath[i], branchDistance[i], subpath.copy(), target,pt,self.Pt2Mask(subpath,target),direct))
                           target = target +1
        elif self._method ==5: # ICML-improve: broader leaf HSV + A*-endpoint leaf filtering + lowest-neighbour extension
            # ICML-improve keeps the original ICML root segmentation.  Leaf
            # candidates use a slightly broader HSV range, while neighbouring-leaf
            # rejection is deferred until the A* crown endpoint has been found.
            _t = time.perf_counter()
            # print("[DEBUG][M5] start leaf segmentation", flush=True)
            if self._image.ndim == 3:
                leaf = self.segment_leaves_improved(self._image)
            else:
                leaf = np.zeros(self._image.shape[:2], dtype=np.uint8)
            # print(f"[DEBUG][M5] leaf segmentation done {time.perf_counter()-_t:.3f}s pixels={int(np.count_nonzero(leaf))}", flush=True)
            _t = time.perf_counter()
            # print("[DEBUG][M5] start Processleaf", flush=True)
            leaf = self.Processleaf(leaf) if np.any(leaf) else leaf
            # print(f"[DEBUG][M5] Processleaf done {time.perf_counter()-_t:.3f}s pixels={int(np.count_nonzero(leaf))}", flush=True)
            raw_leaf = leaf.copy()

            alpha = 1.25
            beta = 40
            gamma = dehaze(self._image)
            image0 = cv2.convertScaleAbs(gamma, alpha=alpha, beta=beta)
            if image0.ndim == 3:
                gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            else:
                gray = np.asarray(image0, dtype=np.uint8)
            gray = cv2.medianBlur(gray, 3)

            # ICML-improve: use Otsu as the high-confidence root seed, then
            # recover locally faint root pixels by constrained weak-threshold
            # reconstruction.  This repairs colour-induced breaks without simply
            # lowering the threshold over the whole ROI.
            _t = time.perf_counter()
            # print("[DEBUG][M5] start weak-root reconstruction", flush=True)
            out, otsu_th = self._recover_weak_root_mask(gray, raw_leaf)
            # print(f"[DEBUG][M5] weak-root reconstruction done {time.perf_counter()-_t:.3f}s root_pixels={int(np.count_nonzero(out))} otsu={otsu_th}", flush=True)

            if self._bShowImg:
                _, strong_dbg = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                strong_dbg[raw_leaf > 0] = 0
                cv2.imshow("ICML-improve Otsu seed", strong_dbg)
                cv2.imshow("ICML-improve recovered root", out.astype(np.uint8) * 255)
                # print("ICML-improve Otsu threshold:", otsu_th)

            # Keep the existing ICML connected-component cleanup after recovery.
            _t = time.perf_counter()
            # print("[DEBUG][M5] start PostProcesLableLeafAware", flush=True)
            out = self.PostProcesLableLeafAware(out, raw_leaf)
            # print(f"[DEBUG][M5] PostProcesLableLeafAware done {time.perf_counter()-_t:.3f}s root_pixels={int(np.count_nonzero(out))}", flush=True)
            self.mlabel = out.copy()
            if not np.any(out):
                return

            _t = time.perf_counter()
            # print("[DEBUG][M5] start skeletonize", flush=True)
            skeleton0 = morphology.skeletonize(out)
            # print(f"[DEBUG][M5] skeletonize done {time.perf_counter()-_t:.3f}s skeleton_pixels={int(np.count_nonzero(skeleton0))}", flush=True)
            if not np.any(skeleton0):
                return
            # Primary-root selection is crown-outward rather than lowest-point
            # inward.  This avoids using a Petri-dish rim as the starting point.
            _t = time.perf_counter()
            # print("[DEBUG][M5] start crown-outward primary path", flush=True)
            mainpath, crown_anchor, crown_target, rejected_primary_components = self._primary_path_with_line_rejection(
                skeleton0, raw_leaf, out, max_retries=3
            )
            # print(f"[DEBUG][M5] primary path done {time.perf_counter()-_t:.3f}s path_points={len(mainpath)}", flush=True)

            # The selector now has an internal nearest-crown fallback, so a
            # non-empty skeleton is never handed back to the old global-lowest
            # A* rule.  That rule was responsible for the long false primary root
            # seen on Petri-dish edges.  If path construction itself fails, retain
            # the ROI/leaf analysis but do not invent a remote primary root.
            if len(mainpath) < 2:
                print("[PrimaryPath] no usable crown-related primary path; "
                      "skip root classification for this ROI (ROI remains saveable)",
                      flush=True)
                if np.any(raw_leaf):
                    leaf = raw_leaf.copy()
                    leaf_prop = self.GetLeafFromMaskImproved(leaf)
                    self.mRootProp.append(leaf_prop)
                return

            # Second-pass segmentation for lateral roots.  The first pass above
            # is intentionally conservative and is used to establish a trustworthy
            # primary centreline.  Now learn an Otsu threshold from the neighbourhood
            # of that primary root and use it to recover faint, primary-connected
            # lateral-root pixels that the whole-ROI threshold may have missed.
            refined_out, lateral_mask_diag = self._recover_lateral_mask_from_primary_otsu(
                gray, out, mainpath, raw_leaf
            )
            if np.any(refined_out) and not np.array_equal(refined_out, out):
                refined_skeleton = morphology.skeletonize(refined_out)
                if np.any(refined_skeleton):
                    refined_mainpath, refined_crown_anchor, refined_crown_target, _ = \
                        self._primary_path_with_line_rejection(
                            refined_skeleton, raw_leaf, refined_out, max_retries=3
                        )
                    # Use the second pass only when the primary can still be found.
                    # This prevents a permissive lateral threshold from degrading a
                    # previously valid primary-root result.
                    if len(refined_mainpath) >= 2:
                        out = refined_out
                        skeleton0 = refined_skeleton
                        mainpath = refined_mainpath
                        crown_anchor = refined_crown_anchor
                        crown_target = refined_crown_target
                        self.mlabel = out.copy()

            # Keep the historical RootProp orientation distal-tip -> crown, because
            # later code and exported measurements were written around that order.
            # _primary_path_with_line_rejection returns crown -> distal-tip.
            mainpath = list(reversed(mainpath))
            astar_endpoint = mainpath[-1]

            # Leaf ownership is resolved only AFTER the primary root has already
            # been selected.  Therefore this removes neighbouring-plant leaf
            # components without changing any of the root-selection logic above.
            _t = time.perf_counter()
            # print("[DEBUG][M5] start leaf plant-component filter", flush=True)
            leaf = self._filter_leaf_components_by_primary_crown(raw_leaf, astar_endpoint)
            # print(f"[DEBUG][M5] leaf plant-component filter done {time.perf_counter()-_t:.3f}s pixels={int(np.count_nonzero(leaf))}", flush=True)
            if np.any(leaf):
                _t = time.perf_counter()
                # print("[DEBUG][M5] start GetLeafFromMaskImproved / leaf count", flush=True)
                leaf_prop = self.GetLeafFromMaskImproved(leaf)
                # print(f"[DEBUG][M5] GetLeafFromMaskImproved done {time.perf_counter()-_t:.3f}s", flush=True)
                self.mRootProp.append(leaf_prop)

            _t = time.perf_counter()
            # print("[DEBUG][M5] start Skan Skeleton", flush=True)
            skeleton = Skeleton(skeleton0, spacing=1)
            # print(f"[DEBUG][M5] Skan Skeleton done {time.perf_counter()-_t:.3f}s", flush=True)
            _t = time.perf_counter()
            # print("[DEBUG][M5] start skeleton_to_csgraph", flush=True)
            g0, c0 = skeleton_to_csgraph(skeleton0)
            # print(f"[DEBUG][M5] skeleton_to_csgraph done {time.perf_counter()-_t:.3f}s", flush=True)
            _t = time.perf_counter()
            # print("[DEBUG][M5] start skan.summarize", flush=True)
            branch_data = summarize(skeleton)
            # print(f"[DEBUG][M5] skan.summarize done {time.perf_counter()-_t:.3f}s branches={len(branch_data)}", flush=True)
            srcIdxSet = branch_data["node-id-src"]
            desIdxSet = branch_data["node-id-dst"]
            branchPath = branch_data["branch-distance"]
            branchDistance = branch_data["euclidean-distance"]

            sumLong = 0.0
            for k in range(len(mainpath) - 1):
                x, y = mainpath[k]
                m, n = mainpath[k + 1]
                sumLong += math.hypot(x - m, y - n)
            st = mainpath[0]
            ed = mainpath[-1]
            euclid = math.hypot(st[0] - ed[0], st[1] - ed[1])
            self.mRootProp.append(RootProp(sumLong, euclid, mainpath.copy(), 2, st,
                                           self.Pt2Mask(mainpath, 2), 1))

            target = 3
            _t_branch = time.perf_counter()

            h0, w0 = skeleton0.shape
            main_set = set((int(p[0]), int(p[1])) for p in mainpath)
            main_zero = np.ones((h0, w0), dtype=np.uint8)
            for py, px in main_set:
                if 0 <= py < h0 and 0 <= px < w0:
                    main_zero[py, px] = 0
            main_dist = cv2.distanceTransform(main_zero, cv2.DIST_L2, 5)

            # Keep every root that directly attaches to primary/leaf as an
            # independent lateral seed.  Only unambiguous higher-order descendants
            # are merged into that seed; ambiguous bridges are display-only.
            merged_props, target = self._merge_nonprimary_branch_networks(
                skeleton, branch_data, mainpath, main_dist, leaf_mask=leaf,
                target_start=target
            )
            self.mRootProp.extend(merged_props)

            if getattr(self, 'lateral_debug', False):
                n_lat = sum(int(getattr(p, 'root_class', 3)) == 3 for p in merged_props)
                n_ext = sum(int(getattr(p, 'root_class', 3)) == 4 for p in merged_props)
                print(f"[RootNetwork] merged classification done laterals={n_lat} external={n_ext}", flush=True)
        elif self._method ==2: #multiotsu
            alpha = 1.25 # Contrast control (1.0-3.0)
            beta = 0 # Brightness control (0-100)
            gamma = dehaze(gray)
            image0 = cv2.convertScaleAbs(gamma, alpha=alpha, beta=beta)        
            if self._bShowImg:
                cv2.imshow("Gamma",image0)
            gray = cv2.GaussianBlur(gray,(5, 5), 0)
            thresholds = threshold_multiotsu(gray)
            bw = np.where(gray>thresholds[1],1,0)
            if self._bShowImg:
                cv2.imshow("bw",bw.astype(np.uint8)*255)
                cv2.waitKey()
            contours, _ = cv2.findContours(bw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_contour_area = 50 
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
            out = np.zeros_like(bw)
            cv2.drawContours(out, filtered_contours, -1, 1, thickness=cv2.FILLED)   
            self.mlabel = out
        elif self._method ==3: #u-net 
                #print("Deep learning")
                #alpha = 1.75 # Contrast control (1.0-3.0)
                #beta = 50 # Brightness control (0-100)
                #image0 = cv2.convertScaleAbs(self._image, alpha=alpha, beta=beta)
                
                inimg = cv2.resize(self._image,(IMAGE_SIZE,IMAGE_SIZE), interpolation = cv2.INTER_LINEAR_EXACT)
                out = self.Unet.predict_segmentation(inimg)
                leafm = out==2
                rootm = out ==1
                
                
                leaf = cv2.resize(leafm.astype(np.uint8),(width,height), interpolation = cv2.INTER_LINEAR_EXACT)
                root = cv2.resize(rootm.astype(np.uint8),(width,height), interpolation = cv2.INTER_LINEAR_EXACT)
                #out[out==2]=0
                out =  root.copy()
                
                if self._bShowImg:
                    #print("out")
                    cv2.imshow("leaf",leaf.astype(np.uint8)*255)
                    cv2.imshow("root",root.astype(np.uint8)*255)
                    
        elif self._method ==4: #u-net guilded
               burimage = cv2.GaussianBlur(self._image, (5, 5), 0)
               inimg = burimage#cv2.cvtColor(burimage, cv2.COLOR_BGR2RGB)
               out = self.Unet.predict_segmentation(inimg)
               leafm = out==2
               rootm = out ==1
               leaf = cv2.resize(leafm.astype(np.uint8),(width,height), interpolation = cv2.INTER_LINEAR_EXACT)
               root = cv2.resize(rootm.astype(np.uint8),(width,height), interpolation = cv2.INTER_LINEAR_EXACT)
               leaf = self.Processleaf(leaf)
               binm =  leaf+root
               self.labelU = binm
               gray = cv2.cvtColor(self._image,cv2.COLOR_BGR2GRAY)
               clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
               egray = clahe.apply(gray)
               if np.sum(rootm)>300:
                  th,out =  self.threshold_optimization(egray,binm)
               else:
                  th,out = cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
               out[leaf==1]=0
               out = self.PostProcess(out)
                
               if np.sum(leaf)>0:
                    prop = self.GetLeafFromMask(leaf)
                    navigate = prop.pt
                    self.mRootProp.append(prop)   
               self.mlabel = out
               skeleton0 = morphology.skeletonize(out)
               
               #endpoints = find_endpoints(skeleton0)
               #pairs = self.match_endpoints(endpoints, max_dist=20)
               #skeleton_filled = self.draw_kalman_bridges(skeleton0.copy(), pairs)
               
               
               white_pixels = np.argwhere(skeleton0 == 1)
               # 找到最低点的坐标
               start_point = tuple(white_pixels[np.argmax(white_pixels[:, 0])])
               mainpath= self.astar_closest(skeleton0, start_point, navigate)
               #mainpath = self.checkMore(mainpath,skeleton0, start_point,1)# check neighbor for more 
               mainpath,branch_data, g0, c0, skeleton = self.ValidateAndCombine(leaf,mainpath,start_point, navigate)
               
 
               sumLong=0
               for k in range(0,len(mainpath)-1):
                       x,y = mainpath[k]
                       m,n = mainpath[k+1]
                       dist = (x-m)*(x-m)+(y-n)*(y-n)
                       sumLong = sumLong+np.sqrt(dist)
               ed = mainpath[-1]
               st = mainpath[0]
               dist = (st[0]-ed[0])*(st[0]-ed[0])+(st[1]-ed[1])*(st[1]-ed[1])
               pt = (start_point[0],start_point[1]+10)
               self.mRootProp.append(RootProp(sumLong, np.sqrt(dist), mainpath, 2,pt,self.Pt2Mask(mainpath,2),0))
               
               srcIdxSet =branch_data["node-id-src"]
               desIdxSet = branch_data["node-id-dst"]
               branchPath= branch_data["branch-distance"]
               branchDistance= branch_data["euclidean-distance"]
               
               target =3
               for i in range(0,len(branch_data)-1):# does not consider start and end point
                    if branchPath[i]>15:
                       srcIdx = srcIdxSet[i]
                       desIdx = desIdxSet[i]
                       
                       coordss = (int(c0[0][srcIdx]),int(c0[1][srcIdx]))#point coordinate
                       coorddd = (int(c0[0][desIdx]),int(c0[1][desIdx]))#point coordinate
                       bss = self.JudgeMain(coordss,mainpath)
                       bdd = self.JudgeMain(coorddd,mainpath)
                       subpath  = skeleton.path_coordinates(i)
                       if self.JudgeOverirde(subpath,mainpath):
                           #print("Not a lateral root")
                           continue
                       elif bss or bdd:
                           pt, direct =0,(0,0)
                           #direct = 1 if bss else (-1 if bdd else 0)
                           direct = 1 if bss else -1
                           pt = coorddd if bss else coordss
                           self.mRootProp.append(RootProp(branchPath[i], branchDistance[i], subpath.copy(), target,pt,self.Pt2Mask(subpath,target),direct))
                           target = target +1
    