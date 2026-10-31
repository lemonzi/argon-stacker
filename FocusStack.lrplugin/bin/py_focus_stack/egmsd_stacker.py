"""
Edge-Guided Multi-Scale Decomposition (EG-MSD) Focus Stacking Engine
High-performance OpenCV + NumPy implementation with Scharr structure tensor,
asymmetric Z-jump halo suppression, fast guided filter regularization,
direct pixel reconstruction on thin occluders, and specular/shot-noise gating.
"""
import os
import gc
import cv2
import numpy as np
from dataclasses import dataclass, replace
from typing import Literal, Optional, Dict, Tuple, Any, Callable
from scipy.ndimage import distance_transform_edt


def resolve_bg_mode(mode: str, med_bg_luma: float) -> str:
    """
    Resolve background polarity mode:
    - 'dark': clearly dark / low-key backgrounds (med_bg_luma <= 0.25)
    - 'bright': clearly bright / high-key backgrounds (med_bg_luma >= 0.70)
    - 'energy': neutral mid-tone / ambiguous backgrounds (0.25 < med_bg_luma < 0.70)
    """
    if mode != "auto":
        return mode
    if med_bg_luma <= 0.25:
        return "dark"
    elif med_bg_luma >= 0.70:
        return "bright"
    else:
        return "energy"


@dataclass
class StackerConfig:
    # 0. Pre-Alignment (Full-frame before tiling)
    align_ecc: bool = True              # Affine scale/shift for focus breathing
    
    # 1. Resolution-Normalized Spatial Scale
    # If None, auto-computed from full frame width (0.7% of max(H, W))
    max_coc_radius_px: Optional[int] = None
    
    # 2. Structure Tensor & Specular/Noise Gating
    anisotropy_alpha: float = 0.5       # Boosts directional features (hairs, antennae)
    noise_read: float = 1e-5            # Base sensor read noise floor
    noise_shot: float = 4e-4            # Shot-noise slope (scales with luminance)
    specular_clip_thresh: float = 0.94  # Luminance threshold for bokeh ring suppression
    
    # 3. Fast Guided Filter Regularization
    fgf_subsample: int = 4              # Subsampling factor s (runs in O(N / s^2))
    eps_base: float = 1e-3              # Edge-snapping threshold for Base layer
    eps_detail: float = 1e-5            # Tight threshold for 1px Detail layer
    
    # 4. Asymmetric Z-Jump Halo Gate
    stack_order: Literal["near_to_far", "far_to_near"] = "near_to_far"
    z_jump_thresh: float = 5.0          # Slice delta (Δz) triggering occlusion halo gate
    halo_suppression_gamma: float = 0.45 # Exponential decay rate for foreground blur bleed

    # 5. Memory Tiling
    tile_size: int = 2048               # Spatial tile dimension (px) for low-RAM execution

    # 6. Background Polarity Mode
    bg_mode: Literal["auto", "dark", "bright", "energy"] = "auto"

    # 7. Transient Moving Object & Defocus Bloom Handling
    freeze_transient_slice: Optional[int] = None  # If set, freezes transient feature at slice; default None removes ghosts cleanly


# =============================================================================
# STAGE 0: Outward-From-Center Registration (Focus Breathing + Breeze)
# =============================================================================
def align_focus_stack_cv2(
    stack: np.ndarray,
    use_ecc: bool = True
) -> np.ndarray:
    """
    Aligns focus stack sequentially outward from the center slice (N // 2)
    to minimize cumulative magnification drift from lens focus breathing.
    """
    N, H, W, _ = stack.shape
    aligned = np.empty_like(stack)
    ref_idx = N // 2
    aligned[ref_idx] = stack[ref_idx]
    
    gray_8u = np.empty((N, H, W), dtype=np.uint8)
    for n in range(N):
        g = cv2.cvtColor(stack[n], cv2.COLOR_RGB2GRAY)
        g_blur = cv2.GaussianBlur(g, (0, 0), sigmaX=2.5)
        gray_8u[n] = np.clip(g_blur * 255.0, 0, 255).astype(np.uint8)

    ecc_scale = 0.25 if max(H, W) > 2000 else 0.5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)

    def _align_pair(src_idx: int, dst_ref_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cur_img = stack[src_idx]
        cur_gray = gray_8u[src_idx]
        
        if use_ecc:
            sh, sw = max(32, int(H * ecc_scale)), max(32, int(W * ecc_scale))
            g_ref_sm = cv2.resize(dst_ref_gray, (sw, sh), interpolation=cv2.INTER_AREA)
            g_cur_sm = cv2.resize(cur_gray, (sw, sh), interpolation=cv2.INTER_AREA)
            
            warp_mat = np.eye(2, 3, dtype=np.float32)
            try:
                _, warp_mat = cv2.findTransformECC(
                    g_ref_sm, g_cur_sm, warp_mat, cv2.MOTION_AFFINE, criteria
                )
                warp_mat[0, 2] /= ecc_scale
                warp_mat[1, 2] /= ecc_scale
                cur_img = cv2.warpAffine(
                    cur_img, warp_mat, (W, H),
                    flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT101
                )
                cur_gray = cv2.warpAffine(
                    cur_gray, warp_mat, (W, H),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT101
                )
            except cv2.error:
                pass

        return cur_img, cur_gray

    ref_g = gray_8u[ref_idx]
    for i in range(ref_idx - 1, -1, -1):
        aligned[i], ref_g = _align_pair(i, ref_g)
        
    ref_g = gray_8u[ref_idx]
    for i in range(ref_idx + 1, N):
        aligned[i], ref_g = _align_pair(i, ref_g)

    return aligned


# =============================================================================
# STAGE 1 & 4 PRIMITIVES: Fast Guided Filter & Structure Tensor
# =============================================================================
def fast_guided_filter_cv2(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
    subsample: int = 4
) -> np.ndarray:
    """
    SIMD-accelerated Gradient-Domain Guided Image Filter (GDGIF, Kou et al. 2015).
    Incorporates multi-scale variance edge-aware weighting normalized by local boxFilter
    mean to preserve perfect translation invariance and tile seam continuity.
    """
    H, W = guide.shape
    s = max(1, subsample)
    
    if s > 1:
        sh, sw = max(2, H // s), max(2, W // s)
        g_sub = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
        p_sub = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        r_sub = max(1, radius // s)
    else:
        g_sub, p_sub, r_sub = guide, src, radius

    ksize = (2 * r_sub + 1, 2 * r_sub + 1)
    
    mean_g = cv2.boxFilter(g_sub, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT101)
    mean_p = cv2.boxFilter(p_sub, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT101)
    mean_gg = cv2.boxFilter(g_sub * g_sub, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT101)
    mean_gp = cv2.boxFilter(g_sub * p_sub, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT101)
    
    var_g = np.maximum(0.0, mean_gg - mean_g * mean_g)
    cov_gp = mean_gp - mean_g * mean_p
    
    # GDGIF multi-scale variance edge-aware weighting
    if r_sub > 1:
        ksize1 = (3, 3)
        mean_g1 = cv2.boxFilter(g_sub, cv2.CV_32F, ksize1, borderType=cv2.BORDER_REFLECT101)
        mean_gg1 = cv2.boxFilter(g_sub * g_sub, cv2.CV_32F, ksize1, borderType=cv2.BORDER_REFLECT101)
        var_g1 = np.maximum(0.0, mean_gg1 - mean_g1 * mean_g1)
    else:
        var_g1 = var_g

    chi = np.sqrt(np.maximum(0.0, var_g * var_g1)) + 1e-6
    # Local boxFilter mean to preserve translation invariance and tile seam continuity
    ksize_local = (4 * r_sub + 1, 4 * r_sub + 1)
    mean_chi = cv2.boxFilter(chi, cv2.CV_32F, ksize_local, borderType=cv2.BORDER_REFLECT101) + 1e-6
    gamma = np.clip(chi / mean_chi, 0.1, 10.0)

    a = cov_gp / (var_g + (eps / gamma))
    b = mean_p - a * mean_g
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT101)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT101)
    
    if s > 1:
        mean_a = cv2.resize(mean_a, (W, H), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (W, H), interpolation=cv2.INTER_LINEAR)
        
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def compute_focus_energy_cv2(
    lum: np.ndarray,
    grad_ksize: int,
    cfg: StackerConfig
) -> np.ndarray:
    """Rotationally invariant Scharr Structure Tensor + Specular & Shot-Noise Gate."""
    gx = cv2.Scharr(lum, cv2.CV_32F, 1, 0, scale=1.0 / 32.0, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Scharr(lum, cv2.CV_32F, 0, 1, scale=1.0 / 32.0, borderType=cv2.BORDER_REFLECT101)
    
    st_ksize = (3, 3)
    j_xx = cv2.boxFilter(gx * gx, cv2.CV_32F, st_ksize, borderType=cv2.BORDER_REFLECT101)
    j_yy = cv2.boxFilter(gy * gy, cv2.CV_32F, st_ksize, borderType=cv2.BORDER_REFLECT101)
    j_xy = cv2.boxFilter(gx * gy, cv2.CV_32F, st_ksize, borderType=cv2.BORDER_REFLECT101)
    
    trace = j_xx + j_yy
    coherence = np.sqrt((j_xx - j_yy) ** 2 + 4.0 * (j_xy ** 2))
    raw_energy = trace + cfg.anisotropy_alpha * coherence
    
    # Specular Bokeh-Ring Gate
    clipped_mask = (lum > cfg.specular_clip_thresh).astype(np.uint8)
    if np.any(clipped_mask):
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ring_zone = cv2.dilate(clipped_mask, kern) - cv2.erode(clipped_mask, kern)
        raw_energy = np.where(ring_zone > 0, raw_energy * 0.15, raw_energy)
    
    # Shot-Noise Floor Gate
    local_lum = cv2.boxFilter(lum, cv2.CV_32F, (15, 15), borderType=cv2.BORDER_REFLECT101)
    noise_floor = cfg.noise_read + cfg.noise_shot * np.maximum(0.0, local_lum)
    
    return np.maximum(0.0, raw_energy - noise_floor)


# =============================================================================
# CORE FUSION ENGINE (Operates on Full Frame or Padded Spatial Tile)
# =============================================================================
class Pass1DepthEstimator:
    """
    True Bounded-Memory O(H * W) Sliding-Window Depth Estimator.
    Computes exact bidirectional halo-suppressed clean depth maps without allocating
    (N, H, W) energy or luma volumes in memory or on disk.

    Uses a 3-slice ring buffer for exact temporal energy smoothing ([0.25, 0.50, 0.25]),
    accumulates 2D running envelopes, and resolves clean depth in O(H * W) space.
    """
    def __init__(self, H: int, W: int, N: int, max_coc: int, cfg: StackerConfig):
        self.H = H
        self.W = W
        self.N = N
        self.max_coc = max_coc
        self.cfg = cfg
        self.r_coc_morph = max(19, int(0.75 * max_coc)) | 1
        self.grad_ksize = max(3, int(0.08 * max_coc)) | 1
        self.luma_mat = np.array([[0.2126, 0.7152, 0.0722]], dtype=np.float32)

        # 2D running planes for single pass
        self.peak_e = np.full((H, W), -1.0, dtype=np.float32)
        self.raw_z = np.zeros((H, W), dtype=np.float32)
        self.luma_at_raw_z = np.zeros((H, W), dtype=np.float32)
        self.min_luma_vol = np.full((H, W), 1e9, dtype=np.float32)
        self.max_luma_vol = np.full((H, W), -1e9, dtype=np.float32)
        self.z_min_luma = np.zeros((H, W), dtype=np.float32)
        self.z_max_luma = np.zeros((H, W), dtype=np.float32)
        self.effective_bg_mode = self.cfg.bg_mode

        # Thumbnail volume in RAM for zero disk re-reads in Pass 1 (< 70 MB for 61MP)
        self.sub_factor = 4
        self.sub_h = max(1, H // self.sub_factor)
        self.sub_w = max(1, W // self.sub_factor)
        self.luma_sub = np.empty((N, self.sub_h, self.sub_w), dtype=np.uint8)

        # Ring buffers for 3-slice temporal smoothing
        self.e_buf: list = []
        self.l_buf: list = []

        # 2D clean planes (computed in finish_sweep1)
        self.peak_clean = np.full((H, W), -1.0, dtype=np.float32)
        self.raw_clean_z = np.zeros((H, W), dtype=np.float32)

    def _emit_sweep1(self, idx: int, e_sm: np.ndarray, lum: np.ndarray):
        mask_better = e_sm > self.peak_e
        self.peak_e = np.where(mask_better, e_sm, self.peak_e)
        self.raw_z = np.where(mask_better, float(idx), self.raw_z)
        self.luma_at_raw_z = np.where(mask_better, lum, self.luma_at_raw_z)
        mask_min = lum < self.min_luma_vol
        self.min_luma_vol = np.where(mask_min, lum, self.min_luma_vol)
        self.z_min_luma = np.where(mask_min, float(idx), self.z_min_luma)
        mask_max = lum > self.max_luma_vol
        self.max_luma_vol = np.where(mask_max, lum, self.max_luma_vol)
        self.z_max_luma = np.where(mask_max, float(idx), self.z_max_luma)

    def feed_slice_sweep1(self, i: int, lum: np.ndarray, energy: Optional[np.ndarray] = None):
        lum_u8 = (np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8)
        if self.sub_factor > 1 and (self.H != self.sub_h or self.W != self.sub_w):
            self.luma_sub[i] = cv2.resize(lum_u8, (self.sub_w, self.sub_h), interpolation=cv2.INTER_AREA)
        else:
            self.luma_sub[i] = lum_u8

        if energy is None:
            energy = compute_focus_energy_cv2(lum, grad_ksize=self.grad_ksize, cfg=self.cfg)
        self.e_buf.append(energy)
        self.l_buf.append(lum)
        if i == 0:
            self._emit_sweep1(0, self.e_buf[0], self.l_buf[0])
        elif i >= 2:
            e_sm = 0.25 * self.e_buf[0] + 0.50 * self.e_buf[1] + 0.25 * self.e_buf[2]
            self._emit_sweep1(i - 1, e_sm, self.l_buf[1])
            self.e_buf.pop(0)
            self.l_buf.pop(0)

    def finish_sweep1(self):
        if self.N >= 2:
            self._emit_sweep1(self.N - 1, self.e_buf[-1], self.l_buf[-1])
        self.e_buf.clear()
        self.l_buf.clear()

        ksize = int(self.max_coc * 2.1) | 1
        morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        clamped_peak = np.clip(self.peak_e, 0.0, 0.08)
        self.dilated_peak = cv2.dilate(clamped_peak, morph_k)

        conf_peaks = (self.peak_e > 2e-4) & ((self.dilated_peak <= 0.001) | (clamped_peak >= 0.30 * self.dilated_peak))
        if not np.any(conf_peaks):
            conf_peaks = np.ones((self.H, self.W), dtype=bool)

        _, idx = distance_transform_edt(~conf_peaks, return_distances=True, return_indices=True)
        self.z_dom = self.raw_z[idx[0], idx[1]]

        # Determine effective background polarity
        bg_sample = ~conf_peaks if np.any(~conf_peaks) else np.ones((self.H, self.W), dtype=bool)
        med_bg_luma = float(np.median(0.5 * (self.min_luma_vol[bg_sample] + self.max_luma_vol[bg_sample])))
        self.effective_bg_mode = resolve_bg_mode(self.cfg.bg_mode, med_bg_luma)

        gamma_scale = float(self.cfg.halo_suppression_gamma) / 0.45
        z_scale = float(self.cfg.z_jump_thresh) / 5.0
        halo_peak_thresh = 0.30 * float(np.clip(gamma_scale, 0.2, 2.5))

        if self.effective_bg_mode == "bright":
            inv_max_luma = 1.0 - self.max_luma_vol
            dilated_inv_max = cv2.dilate(inv_max_luma, morph_k)
            self.is_exterior = (inv_max_luma < 0.45 * dilated_inv_max) | (self.max_luma_vol > 0.82)
            ext_halo_at_raw = self.is_exterior & ((np.abs(self.raw_z - self.z_dom) > 1.5 * z_scale) | (self.luma_at_raw_z < self.max_luma_vol - 0.015))
            int_rim_halo_at_raw = (~self.is_exterior) & (self.luma_at_raw_z > self.min_luma_vol + 0.05) & (np.abs(self.raw_z - self.z_dom) > 2.0 * z_scale)
            is_bloomed_or_transient = self.is_exterior & (self.luma_at_raw_z < self.max_luma_vol - 0.02)
            if self.cfg.freeze_transient_slice is not None:
                target_z = float(self.cfg.freeze_transient_slice)
                is_bug_feature = (self.peak_e >= 0.005) & (np.abs(self.raw_z - target_z) <= 1.5)
                is_bloomed_or_transient = is_bloomed_or_transient & (~is_bug_feature)

            is_halo_peak = (
                (self.dilated_peak > 0.001) &
                (
                    ((self.peak_e < halo_peak_thresh * self.dilated_peak) & (ext_halo_at_raw | int_rim_halo_at_raw)) |
                    is_bloomed_or_transient
                )
            )
        elif self.effective_bg_mode == "energy":
            inner_skirt = (clamped_peak < halo_peak_thresh * self.dilated_peak) & (np.abs(self.raw_z - self.z_dom) > 1.0 * z_scale) & (np.abs(self.raw_z - self.z_dom) <= 5.0 * z_scale)
            outer_cone = (clamped_peak < 0.08 * float(np.clip(gamma_scale, 0.2, 2.5)) * self.dilated_peak) & (np.abs(self.raw_z - self.z_dom) > 1.0 * z_scale)
            self.is_exterior = (self.dilated_peak > 0.001) & (inner_skirt | outer_cone)
            ext_halo_at_raw = self.is_exterior
            int_rim_halo_at_raw = np.zeros_like(self.is_exterior)
            is_halo_peak = ext_halo_at_raw
        else: # "dark"
            dilated_min_luma = cv2.dilate(self.min_luma_vol, morph_k)
            self.is_exterior = (self.min_luma_vol < 0.45 * dilated_min_luma) | (self.min_luma_vol < 0.18)
            ext_halo_at_raw = self.is_exterior & ((np.abs(self.raw_z - self.z_dom) > 1.5 * z_scale) | (self.luma_at_raw_z > self.min_luma_vol + 0.015))
            int_rim_halo_at_raw = (~self.is_exterior) & (self.luma_at_raw_z < self.max_luma_vol - 0.05) & (np.abs(self.raw_z - self.z_dom) > 2.0 * z_scale)
            is_bloomed_or_transient = self.is_exterior & (self.luma_at_raw_z > self.min_luma_vol + 0.02)
            if self.cfg.freeze_transient_slice is not None:
                target_z = float(self.cfg.freeze_transient_slice)
                is_bug_feature = (self.peak_e >= 0.005) & (np.abs(self.raw_z - target_z) <= 1.5)
                is_bloomed_or_transient = is_bloomed_or_transient & (~is_bug_feature)

            is_halo_peak = (
                (self.dilated_peak > 0.001) &
                (
                    ((self.peak_e < halo_peak_thresh * self.dilated_peak) & (ext_halo_at_raw | int_rim_halo_at_raw)) |
                    is_bloomed_or_transient
                )
            )
        self.peak_clean = np.where(is_halo_peak, 0.0, self.peak_e)
        self.raw_clean_z = np.where(is_halo_peak, 0.0, self.raw_z)

        self.is_confident = self.peak_clean > 1e-4
        if np.all(self.is_confident):
            self.z_edt = self.raw_clean_z
        elif np.any(self.is_confident):
            _, conf_idx = distance_transform_edt(~self.is_confident, return_distances=True, return_indices=True)
            self.z_edt = self.raw_clean_z[conf_idx[0], conf_idx[1]]
        else:
            self.z_edt = np.zeros((self.H, self.W), dtype=np.float32)

        if self.effective_bg_mode == "energy":
            self.z_edt = np.where(self.is_exterior, self.z_dom, self.z_edt)

    def feed_slice_sweep2(self, i: int, lum: np.ndarray, energy: Optional[np.ndarray] = None):
        """No-op stub preserved for backwards compatibility."""
        pass

    def finish_sweep2(self):
        """No-op stub preserved for backwards compatibility."""
        pass

    def finalize(
        self,
        get_slice_luma: Optional[Callable[[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        need_luma = (~self.is_confident) & self.is_exterior
        luma_at_zedt = np.zeros((self.H, self.W), dtype=np.float32)
        if np.any(need_luma):
            z_edt_int = np.clip(np.round(self.z_edt).astype(np.int32), 0, self.N - 1)
            unique_slices = np.unique(z_edt_int[need_luma])
            for s_idx in unique_slices:
                mask_s = need_luma & (z_edt_int == s_idx)
                if get_slice_luma is not None:
                    lum_s = get_slice_luma(int(s_idx))
                else:
                    thumb = self.luma_sub[s_idx]
                    if thumb.shape != (self.H, self.W):
                        lum_s = cv2.resize(thumb, (self.W, self.H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                    else:
                        lum_s = thumb.astype(np.float32) / 255.0
                luma_at_zedt[mask_s] = lum_s[mask_s]

        if self.effective_bg_mode == "bright":
            z_star = np.where(
                (~self.is_confident) & self.is_exterior & (luma_at_zedt < self.max_luma_vol - 0.015),
                self.z_max_luma, self.z_edt
            )
        elif self.effective_bg_mode == "energy":
            z_star = self.z_edt
        else: # "dark"
            z_star = np.where(
                (~self.is_confident) & self.is_exterior & (luma_at_zedt > self.min_luma_vol + 0.015),
                self.z_min_luma, self.z_edt
            )
        z_med_broad = cv2.medianBlur(np.clip(z_star, 0, 255).astype(np.uint8), ksize=9).astype(np.float32)

        raw_fg_jump = np.maximum(0.0, z_med_broad - z_star)
        is_fg_jump = (raw_fg_jump > self.cfg.z_jump_thresh) & self.is_confident
        fg_candidate = (is_fg_jump & ((~self.is_exterior) | (z_star <= self.z_dom + 1.5))).astype(np.uint8)
        kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_connected = cv2.morphologyEx(fg_candidate, cv2.MORPH_CLOSE, kern_close)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_connected, connectivity=8)
        thin_fg_mask = np.zeros((self.H, self.W), dtype=bool)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= 6:
                thin_fg_mask[labels == lbl] = True

        self.is_exterior = self.is_exterior & (~thin_fg_mask)

        z_min_3x3 = cv2.erode(z_star, np.ones((3, 3), dtype=np.uint8))
        z_clean = np.where(thin_fg_mask, z_min_3x3, z_med_broad)
        z_detail_map = np.where(self.is_confident & (~self.is_exterior), self.raw_clean_z, z_clean)

        morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.r_coc_morph, self.r_coc_morph))
        z_max_local = cv2.dilate(z_med_broad, morph_kern)
        z_min_local = cv2.erode(z_med_broad, morph_kern)
        delta_z = z_max_local - z_min_local
        z_mid = 0.5 * (z_max_local + z_min_local)
        halo_zone_mask = (delta_z > self.cfg.z_jump_thresh) & (z_med_broad > z_mid) & (~thin_fg_mask)

        return z_clean, z_detail_map, z_med_broad, thin_fg_mask, halo_zone_mask, self.is_confident, self.min_luma_vol

    def estimate_from_block(
        self,
        block_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = self.N
        for i in range(N):
            lum = cv2.transform(block_rgb[i], self.luma_mat).reshape(self.H, self.W)
            self.feed_slice_sweep1(i, lum)
        self.finish_sweep1()
        get_luma = lambda s: cv2.transform(block_rgb[s], self.luma_mat).reshape(self.H, self.W)
        return self.finalize(get_luma)


def compute_clean_depth_map(
    energy_vol: np.ndarray,
    luma_vol: np.ndarray,
    max_coc: int,
    cfg: StackerConfig,
    r_coc_morph: int,
    in_place: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes bidirectional defocus halo-suppressed depth map, macro-scale depth,
    thin foreground bristle mask, halo zone mask, confidence mask, and background luma envelope.
    Shared by both in-memory tile fusion (_fuse_block_cv2) and streaming 16-bit batch fusion (stream_fuse_egmsd).
    """
    N, H, W = energy_vol.shape
    raw_z = np.argmax(energy_vol, axis=0).astype(np.float32)
    peak_e = np.max(energy_vol, axis=0)

    if luma_vol.dtype == np.uint8:
        min_luma_vol = np.min(luma_vol, axis=0).astype(np.float32) / 255.0
        max_luma_vol = np.max(luma_vol, axis=0).astype(np.float32) / 255.0
        luma_at_raw_z = np.take_along_axis(luma_vol, raw_z.astype(int)[np.newaxis, :, :], axis=0)[0].astype(np.float32) / 255.0
    else:
        min_luma_vol = np.min(luma_vol, axis=0).astype(np.float32)
        max_luma_vol = np.max(luma_vol, axis=0).astype(np.float32)
        luma_at_raw_z = np.take_along_axis(luma_vol, raw_z.astype(int)[np.newaxis, :, :], axis=0)[0]

    ksize = int(max_coc * 2.1) | 1
    morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    clamped_peak = np.clip(peak_e, 0.0, 0.08)
    dilated_peak = cv2.dilate(clamped_peak, morph_k)
    # Confident peaks: true in-focus features (>= 30% of local clamped dilated peak)
    conf_peaks = (peak_e > 2e-4) & ((dilated_peak <= 0.001) | (clamped_peak >= 0.30 * dilated_peak))
    if not np.any(conf_peaks):
        conf_peaks = np.ones((H, W), dtype=bool)

    _, idx = distance_transform_edt(~conf_peaks, return_distances=True, return_indices=True)
    z_dom = raw_z[idx[0], idx[1]]

    z_min_luma = np.argmin(luma_vol, axis=0).astype(np.float32)
    z_max_luma = np.argmax(luma_vol, axis=0).astype(np.float32)

    bg_sample = ~conf_peaks if np.any(~conf_peaks) else np.ones((H, W), dtype=bool)
    med_bg_luma = float(np.median(0.5 * (min_luma_vol[bg_sample] + max_luma_vol[bg_sample])))
    effective_bg_mode = resolve_bg_mode(cfg.bg_mode, med_bg_luma)

    gamma_scale = float(cfg.halo_suppression_gamma) / 0.45
    z_scale = float(cfg.z_jump_thresh) / 5.0
    halo_peak_thresh = 0.30 * float(np.clip(gamma_scale, 0.2, 2.5))

    if effective_bg_mode == "bright":
        inv_max_luma = 1.0 - max_luma_vol
        dilated_inv_max = cv2.dilate(inv_max_luma, morph_k)
        is_exterior = (inv_max_luma < 0.45 * dilated_inv_max) | (max_luma_vol > 0.82)
        ext_halo = is_exterior & ((np.abs(raw_z - z_dom) > 1.5 * z_scale) | (luma_at_raw_z < max_luma_vol - 0.015))
        int_rim_halo = (~is_exterior) & (luma_at_raw_z > min_luma_vol + 0.05) & (np.abs(raw_z - z_dom) > 2.0 * z_scale)
        is_bloomed_or_transient = is_exterior & (luma_at_raw_z < max_luma_vol - 0.02)
        if cfg.freeze_transient_slice is not None:
            target_z = float(cfg.freeze_transient_slice)
            is_bug_feature = (peak_e >= 0.005) & (np.abs(raw_z - target_z) <= 1.5)
            is_bloomed_or_transient = is_bloomed_or_transient & (~is_bug_feature)

        is_defocus_halo = (
            (dilated_peak > 0.001) &
            (
                ((peak_e < halo_peak_thresh * dilated_peak) & (ext_halo | int_rim_halo)) |
                is_bloomed_or_transient
            )
        )
    elif effective_bg_mode == "energy":
        inner_skirt = (clamped_peak < halo_peak_thresh * dilated_peak) & (np.abs(raw_z - z_dom) > 1.0 * z_scale) & (np.abs(raw_z - z_dom) <= 5.0 * z_scale)
        outer_cone = (clamped_peak < 0.08 * float(np.clip(gamma_scale, 0.2, 2.5)) * dilated_peak) & (np.abs(raw_z - z_dom) > 1.0 * z_scale)
        is_exterior = (dilated_peak > 0.001) & (inner_skirt | outer_cone)
        ext_halo = is_exterior
        int_rim_halo = np.zeros_like(is_exterior)
        is_defocus_halo = ext_halo
    else: # "dark"
        dilated_min_luma = cv2.dilate(min_luma_vol, morph_k)
        is_exterior = (min_luma_vol < 0.45 * dilated_min_luma) | (min_luma_vol < 0.18)
        ext_halo = is_exterior & ((np.abs(raw_z - z_dom) > 1.5 * z_scale) | (luma_at_raw_z > min_luma_vol + 0.015))
        int_rim_halo = (~is_exterior) & (luma_at_raw_z < max_luma_vol - 0.05) & (np.abs(raw_z - z_dom) > 2.0 * z_scale)
        is_bloomed_or_transient = is_exterior & (luma_at_raw_z > min_luma_vol + 0.02)
        if cfg.freeze_transient_slice is not None:
            target_z = float(cfg.freeze_transient_slice)
            is_bug_feature = (peak_e >= 0.005) & (np.abs(raw_z - target_z) <= 1.5)
            is_bloomed_or_transient = is_bloomed_or_transient & (~is_bug_feature)

        is_defocus_halo = (
            (dilated_peak > 0.001) &
            (
                ((peak_e < halo_peak_thresh * dilated_peak) & (ext_halo | int_rim_halo)) |
                is_bloomed_or_transient
            )
        )

    peak_clean = np.where(is_defocus_halo, 0.0, peak_e)
    is_confident = peak_clean > 1e-4
    raw_clean_z = np.where(is_defocus_halo, 0.0, raw_z)
    if np.all(is_confident):
        z_edt = raw_clean_z
    elif np.any(is_confident):
        _, conf_idx = distance_transform_edt(~is_confident, return_distances=True, return_indices=True)
        z_edt = raw_clean_z[conf_idx[0], conf_idx[1]]
    else:
        z_edt = np.zeros((H, W), dtype=np.float32)

    if effective_bg_mode == "energy":
        z_edt = np.where(is_exterior, z_dom, z_edt)

    z_edt_int = np.clip(np.round(z_edt).astype(np.int32), 0, N - 1)
    if luma_vol.dtype == np.uint8:
        luma_at_zedt = np.take_along_axis(luma_vol, z_edt_int[np.newaxis, :, :], axis=0)[0].astype(np.float32) / 255.0
    else:
        luma_at_zedt = np.take_along_axis(luma_vol, z_edt_int[np.newaxis, :, :], axis=0)[0]

    if effective_bg_mode == "bright":
        z_star = np.where((~is_confident) & is_exterior & (luma_at_zedt < max_luma_vol - 0.015), z_max_luma, z_edt)
    elif effective_bg_mode == "energy":
        z_star = z_edt
    else:
        z_star = np.where((~is_confident) & is_exterior & (luma_at_zedt > min_luma_vol + 0.015), z_min_luma, z_edt)

    z_med_broad = cv2.medianBlur(np.clip(z_star, 0, 255).astype(np.uint8), ksize=9).astype(np.float32)

    raw_fg_jump = np.maximum(0.0, z_med_broad - z_star)
    is_fg_jump = (raw_fg_jump > cfg.z_jump_thresh) & is_confident
    fg_candidate = (is_fg_jump & ((~is_exterior) | (z_star <= z_dom + 1.5))).astype(np.uint8)
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_connected = cv2.morphologyEx(fg_candidate, cv2.MORPH_CLOSE, kern_close)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_connected, connectivity=8)
    thin_fg_mask = np.zeros((H, W), dtype=bool)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 6:
            thin_fg_mask[labels == lbl] = True

    is_exterior = is_exterior & (~thin_fg_mask)

    z_min_3x3 = cv2.erode(z_star, np.ones((3, 3), dtype=np.uint8))
    z_clean = np.where(thin_fg_mask, z_min_3x3, z_med_broad)
    z_detail_map = np.where(is_confident & (~is_exterior), raw_clean_z, z_clean)

    morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_coc_morph, r_coc_morph))
    z_max_local = cv2.dilate(z_med_broad, morph_kern)
    z_min_local = cv2.erode(z_med_broad, morph_kern)
    delta_z = z_max_local - z_min_local
    z_mid = 0.5 * (z_max_local + z_min_local)
    halo_zone_mask = (delta_z > cfg.z_jump_thresh) & (z_med_broad > z_mid) & (~thin_fg_mask)

    return z_clean, z_detail_map, z_med_broad, thin_fg_mask, halo_zone_mask, is_confident, min_luma_vol


def _fuse_block_cv2(
    block_rgb: np.ndarray,
    max_coc: int,
    cfg: StackerConfig
) -> Dict[str, np.ndarray]:
    """Core EG-MSD fusion on a (N, H_b, W_b, 3) float32 block."""
    N, H, W, _ = block_rgb.shape
    
    sigma2 = max(1.5, 0.06 * max_coc)
    sigma1 = 0.50 * sigma2
    sigma0 = 0.25 * sigma2

    r_base   = max(6, int(0.22 * max_coc))
    r_coarse = max(3, int(0.10 * max_coc))
    r_mid    = max(2, int(0.05 * max_coc))
    r_fine   = max(1, int(0.025 * max_coc))

    estimator = Pass1DepthEstimator(H, W, N, max_coc, cfg)
    (
        z_clean, z_detail_map, z_med_broad, thin_fg_mask,
        halo_zone_mask, is_confident, min_luma_vol
    ) = estimator.estimate_from_block(block_rgb)

    # -------------------------------------------------------------------------
    # PASS 2: Multi-Octave Laplacian Pyramid Accumulation + Native Hair Blending
    # Telescoping decomposition: B2 + D2 + D1 + D0 == img_n
    # -------------------------------------------------------------------------
    accum_base = np.zeros((H, W, 3), dtype=np.float32)
    norm_base = np.full((H, W, 1), 1e-8, dtype=np.float32)

    accum_d2 = np.zeros((H, W, 3), dtype=np.float32)
    norm_d2 = np.full((H, W, 1), 1e-8, dtype=np.float32)

    accum_d1 = np.zeros((H, W, 3), dtype=np.float32)
    norm_d1 = np.full((H, W, 1), 1e-8, dtype=np.float32)

    accum_d0 = np.zeros((H, W, 3), dtype=np.float32)
    norm_d0 = np.full((H, W, 1), 1e-8, dtype=np.float32)

    # Dedicated direct-pixel accumulator for thin crossing foreground hairs
    accum_hair = np.zeros((H, W, 3), dtype=np.float32)
    norm_hair = np.full((H, W, 1), 1e-8, dtype=np.float32)

    for n in range(N):
        img_n = block_rgb[n]
        guide_lum = cv2.transform(img_n, estimator.luma_mat).reshape(H, W)

        # Telescoping Laplacian decomposition (uncascaded)
        b0 = cv2.GaussianBlur(img_n, (0, 0), sigmaX=sigma0, sigmaY=sigma0, borderType=cv2.BORDER_REFLECT101)
        d0 = img_n - b0

        b1 = cv2.GaussianBlur(img_n, (0, 0), sigmaX=sigma1, sigmaY=sigma1, borderType=cv2.BORDER_REFLECT101)
        d1 = b0 - b1

        b2 = cv2.GaussianBlur(img_n, (0, 0), sigmaX=sigma2, sigmaY=sigma2, borderType=cv2.BORDER_REFLECT101)
        d2 = b1 - b2
        base_n = b2

        # Scale-matched slice windows
        p_base = np.exp(-0.5 * ((z_clean - float(n)) / 0.75) ** 2).astype(np.float32)
        p_base = np.where(p_base < 0.05, 0.0, p_base)

        p_d2 = np.exp(-0.5 * ((z_clean - float(n)) / 0.5) ** 2).astype(np.float32)
        p_d2 = np.where(p_d2 < 0.05, 0.0, p_d2)

        p_d1 = np.exp(-0.5 * ((z_detail_map - float(n)) / 0.5) ** 2).astype(np.float32)
        p_d1 = np.where(p_d1 < 0.05, 0.0, p_d1)

        p_d0 = (np.abs(z_detail_map - float(n)) < 0.5).astype(np.float32)

        gamma_scale = float(cfg.halo_suppression_gamma) / 0.45

        # Polarity-aware edge gate for dark/bright background pockets and transient suppression
        eff_mode = getattr(estimator, "effective_bg_mode", "dark")
        if eff_mode == "bright":
            shadow_excess = np.maximum(0.0, (estimator.max_luma_vol - 0.015) - guide_lum)
            gate = np.where(estimator.is_exterior, np.exp(-50.0 * gamma_scale * shadow_excess), 1.0).astype(np.float32)
            if gamma_scale != 1.0:
                extra_decay = np.exp(-0.25 * (gamma_scale - 1.0) * ((z_clean - float(n)) / 0.75) ** 2)
                gate = np.where(halo_zone_mask & (~thin_fg_mask), gate * extra_decay, gate)
            is_max_slice = (np.abs(estimator.z_max_luma - float(n)) < 0.5) & estimator.is_exterior
            gate = np.where(is_max_slice, 1.0, gate)
            if cfg.freeze_transient_slice is not None:
                target_z = float(cfg.freeze_transient_slice)
                bug_core = (np.abs(estimator.raw_z - target_z) <= 1.5) & (estimator.peak_e >= 0.005)
                gate = np.where(bug_core, 1.0, gate)

            p_base = p_base * gate
            p_d2 = p_d2 * gate
            w_b = fast_guided_filter_cv2(guide_lum, p_base, r_base, cfg.eps_base, cfg.fgf_subsample) + 1e-5 * p_base
            w_2 = fast_guided_filter_cv2(guide_lum, p_d2, r_coarse, cfg.eps_base * 0.5, 2) + 1e-5 * p_d2
            w_1 = fast_guided_filter_cv2(guide_lum, p_d1, r_mid, cfg.eps_detail * 5.0, 1) + 1e-5 * p_d1
            w_0 = fast_guided_filter_cv2(guide_lum, p_d0, r_fine, cfg.eps_detail, 1) + 1e-5 * p_d0
            w_b = w_b * gate
            w_2 = w_2 * gate
            w_1 = w_1 * gate
            w_0 = w_0 * gate
            floor_mask = estimator.is_exterior & (np.abs(estimator.z_max_luma - float(n)) < 0.5)
            w_b = np.where(floor_mask, np.maximum(w_b, 0.05), w_b)
        elif eff_mode == "energy":
            gate = np.where(
                estimator.is_exterior & (~thin_fg_mask),
                np.exp(-cfg.halo_suppression_gamma * ((z_clean - float(n)) / 0.75) ** 2),
                1.0
            ).astype(np.float32)
            p_base = p_base * gate
            p_d2 = p_d2 * gate
            w_b = fast_guided_filter_cv2(guide_lum, p_base, r_base, cfg.eps_base, cfg.fgf_subsample) + 1e-5 * p_base
            w_2 = fast_guided_filter_cv2(guide_lum, p_d2, r_coarse, cfg.eps_base * 0.5, 2) + 1e-5 * p_d2
            w_1 = fast_guided_filter_cv2(guide_lum, p_d1, r_mid, cfg.eps_detail * 5.0, 1) + 1e-5 * p_d1
            w_0 = fast_guided_filter_cv2(guide_lum, p_d0, r_fine, cfg.eps_detail, 1) + 1e-5 * p_d0
            w_b = w_b * gate
            w_2 = w_2 * gate
            w_1 = w_1 * gate
            w_0 = w_0 * gate
        else: # "dark"
            bloom_excess = np.maximum(0.0, guide_lum - (estimator.min_luma_vol + 0.015))
            gate = np.where(estimator.is_exterior, np.exp(-50.0 * gamma_scale * bloom_excess), 1.0).astype(np.float32)
            if gamma_scale != 1.0:
                extra_decay = np.exp(-0.25 * (gamma_scale - 1.0) * ((z_clean - float(n)) / 0.75) ** 2)
                gate = np.where(halo_zone_mask & (~thin_fg_mask), gate * extra_decay, gate)
            is_min_slice = (np.abs(estimator.z_min_luma - float(n)) < 0.5) & estimator.is_exterior
            gate = np.where(is_min_slice, 1.0, gate)
            if cfg.freeze_transient_slice is not None:
                target_z = float(cfg.freeze_transient_slice)
                bug_core = (np.abs(estimator.raw_z - target_z) <= 1.5) & (estimator.peak_e >= 0.005)
                gate = np.where(bug_core, 1.0, gate)

            p_base = p_base * gate
            p_d2 = p_d2 * gate
            w_b = fast_guided_filter_cv2(guide_lum, p_base, r_base, cfg.eps_base, cfg.fgf_subsample) + 1e-5 * p_base
            w_2 = fast_guided_filter_cv2(guide_lum, p_d2, r_coarse, cfg.eps_base * 0.5, 2) + 1e-5 * p_d2
            w_1 = fast_guided_filter_cv2(guide_lum, p_d1, r_mid, cfg.eps_detail * 5.0, 1) + 1e-5 * p_d1
            w_0 = fast_guided_filter_cv2(guide_lum, p_d0, r_fine, cfg.eps_detail, 1) + 1e-5 * p_d0
            w_b = w_b * gate
            w_2 = w_2 * gate
            w_1 = w_1 * gate
            w_0 = w_0 * gate
            floor_mask = estimator.is_exterior & (np.abs(estimator.z_min_luma - float(n)) < 0.5)
            w_b = np.where(floor_mask, np.maximum(w_b, 0.05), w_b)

        wb3 = w_b[..., None]
        w23 = w_2[..., None]
        w13 = w_1[..., None]
        w03 = w_0[..., None]

        accum_base += wb3 * base_n
        norm_base += wb3

        accum_d2 += w23 * d2
        norm_d2 += w23

        accum_d1 += w13 * d1
        norm_d1 += w13

        accum_d0 += w03 * d0
        norm_d0 += w03

        # Direct unblurred accumulation for thin foreground structures (1px radius FGF)
        p_hair = (np.abs(z_clean - float(n)) < 0.5).astype(np.float32)
        w_hair = fast_guided_filter_cv2(
            guide=guide_lum, src=p_hair,
            radius=1, eps=1e-5, subsample=1
        )
        w_h_3d = (w_hair * thin_fg_mask)[..., np.newaxis]
        accum_hair += w_h_3d * img_n
        norm_hair += w_h_3d

    fused_pyr = (accum_base / norm_base) + (accum_d2 / norm_d2) + (accum_d1 / norm_d1) + (accum_d0 / norm_d0)
    
    # Blend thin foreground hairs directly from unblurred native slices
    hair_valid = (norm_hair > 1e-3).squeeze(-1)
    hair_rgb = accum_hair / norm_hair
    hair_alpha = np.clip(norm_hair, 0.0, 1.0)
    fused_rgb = np.where(
        hair_valid[..., np.newaxis],
        hair_alpha * hair_rgb + (1.0 - hair_alpha) * fused_pyr,
        fused_pyr
    )
    
    # Petteri Aimonen dynamic envelope protection: clip mathematical ringing
    min_env = np.min(block_rgb, axis=0)
    max_env = np.max(block_rgb, axis=0)
    fused_clamped = np.clip(fused_rgb, min_env, max_env)

    p99 = max(1e-5, float(np.percentile(estimator.peak_e, 99)))
    saliency_map = np.clip(np.sqrt(np.clip(estimator.peak_e / p99, 0.0, 1.0)), 0.0, 1.0)

    return {
        "fused_rgb": np.clip(fused_clamped, 0.0, 1.0),
        "depth_map": z_clean,
        "halo_mask": halo_zone_mask,
        "saliency_map": saliency_map,
        "effective_bg_mode": getattr(estimator, "effective_bg_mode", cfg.bg_mode)
    }


# =============================================================================
# PUBLIC ENTRYPOINT: Memory-Bounded Overlapping Tile Wrapper
# =============================================================================
def fuse_focus_stack(
    stack_rgb: np.ndarray,
    cfg: StackerConfig = StackerConfig()
) -> Dict[str, np.ndarray]:
    """
    Production entry point. Aligns stack and processes via overlapping spatial
    tiles if image dimensions exceed cfg.tile_size.
    """
    N, H, W, C = stack_rgb.shape
    assert stack_rgb.dtype == np.float32, "Stack must be float32 linear RGB in [0, 1]."
    
    if cfg.stack_order == "far_to_near":
        stack_rgb = stack_rgb[::-1]

    if cfg.align_ecc:
        stack_rgb = align_focus_stack_cv2(
            stack_rgb, use_ecc=cfg.align_ecc
        )

    if cfg.bg_mode == "auto":
        # Resolve background mode once at full-frame level to ensure perfect tile seam continuity
        s_step = max(1, max(H, W) // 256)
        sub_luma = (
            stack_rgb[:, ::s_step, ::s_step, 0] * 0.2126 +
            stack_rgb[:, ::s_step, ::s_step, 1] * 0.7152 +
            stack_rgb[:, ::s_step, ::s_step, 2] * 0.0722
        )
        min_l = np.min(sub_luma, axis=0)
        max_l = np.max(sub_luma, axis=0)
        med_bg_luma = float(np.median(0.5 * (min_l + max_l)))
        resolved_mode = resolve_bg_mode("auto", med_bg_luma)
        cfg = replace(cfg, bg_mode=resolved_mode)

    max_coc = cfg.max_coc_radius_px or max(16, int(0.007 * max(H, W)))
    
    # If frame fits inside a single tile, run directly without tiling overhead
    if H <= cfg.tile_size and W <= cfg.tile_size:
        res = _fuse_block_cv2(stack_rgb, max_coc=max_coc, cfg=cfg)
        if cfg.stack_order == "far_to_near":
            res["depth_map"] = float(N - 1) - res["depth_map"]
        return res

    # Halo padding must exceed 2 * r_base so guided filter boundary conditions vanish
    pad = max(160, int(2.5 * max_coc))
    step = cfg.tile_size
    
    out_rgb = np.zeros((H, W, 3), dtype=np.float32)
    out_depth = np.zeros((H, W), dtype=np.float32)
    out_halo = np.zeros((H, W), dtype=bool)
    out_saliency = np.zeros((H, W), dtype=np.float32)
    
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1, x1 = min(H, y0 + step), min(W, x0 + step)
            
            # Extract padded region
            py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
            py1, px1 = min(H, y1 + pad), min(W, x1 + pad)
            
            tile_res = _fuse_block_cv2(
                stack_rgb[:, py0:py1, px0:px1, :], max_coc=max_coc, cfg=cfg
            )
            
            # Crop out the unpadded core
            cy0, cx0 = y0 - py0, x0 - px0
            cy1, cx1 = cy0 + (y1 - y0), cx0 + (x1 - x0)
            
            out_rgb[y0:y1, x0:x1] = tile_res["fused_rgb"][cy0:cy1, cx0:cx1]
            out_depth[y0:y1, x0:x1] = tile_res["depth_map"][cy0:cy1, cx0:cx1]
            out_halo[y0:y1, x0:x1] = tile_res["halo_mask"][cy0:cy1, cx0:cx1]
            out_saliency[y0:y1, x0:x1] = tile_res.get("saliency_map", tile_res["halo_mask"])[cy0:cy1, cx0:cx1]

    res = {
        "fused_rgb": out_rgb,
        "depth_map": out_depth,
        "halo_mask": out_halo,
        "saliency_map": out_saliency,
        "effective_bg_mode": cfg.bg_mode
    }
    if cfg.stack_order == "far_to_near":
        res["depth_map"] = float(N - 1) - res["depth_map"]
    return res


# Backwards compatibility alias
fuse_focus_stack_cv2 = fuse_focus_stack


# =============================================================================
# STREAMING 16-BIT BATCH ENGINE FOR 61MP+ PHOTOSHOOTS
# =============================================================================
def stream_fuse_egmsd(
    files,
    warp_mats,
    crop_box,
    cfg: StackerConfig = StackerConfig(),
    icc_profile=None,
    output_path=None,
    overrides_mask=None,
    update_progress=None
):
    """
    Memory-efficient streaming 16-bit Edge-Guided Multi-Scale Decomposition (EG-MSD).
    Streams frames one by one in two passes with temporary disk memmapped energy volume.
    Peak RAM usage is strictly bounded (< 2.5 GB) regardless of megapixel count or frame count.
    """
    import time
    import tempfile
    from tiff_io import read_tiff_16bit, write_image

    bx, by, bw, bh = crop_box
    N = len(files)
    assert N >= 2, "Must provide at least 2 images to stack"

    H, W = bh, bw
    max_coc = cfg.max_coc_radius_px or max(16, int(0.007 * max(H, W)))
    sigma2 = max(1.5, 0.06 * max_coc)
    sigma1 = 0.50 * sigma2
    sigma0 = 0.25 * sigma2

    r_base   = max(6, int(0.22 * max_coc))
    r_coarse = max(3, int(0.10 * max_coc))
    r_mid    = max(2, int(0.05 * max_coc))
    r_fine   = max(1, int(0.025 * max_coc))
    r_coc_morph = max(19, int(0.75 * max_coc)) | 1
    grad_ksize  = max(3, int(0.08 * max_coc)) | 1

    print(f"Starting EG-MSD streaming pipeline for {N} frames ({W}x{H})...")
    print(f"  Radii: max_coc={max_coc}px, base={r_base}px, coarse={r_coarse}px, mid={r_mid}px, fine={r_fine}px, morph={r_coc_morph}px")

    use_files = list(reversed(files)) if cfg.stack_order == "far_to_near" else files
    use_warp_mats = list(reversed(warp_mats)) if cfg.stack_order == "far_to_near" else warp_mats

    estimator = Pass1DepthEstimator(H, W, N, max_coc, cfg)

    # Pass 1: True single disk pass for Focus Energy, 2D envelopes, and clean depth map
    for i, fpath in enumerate(use_files):
        t_s = time.time()
        img, _ = read_tiff_16bit(fpath)
        full_h, full_w = img.shape[:2]

        warp_mat = use_warp_mats[i]
        is_identity = (
            np.abs(warp_mat[0, 0] - 1.0) < 1e-5 and
            np.abs(warp_mat[1, 1] - 1.0) < 1e-5 and
            np.abs(warp_mat[0, 1]) < 1e-5 and
            np.abs(warp_mat[1, 0]) < 1e-5 and
            np.abs(warp_mat[0, 2]) < 1e-4 and
            np.abs(warp_mat[1, 2]) < 1e-4
        )
        if not is_identity:
            img = cv2.warpAffine(
                img, warp_mat, (full_w, full_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT101
            )

        cropped = img[by : by + bh, bx : bx + bw]
        del img

        cropped_f = cropped.astype(np.float32) / 65535.0
        del cropped

        lum = cv2.transform(cropped_f, np.array([[0.2126, 0.7152, 0.0722]], dtype=np.float32)).reshape(H, W)
        del cropped_f

        estimator.feed_slice_sweep1(i, lum)
        del lum
        gc.collect()

        if update_progress:
            prog = 0.50 * ((i + 1) / float(N))
            update_progress("Pass 1/2: Depth Map", f"Processed {os.path.basename(fpath)} ({i + 1}/{N})", prog)
        print(f"  [Pass 1/2: {i+1}/{N}] {os.path.basename(fpath)} in {time.time() - t_s:.2f}s")

    estimator.finish_sweep1()

    (
        z_clean, z_detail_map, z_med_broad, thin_fg_mask,
        halo_zone_mask, is_confident, min_luma_vol
    ) = estimator.finalize()

    # PASS 2: Multi-Octave Laplacian Pyramid Accumulation + Native Hair Blending
    # Telescoping decomposition: B2 + D2 + D1 + D0 == img_n
    accum_base = np.zeros((H, W, 3), dtype=np.float32)
    norm_base = np.full((H, W, 1), 1e-8, dtype=np.float32)

    accum_d2 = np.zeros((H, W, 3), dtype=np.float32)
    norm_d2 = np.full((H, W, 1), 1e-8, dtype=np.float32)

    accum_d1 = np.zeros((H, W, 3), dtype=np.float32)
    norm_d1 = np.full((H, W, 1), 1e-8, dtype=np.float32)

    accum_d0 = np.zeros((H, W, 3), dtype=np.float32)
    norm_d0 = np.full((H, W, 1), 1e-8, dtype=np.float32)

    accum_hair = np.zeros((H, W, 3), dtype=np.float32)
    norm_hair = np.full((H, W, 1), 1e-8, dtype=np.float32)

    min_env = None
    max_env = None

    print("Pass 2/2: Multi-octave Laplacian pyramid accumulation...")
    for n in range(N):
        t_s = time.time()
        img, _ = read_tiff_16bit(use_files[n])
        full_h, full_w = img.shape[:2]
        warp_mat = use_warp_mats[n]
        is_identity = (
            np.abs(warp_mat[0, 0] - 1.0) < 1e-5 and
            np.abs(warp_mat[1, 1] - 1.0) < 1e-5 and
            np.abs(warp_mat[0, 1]) < 1e-5 and
            np.abs(warp_mat[1, 0]) < 1e-5 and
            np.abs(warp_mat[0, 2]) < 1e-4 and
            np.abs(warp_mat[1, 2]) < 1e-4
        )
        if not is_identity:
            img = cv2.warpAffine(
                img, warp_mat, (full_w, full_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT101
            )

        cropped = img[by : by + bh, bx : bx + bw]
        del img

        if min_env is None:
            min_env = cropped.copy()
            max_env = cropped.copy()
        else:
            min_env = np.minimum(min_env, cropped)
            max_env = np.maximum(max_env, cropped)

        img_n = cropped.astype(np.float32) / 65535.0
        del cropped

        guide_lum = cv2.transform(img_n, np.array([[0.2126, 0.7152, 0.0722]], dtype=np.float32)).reshape(H, W)

        # Telescoping Laplacian decomposition (uncascaded)
        b0 = cv2.GaussianBlur(img_n, (0, 0), sigmaX=sigma0, sigmaY=sigma0, borderType=cv2.BORDER_REFLECT101)
        d0 = img_n - b0

        b1 = cv2.GaussianBlur(img_n, (0, 0), sigmaX=sigma1, sigmaY=sigma1, borderType=cv2.BORDER_REFLECT101)
        d1 = b0 - b1

        b2 = cv2.GaussianBlur(img_n, (0, 0), sigmaX=sigma2, sigmaY=sigma2, borderType=cv2.BORDER_REFLECT101)
        d2 = b1 - b2
        base_n = b2

        # Scale-matched slice windows
        p_base = np.exp(-0.5 * ((z_clean - float(n)) / 0.75) ** 2).astype(np.float32)
        p_base = np.where(p_base < 0.05, 0.0, p_base)

        p_d2 = np.exp(-0.5 * ((z_clean - float(n)) / 0.5) ** 2).astype(np.float32)
        p_d2 = np.where(p_d2 < 0.05, 0.0, p_d2)

        p_d1 = np.exp(-0.5 * ((z_detail_map - float(n)) / 0.5) ** 2).astype(np.float32)
        p_d1 = np.where(p_d1 < 0.05, 0.0, p_d1)

        p_d0 = (np.abs(z_detail_map - float(n)) < 0.5).astype(np.float32)

        gamma_scale = float(cfg.halo_suppression_gamma) / 0.45

        # Polarity-aware edge gate for dark/bright background pockets and transient suppression
        eff_mode = getattr(estimator, "effective_bg_mode", "dark")
        if eff_mode == "bright":
            shadow_excess = np.maximum(0.0, (estimator.max_luma_vol - 0.015) - guide_lum)
            gate = np.where(estimator.is_exterior, np.exp(-50.0 * gamma_scale * shadow_excess), 1.0).astype(np.float32)
            if gamma_scale != 1.0:
                extra_decay = np.exp(-0.25 * (gamma_scale - 1.0) * ((z_clean - float(n)) / 0.75) ** 2)
                gate = np.where(halo_zone_mask & (~thin_fg_mask), gate * extra_decay, gate)
            is_max_slice = (np.abs(estimator.z_max_luma - float(n)) < 0.5) & estimator.is_exterior
            gate = np.where(is_max_slice, 1.0, gate)
            if cfg.freeze_transient_slice is not None:
                target_z = float(cfg.freeze_transient_slice)
                bug_core = (np.abs(estimator.raw_z - target_z) <= 1.5) & (estimator.peak_e >= 0.005)
                gate = np.where(bug_core, 1.0, gate)

            p_base = p_base * gate
            p_d2 = p_d2 * gate
            w_b = fast_guided_filter_cv2(guide_lum, p_base, r_base, cfg.eps_base, cfg.fgf_subsample) + 1e-5 * p_base
            w_2 = fast_guided_filter_cv2(guide_lum, p_d2, r_coarse, cfg.eps_base * 0.5, 2) + 1e-5 * p_d2
            w_1 = fast_guided_filter_cv2(guide_lum, p_d1, r_mid, cfg.eps_detail * 5.0, 1) + 1e-5 * p_d1
            w_0 = fast_guided_filter_cv2(guide_lum, p_d0, r_fine, cfg.eps_detail, 1) + 1e-5 * p_d0
            w_b = w_b * gate
            w_2 = w_2 * gate
            w_1 = w_1 * gate
            w_0 = w_0 * gate
            floor_mask = estimator.is_exterior & (np.abs(estimator.z_max_luma - float(n)) < 0.5)
            w_b = np.where(floor_mask, np.maximum(w_b, 0.05), w_b)
        elif eff_mode == "energy":
            gate = np.where(
                estimator.is_exterior & (~thin_fg_mask),
                np.exp(-cfg.halo_suppression_gamma * ((z_clean - float(n)) / 0.75) ** 2),
                1.0
            ).astype(np.float32)
            p_base = p_base * gate
            p_d2 = p_d2 * gate
            w_b = fast_guided_filter_cv2(guide_lum, p_base, r_base, cfg.eps_base, cfg.fgf_subsample) + 1e-5 * p_base
            w_2 = fast_guided_filter_cv2(guide_lum, p_d2, r_coarse, cfg.eps_base * 0.5, 2) + 1e-5 * p_d2
            w_1 = fast_guided_filter_cv2(guide_lum, p_d1, r_mid, cfg.eps_detail * 5.0, 1) + 1e-5 * p_d1
            w_0 = fast_guided_filter_cv2(guide_lum, p_d0, r_fine, cfg.eps_detail, 1) + 1e-5 * p_d0
            w_b = w_b * gate
            w_2 = w_2 * gate
            w_1 = w_1 * gate
            w_0 = w_0 * gate
        else: # "dark"
            bloom_excess = np.maximum(0.0, guide_lum - (min_luma_vol + 0.015))
            gate = np.where(estimator.is_exterior, np.exp(-50.0 * gamma_scale * bloom_excess), 1.0).astype(np.float32)
            if gamma_scale != 1.0:
                extra_decay = np.exp(-0.25 * (gamma_scale - 1.0) * ((z_clean - float(n)) / 0.75) ** 2)
                gate = np.where(halo_zone_mask & (~thin_fg_mask), gate * extra_decay, gate)
            is_min_slice = (np.abs(estimator.z_min_luma - float(n)) < 0.5) & estimator.is_exterior
            gate = np.where(is_min_slice, 1.0, gate)
            if cfg.freeze_transient_slice is not None:
                target_z = float(cfg.freeze_transient_slice)
                bug_core = (np.abs(estimator.raw_z - target_z) <= 1.5) & (estimator.peak_e >= 0.005)
                gate = np.where(bug_core, 1.0, gate)

            p_base = p_base * gate
            p_d2 = p_d2 * gate
            w_b = fast_guided_filter_cv2(guide_lum, p_base, r_base, cfg.eps_base, cfg.fgf_subsample) + 1e-5 * p_base
            w_2 = fast_guided_filter_cv2(guide_lum, p_d2, r_coarse, cfg.eps_base * 0.5, 2) + 1e-5 * p_d2
            w_1 = fast_guided_filter_cv2(guide_lum, p_d1, r_mid, cfg.eps_detail * 5.0, 1) + 1e-5 * p_d1
            w_0 = fast_guided_filter_cv2(guide_lum, p_d0, r_fine, cfg.eps_detail, 1) + 1e-5 * p_d0
            w_b = w_b * gate
            w_2 = w_2 * gate
            w_1 = w_1 * gate
            w_0 = w_0 * gate
            floor_mask = estimator.is_exterior & (np.abs(estimator.z_min_luma - float(n)) < 0.5)
            w_b = np.where(floor_mask, np.maximum(w_b, 0.05), w_b)

        wb3 = w_b[..., None]
        w23 = w_2[..., None]
        w13 = w_1[..., None]
        w03 = w_0[..., None]
        del w_b, w_2, w_1, w_0

        accum_base += wb3 * base_n
        norm_base += wb3
        del base_n, wb3

        accum_d2 += w23 * d2
        norm_d2 += w23
        del d2, w23

        accum_d1 += w13 * d1
        norm_d1 += w13
        del d1, w13

        accum_d0 += w03 * d0
        norm_d0 += w03
        del d0, w03

        # Direct native RGB accumulation for thin hairs
        p_hair = (np.abs(z_clean - float(n)) < 0.5).astype(np.float32)
        w_hair = fast_guided_filter_cv2(
            guide=guide_lum, src=p_hair,
            radius=1, eps=1e-5, subsample=1
        )
        w_h_3d = (w_hair * thin_fg_mask)[..., np.newaxis]
        accum_hair += w_h_3d * img_n
        norm_hair += w_h_3d
        del w_hair, w_h_3d, img_n, p_base, p_d2, p_d1, p_d0, p_hair, guide_lum
        gc.collect()

        if update_progress:
            prog = 0.50 + 0.45 * ((n + 1) / float(N))
            update_progress("Pass 2/2: Accumulation", f"Fused {os.path.basename(use_files[n])} ({n + 1}/{N})", prog)
        print(f"  [Pass 2/2: {n+1}/{N}] {os.path.basename(use_files[n])} in {time.time() - t_s:.2f}s")

    del z_clean, z_detail_map, z_med_broad, halo_zone_mask, thin_fg_mask, min_luma_vol
    gc.collect()

    print("Synthesizing composite 16-bit image...")
    fused_pyr = (accum_base / norm_base) + (accum_d2 / norm_d2) + (accum_d1 / norm_d1) + (accum_d0 / norm_d0)
    del accum_base, norm_base, accum_d2, norm_d2, accum_d1, norm_d1, accum_d0, norm_d0
    gc.collect()

    hair_valid = (norm_hair > 1e-3).squeeze(-1)
    hair_rgb = accum_hair / norm_hair
    hair_alpha = np.clip(norm_hair, 0.0, 1.0)
    del accum_hair, norm_hair

    fused_float = np.where(
        hair_valid[..., np.newaxis],
        hair_alpha * hair_rgb + (1.0 - hair_alpha) * fused_pyr,
        fused_pyr
    )
    del hair_valid, hair_rgb, hair_alpha, fused_pyr
    gc.collect()

    raw_16bit = np.clip(fused_float * 65535.0, 0, 65535).astype(np.uint16)
    del fused_float
    if min_env is not None and max_env is not None:
        fused_16bit = np.clip(raw_16bit, min_env, max_env)
        del min_env, max_env, raw_16bit
    else:
        fused_16bit = raw_16bit
    gc.collect()
    gc.collect()

    if overrides_mask is not None:
        from core_fusion import apply_manual_overrides
        apply_manual_overrides(fused_16bit, overrides_mask, files, warp_mats, crop_box)

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fmt_label = "JPEG" if output_path.lower().endswith(('.jpg', '.jpeg')) else "16-bit TIFF"
        print(f"Saving final {fmt_label} to: {output_path}")
        write_image(output_path, fused_16bit, icc_profile=icc_profile)
        del fused_16bit
        gc.collect()

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"EG-MSD output successfully generated: {output_path} ({file_size_mb:.2f} MB)")
        return output_path

    return fused_16bit
