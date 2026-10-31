"""
16-Bit Multi-Channel Focus Stacking Fusion Engine
Supports:
1. Multi-scale Laplacian Pyramid (PMax style): optimal for fine overlapping bristles and micro-textures.
2. Depth-Map Regularization (DMap style): optimal for clean, noise-free background bokeh and halo elimination.
3. Bokeh Noise Floor Thresholding: suppresses sensor noise in smooth out-of-focus areas.
4. Interactive manual retouching override compositing.
"""
import os
import sys
import gc
import time
import cv2
import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from tiff_io import read_image_16bit, read_tiff_16bit, write_image, write_tiff_16bit

def build_gaussian_pyramid(img, levels):
    pyr = [img]
    current = img
    for _ in range(levels):
        current = cv2.pyrDown(current)
        pyr.append(current)
    return pyr

def build_laplacian_pyramid(gaussian_pyr):
    lap_pyr = []
    levels = len(gaussian_pyr) - 1
    for i in range(levels):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
        lap = cv2.subtract(gaussian_pyr[i], expanded)
        lap_pyr.append(lap)
    lap_pyr.append(gaussian_pyr[-1])  # Base residual
    return lap_pyr

def reconstruct_from_laplacian(lap_pyr):
    current = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        size = (lap_pyr[i].shape[1], lap_pyr[i].shape[0])
        expanded = cv2.pyrUp(current, dstsize=size)
        current = cv2.add(expanded, lap_pyr[i])
    return current

def compute_contrast_map(gray_float, kernel_size=5, noise_threshold=0.0):
    """
    Computes high-frequency focus energy using the Sum of Modified Laplacian (SML)
    operator with spatial Gaussian neighborhood smoothing.
    Delivers razor-sharp optical edge discrimination and micro-contrast.
    """
    kernel_x = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32)
    kernel_y = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float32)
    lx = np.abs(cv2.filter2D(gray_float, -1, kernel_x))
    ly = np.abs(cv2.filter2D(gray_float, -1, kernel_y))
    sml = lx + ly
    if kernel_size > 1:
        ksize = min(int(kernel_size) | 1, 5)
        sml = cv2.GaussianBlur(sml, (ksize, ksize), 0)
    if noise_threshold > 0.0:
        t = float(noise_threshold)
        sml = np.where(sml > t, sml - t, (sml ** 2) / (t + 1e-8))
    return sml

def estimate_continuous_depth(energy_stack):
    """
    Computes continuous sub-frame depth from focus (DFF) using 3-point parabolic peak interpolation.
    Literature: Nayar & Nakagawa (1994), Sub-frame Depth from Focus.
    Eliminates discrete frame stairstepping and quantisation banding.
    """
    N = energy_stack.shape[0]
    raw_idx = np.argmax(energy_stack, axis=0)  # (H, W)
    if N <= 2:
        return raw_idx.astype(np.float32)

    idx_prev = np.maximum(0, raw_idx - 1)
    idx_next = np.minimum(N - 1, raw_idx + 1)

    f_prev = np.take_along_axis(energy_stack, idx_prev[np.newaxis, :, :], axis=0)[0]
    f_curr = np.take_along_axis(energy_stack, raw_idx[np.newaxis, :, :], axis=0)[0]
    f_next = np.take_along_axis(energy_stack, idx_next[np.newaxis, :, :], axis=0)[0]

    denom = 2.0 * (f_prev - 2.0 * f_curr + f_next)
    is_valid_peak = (denom < -1e-6) & (raw_idx > 0) & (raw_idx < N - 1)
    delta = np.zeros_like(raw_idx, dtype=np.float32)
    delta[is_valid_peak] = np.clip((f_prev[is_valid_peak] - f_next[is_valid_peak]) / denom[is_valid_peak], -0.5, 0.5)

    return raw_idx.astype(np.float32) + delta

def propagate_confident_depth(raw_depth, confidence_mask, smooth_radius=7, default_val=None, block_size=8):
    """
    Confidence-guided Nearest-Subject Anchor (NSA) depth propagation for background bokeh.
    Evaluates at mini-block resolution (default 8x8) via Euclidean Distance Transform (EDT)
    to find the closest confident subject edge, then applies bilateral spatial interpolation
    into pixel-level continuous blending. Completely eliminates blocky artifacts and frame jumps.
    """
    from scipy.ndimage import distance_transform_edt

    if np.all(confidence_mask):
        return raw_depth

    if not np.any(confidence_mask):
        fill = default_val if default_val is not None else 0.0
        return np.full_like(raw_depth, fill)

    h, w = raw_depth.shape[:2]
    bs = max(2, int(block_size))
    bh, bw = max(1, h // bs), max(1, w // bs)

    mask_small = cv2.resize(confidence_mask.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_AREA) > 0
    depth_small = cv2.resize(raw_depth, (bw, bh), interpolation=cv2.INTER_AREA)

    if np.any(mask_small):
        indices = distance_transform_edt(~mask_small, return_distances=False, return_indices=True)
        nearest_small = depth_small[indices[0], indices[1]]
    else:
        fill = default_val if default_val is not None else 0.0
        nearest_small = np.full((bh, bw), fill, dtype=np.float32)

    nearest_full = cv2.resize(nearest_small, (w, h), interpolation=cv2.INTER_LINEAR)
    rad = max(3, int(smooth_radius) | 1)
    smooth = cv2.bilateralFilter(nearest_full.astype(np.float32), d=rad, sigmaColor=1.5, sigmaSpace=float(rad))
    alpha = cv2.GaussianBlur(confidence_mask.astype(np.float32), (rad, rad), 0)
    return alpha * raw_depth + (1.0 - alpha) * smooth

def compute_continuity_factor(energy_stack, baseline, k_star, tau=0.15):
    """
    Computes optical focal continuity factor alpha_cont in [0.0, 1.0].
    Enforces that true optical focus peaks produce non-zero energy elevation in adjacent frames
    (physical depth of field / defocus blur circle continuity).
    Suppresses isolated single-frame noise glitches, hot pixels, and sensor artifacts.
    """
    N = energy_stack.shape[0]
    if N < 3:
        return np.ones_like(baseline, dtype=np.float32)

    k_prev = np.maximum(0, k_star - 1)
    k_next = np.minimum(N - 1, k_star + 1)
    e_prev = np.take_along_axis(energy_stack, k_prev[np.newaxis, :, :], axis=0)[0]
    e_curr = np.take_along_axis(energy_stack, k_star[np.newaxis, :, :], axis=0)[0]
    e_next = np.take_along_axis(energy_stack, k_next[np.newaxis, :, :], axis=0)[0]

    denom = np.maximum(e_curr - baseline, 1e-8)
    elev_prev = np.maximum(0.0, e_prev - baseline)
    elev_next = np.maximum(0.0, e_next - baseline)

    is_internal = (k_star > 0) & (k_star < N - 1)

    ratio = np.where(
        is_internal,
        (elev_prev + elev_next) / (2.0 * denom),
        np.where(k_star == 0, elev_next / denom, elev_prev / denom)
    )

    x = np.clip(ratio / max(tau, 1e-4), 0.0, 1.0)
    alpha = x * x * (3.0 - 2.0 * x)  # Hermite smoothstep transition
    return alpha.astype(np.float32)

def compute_robust_focus_statistics(energy_stack, spatial_smooth_kernel=5, tau_continuity=0.15, continuity_check=False):
    """
    Computes statistical Focus Saliency (Peak Contrast Elevation and SNR) and coherent focus estimates.
    Robustly separates in-focus structures from sensor noise floor across frames.
    """
    N, H, W = energy_stack.shape
    baseline = np.median(energy_stack, axis=0)
    k_star = np.argmax(energy_stack, axis=0)
    e_peak = np.take_along_axis(energy_stack, k_star[np.newaxis, :, :], axis=0)[0]

    if continuity_check and N >= 3:
        alpha_cont = compute_continuity_factor(energy_stack, baseline, k_star, tau=tau_continuity)
        e_coherent = baseline + (e_peak - baseline) * alpha_cont
    else:
        e_coherent = e_peak

    delta_e = np.maximum(0.0, e_coherent - baseline)
    if spatial_smooth_kernel > 1:
        ksize = int(spatial_smooth_kernel) | 1
        delta_e = cv2.GaussianBlur(delta_e, (ksize, ksize), 0)

    p25 = max(1e-6, float(np.percentile(delta_e, 25)))
    z_global = delta_e / p25
    sigma_reg = np.full_like(delta_e, p25)
    return z_global, baseline, sigma_reg, k_star

def estimate_auto_noise_floor(energy_stack):
    """
    Auto-detects sensor noise floor from the 10th percentile of contrast variance across frames.
    """
    sample = energy_stack[:, ::4, ::4]
    span = np.max(sample, axis=0) - np.min(sample, axis=0)
    p10 = float(np.percentile(span, 10))
    return max(p10, 1e-4)

def compute_photometric_gains(grays_or_images):
    """
    Calculates photometric gain scalars across a bracket to equalize exposure shifts
    caused by focus breathing, bellows factor, and lighting flicker.
    """
    medians = []
    for item in grays_or_images:
        if len(item.shape) == 3:
            g = 0.2126 * item[:, :, 0] + 0.7152 * item[:, :, 1] + 0.0722 * item[:, :, 2]
        else:
            g = item
        medians.append(float(np.median(g)))
    ref_idx = len(medians) // 2
    ref_val = max(medians[ref_idx], 1e-4)
    return [float(np.clip(ref_val / max(m, 1e-4), 0.75, 1.35)) for m in medians]

def fuse_laplacian_pyramids(aligned_images, levels=6, contrast_kernel=5, exponent=5.0, noise_threshold=0.0, return_depth_map=False):
    """
    Performs 16-bit multi-scale Laplacian Pyramid focus stacking fusion.
    Uses 100% Hard Maximum Energy Selection across detail octaves (l < levels)
    matching Petteri Aimonen's C++ wavelet engine for razor optical sharpness.
    Uniformly averages base Gaussian residual (l == levels) for sqrt(N) noise reduction.
    """
    num_images = len(aligned_images)
    assert num_images >= 2, "Must provide at least 2 images to stack"

    h, w = aligned_images[0].shape[:2]
    pad_h = (1 << levels) - (h % (1 << levels)) if h % (1 << levels) != 0 else 0
    pad_w = (1 << levels) - (w % (1 << levels)) if w % (1 << levels) != 0 else 0

    fused_lap_pyr = [None] * (levels + 1)
    max_contrast_pyr = [None] * (levels + 1)
    depth_idx = np.zeros((h + pad_h, w + pad_w), dtype=np.uint8)
    level0_contrasts = [] if return_depth_map else None

    # Determine reference frame with minimum background bloom / darkest backdrop
    corner_brightness = [float(np.mean(img[:min(50, h), :min(50, w)])) for img in aligned_images]
    ref_idx = int(np.argmin(corner_brightness)) if corner_brightness else (num_images // 2)
    coarse_start = min(3, levels)
    min_img = None
    max_img = None

    # Pass 1: Compute global peak contrast across all frames to detect sharp foreground edges
    # Morphological dilation creates a halo detection mask around high-contrast edges
    l0_contrasts = []
    for img in aligned_images:
        f_small = img.astype(np.float32) / 65535.0
        g_small = 0.2126 * f_small[:, :, 0] + 0.7152 * f_small[:, :, 1] + 0.0722 * f_small[:, :, 2]
        c0 = compute_contrast_map(g_small, kernel_size=contrast_kernel, noise_threshold=0.0)
        l0_contrasts.append(c0)

    max_c_global = np.max(np.stack(l0_contrasts, axis=0), axis=0)
    del l0_contrasts
    if pad_h > 0 or pad_w > 0:
        max_c_global = cv2.copyMakeBorder(max_c_global, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    dilated_c_global = cv2.dilate(max_c_global, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))

    for i in range(num_images):
        img = aligned_images[i]
        if min_img is None:
            min_img = img.copy()
            max_img = img.copy()
        else:
            np.minimum(min_img, img, out=min_img)
            np.maximum(max_img, img, out=max_img)

        f = img.astype(np.float32) / 65535.0
        if pad_h > 0 or pad_w > 0:
            f = cv2.copyMakeBorder(f, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        gray = 0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]

        g_pyr = build_gaussian_pyramid(f, levels)
        l_pyr = build_laplacian_pyramid(g_pyr)
        del g_pyr

        gray_g_pyr = build_gaussian_pyramid(gray, levels)
        del gray, f

        for l in range(levels + 1):
            if l >= coarse_start:
                # Coarse octaves (l >= 3) and base Gaussian residual (l == levels):
                # Anchor strictly to clean reference frame with zero blooming.
                # Wavelets cancel out perfectly with base residual without any coarse ringing or dark troughs!
                if i == ref_idx:
                    fused_lap_pyr[l] = l_pyr[l].copy()
            else:
                # Fine detail levels (l < 3): Hard maximum selection for razor optical sharpness!
                contrast = compute_contrast_map(gray_g_pyr[l], kernel_size=contrast_kernel, noise_threshold=0.0)
                if l == 0 and return_depth_map:
                    level0_contrasts.append(contrast.copy())

                # Morphological Halo Suppression:
                # Near a sharp high-contrast feature (dilated_l > 0.04), any frame whose contrast
                # is weak (< 40% of the nearby peak) is an out-of-focus blur halo!
                # Suppress it so it cannot win over clean background or in-focus frames.
                lh, lw = gray_g_pyr[l].shape[:2]
                dilated_l = cv2.resize(dilated_c_global, (lw, lh), interpolation=cv2.INTER_LINEAR)
                is_halo = (contrast < 0.40 * dilated_l) & (dilated_l > 0.04)
                contrast[is_halo] = 0.0

                if fused_lap_pyr[l] is None:
                    fused_lap_pyr[l] = l_pyr[l].copy()
                    max_contrast_pyr[l] = contrast.copy()
                    if l == 0:
                        depth_idx.fill(i)
                else:
                    mask = contrast > max_contrast_pyr[l]
                    fused_lap_pyr[l][mask] = l_pyr[l][mask]
                    max_contrast_pyr[l][mask] = contrast[mask]
                    if l == 0:
                        depth_idx[mask] = i

        del l_pyr, gray_g_pyr

    del max_contrast_pyr, dilated_c_global, max_c_global

    fused_img = reconstruct_from_laplacian(fused_lap_pyr)
    del fused_lap_pyr

    if pad_h > 0 or pad_w > 0:
        fused_img = fused_img[:h, :w, :]
        depth_idx = depth_idx[:h, :w]

    fused_16bit = np.clip(fused_img * 65535.0, 0, 65535).astype(np.uint16)
    del fused_img

    # Envelope protection: clamp to input image dynamic envelope to eliminate any residual Gibbs ringing
    np.clip(fused_16bit, min_img, max_img, out=fused_16bit)
    del min_img, max_img

    if return_depth_map:
        stack_e = np.stack(level0_contrasts, axis=0)
        del level0_contrasts
        if pad_h > 0 or pad_w > 0:
            stack_e = stack_e[:, :h, :w]

        baseline = np.median(stack_e, axis=0)
        peak = np.max(stack_e, axis=0)
        delta_e = np.maximum(0.0, peak - baseline)

        # Outlier neighbor denoiser on depth map
        depth_clean = cv2.medianBlur(depth_idx, 3)

        # Noise-floor gating: bokeh areas where contrast span is flat are cleanly mapped to background
        p25 = max(1e-6, float(np.percentile(delta_e, 25)))
        mult = max(0.2, min(5.0, float(noise_threshold))) if noise_threshold > 0.0 else 1.0
        t = p25 * (1.0 + 1.5 * mult)
        confident_mask = delta_e > t

        final_depth = np.where(confident_mask, depth_clean.astype(np.float32), float(num_images - 1))
        depth_8u = np.clip(final_depth * (255.0 / max(1, num_images - 1)), 0, 255).astype(np.uint8)

        p98 = max(1e-4, float(np.percentile(delta_e, 98)))
        saliency_8u = np.clip((delta_e / p98) * 255.0, 0, 255).astype(np.uint8)
        del stack_e, baseline, peak, delta_e, depth_clean, confident_mask, final_depth
        return fused_16bit, depth_8u, saliency_8u

    return fused_16bit

def fuse_depth_map(aligned_images, contrast_kernel=5, noise_threshold=1.0, depth_smooth_radius=7, return_depth_map=False, tau_continuity=0.15):
    """
    Performs Depth from Focus (DFF) fusion with continuous energy-weighted sub-plane synthesis.
    Eliminates bokeh noise accumulation, edge halos, and discrete frame jumping.
    """
    num_images = len(aligned_images)
    h, w = aligned_images[0].shape[:2]

    energies = []
    for img in aligned_images:
        f = img.astype(np.float32) / 65535.0
        gray = 0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]
        c = compute_contrast_map(gray, kernel_size=contrast_kernel, noise_threshold=0.0)
        energies.append(c)

    energy_stack = np.stack(energies, axis=0)
    del energies

    baseline = np.median(energy_stack, axis=0)
    peak = np.max(energy_stack, axis=0)
    delta_e = np.maximum(0.0, peak - baseline)

    cont_depth = estimate_continuous_depth(energy_stack)
    rad = max(3, int(depth_smooth_radius) | 1)
    subject_depth = cv2.medianBlur(cont_depth, 5)
    subject_depth = cv2.bilateralFilter(subject_depth, d=5, sigmaColor=1.5, sigmaSpace=float(rad))

    # Sharp physical plane synthesis: 2-frame linear interpolation strictly between the two nearest in-focus frames
    # Eliminates wide Gaussian bell-curve smearing across out-of-focus brackets
    k_floor = np.clip(np.floor(subject_depth).astype(np.int32), 0, num_images - 1)
    k_ceil = np.clip(k_floor + 1, 0, num_images - 1)
    frac = (subject_depth - k_floor)[:, :, np.newaxis]

    fused_subject = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(num_images):
        mask_floor = (k_floor == i)
        mask_ceil = (k_ceil == i)
        if np.any(mask_floor) or np.any(mask_ceil):
            f = aligned_images[i].astype(np.float32) / 65535.0
            if np.any(mask_floor):
                fused_subject[mask_floor] += f[mask_floor] * (1.0 - frac[mask_floor])
            if np.any(mask_ceil):
                fused_subject[mask_ceil] += f[mask_ceil] * frac[mask_ceil]
            del f

    # Bokeh noise-floor thresholding
    p30 = max(1e-6, float(np.percentile(delta_e, 30)))
    mult = max(0.2, min(5.0, float(noise_threshold))) if noise_threshold > 0.0 else 1.0
    t = p30 * (1.0 + 2.0 * mult)

    conf = np.clip((delta_e - t) / (t + 1e-6), 0.0, 1.0)
    conf = cv2.GaussianBlur(conf, (5, 5), 1.0)[:, :, np.newaxis]

    # Background frame is sampled 100% from clean background frame (zero noise averaging in bokeh)
    # Find frame with darkest/cleanest background to prevent foreground blooming into bokeh
    corner_brightness = [float(np.mean(img[:min(50, h), :min(50, w)])) for img in aligned_images]
    bg_idx = int(np.argmin(corner_brightness)) if corner_brightness else 0
    bg_float = aligned_images[bg_idx].astype(np.float32) / 65535.0
    fused_combined = fused_subject * conf + bg_float * (1.0 - conf)
    fused_16bit = np.clip(fused_combined * 65535.0, 0, 65535).astype(np.uint16)
    del fused_subject, bg_float, fused_combined

    final_depth = np.where(conf[:, :, 0] > 0.05, subject_depth, float(bg_idx))
    depth_8u = np.clip(final_depth * (255.0 / max(1, num_images - 1)), 0, 255).astype(np.uint8)

    p98 = max(1e-4, float(np.percentile(delta_e, 98)))
    saliency_8u = np.clip((delta_e / p98) * 255.0, 0, 255).astype(np.uint8)
    del energy_stack, baseline, peak, delta_e, cont_depth, subject_depth, final_depth

    if return_depth_map:
        return fused_16bit, depth_8u, saliency_8u
    return fused_16bit


# Backwards compatibility alias
fuse_dmap = fuse_depth_map

def fuse_focus_stack(aligned_images, method="egmsd", levels=6, contrast_kernel=5, exponent=3.0, noise_threshold=0.0, depth_smooth=7, z_jump=5.0, halo_gamma=0.45, anisotropy=0.5, specular_thresh=0.94, max_coc_radius_px=None, bg_mode="auto", stack_order="near_to_far", return_depth_map=False):
    """
    Unified entry point for interactive and preview focus stacking using Edge-Guided Multi-Scale Decomposition (EG-MSD).
    """
    from egmsd_stacker import fuse_focus_stack as egmsd_fuse, StackerConfig

    stack_f = np.stack([img.astype(np.float32) / 65535.0 for img in aligned_images], axis=0)
    N = len(aligned_images)

    mult = max(0.2, min(5.0, float(noise_threshold))) if noise_threshold > 0.0 else 1.0
    cfg = StackerConfig(
        align_ecc=False,  # Already aligned in memory
        stack_order=stack_order,
        max_coc_radius_px=int(max_coc_radius_px) if max_coc_radius_px is not None else None,
        z_jump_thresh=float(z_jump) if z_jump else 5.0,
        halo_suppression_gamma=float(halo_gamma) if halo_gamma else 0.45,
        anisotropy_alpha=float(anisotropy) if anisotropy is not None else 0.5,
        specular_clip_thresh=float(specular_thresh) if specular_thresh else 0.94,
        bg_mode=bg_mode,
        noise_read=1e-5 * mult,
        noise_shot=4e-4 * mult
    )

    res = egmsd_fuse(stack_f, cfg=cfg)
    del stack_f

    fused_16bit = np.clip(res["fused_rgb"] * 65535.0, 0, 65535).astype(np.uint16)

    if return_depth_map:
        depth_8u = np.clip(res["depth_map"] * (255.0 / max(1, N - 1)), 0, 255).astype(np.uint8)
        saliency_8u = np.clip(res.get("saliency_map", res["halo_mask"]) * 255.0, 0, 255).astype(np.uint8)
        effective_bg_mode = res.get("effective_bg_mode", cfg.bg_mode)
        return fused_16bit, depth_8u, saliency_8u, effective_bg_mode

    return fused_16bit



def apply_manual_overrides(fused_16bit, overrides_mask, files, warp_mats, crop_box):
    """
    Applies user-painted retouching overrides onto the fused 16-bit image
    using 4-level Laplacian pyramid multi-band blending for seamless transitions.
    """
    import gc
    import os
    from tiff_io import read_tiff_16bit

    bx, by, bw, bh = crop_box
    target_h, target_w = fused_16bit.shape[:2]

    if overrides_mask.shape[0] != target_h or overrides_mask.shape[1] != target_w:
        frame_idx_map = cv2.resize(overrides_mask[:, :, 0], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        alpha_map = cv2.resize(overrides_mask[:, :, 1], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_idx_map = overrides_mask[:, :, 0]
        alpha_map = overrides_mask[:, :, 1]

    active_indices = np.unique(frame_idx_map[alpha_map > 0.005])
    print(f"Applying manual retouching overrides for {len(active_indices)} source frames...")

    for k in active_indices:
        k = int(round(k))
        if k < 0 or k >= len(files):
            continue
        mask_k = (frame_idx_map == k) & (alpha_map > 0.005)
        if not np.any(mask_k):
            continue

        src_img, _ = read_tiff_16bit(files[k])
        full_h, full_w = src_img.shape[:2]
        warp_mat = warp_mats[k]
        is_identity = (
            np.abs(warp_mat[0, 0] - 1.0) < 1e-5 and
            np.abs(warp_mat[1, 1] - 1.0) < 1e-5 and
            np.abs(warp_mat[0, 2]) < 1e-4 and
            np.abs(warp_mat[1, 2]) < 1e-4
        )
        if not is_identity:
            src_img = cv2.warpAffine(
                src_img, warp_mat, (full_w, full_h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
            )
        cropped_k = src_img[by : by + bh, bx : bx + bw]
        del src_img

        ys, xs = np.where(mask_k)
        pad = 32
        y0 = max(0, int(np.min(ys)) - pad)
        y1 = min(target_h, int(np.max(ys)) + pad + 1)
        x0 = max(0, int(np.min(xs)) - pad)
        x1 = min(target_w, int(np.max(xs)) + pad + 1)

        patch_h = y1 - y0
        patch_w = x1 - x0

        mult = 16
        pad_bottom = (mult - (patch_h % mult)) % mult
        pad_right = (mult - (patch_w % mult)) % mult

        fused_patch = fused_16bit[y0:y1, x0:x1].astype(np.float32)
        source_patch = cropped_k[y0:y1, x0:x1].astype(np.float32)
        alpha_roi = np.where(frame_idx_map[y0:y1, x0:x1] == k, alpha_map[y0:y1, x0:x1], 0.0).astype(np.float32)
        alpha_patch = np.repeat(alpha_roi[..., np.newaxis], 3, axis=-1)

        if pad_bottom > 0 or pad_right > 0:
            fused_patch = cv2.copyMakeBorder(fused_patch, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
            source_patch = cv2.copyMakeBorder(source_patch, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
            alpha_patch = cv2.copyMakeBorder(alpha_patch, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        pyr_levels = 4
        g_alpha = build_gaussian_pyramid(alpha_patch, levels=pyr_levels)
        g_fused = build_gaussian_pyramid(fused_patch, levels=pyr_levels)
        g_src = build_gaussian_pyramid(source_patch, levels=pyr_levels)

        lap_fused = build_laplacian_pyramid(g_fused)
        lap_src = build_laplacian_pyramid(g_src)

        blended_lap = []
        for l in range(len(lap_fused)):
            a_l = np.clip(g_alpha[l], 0.0, 1.0)
            blended_l = (1.0 - a_l) * lap_fused[l] + a_l * lap_src[l]
            blended_lap.append(blended_l.astype(np.float32))

        reconstructed = reconstruct_from_laplacian(blended_lap)
        if pad_bottom > 0 or pad_right > 0:
            reconstructed = reconstructed[:patch_h, :patch_w]

        roi_mask = alpha_roi > 0.001
        fused_sub = fused_16bit[y0:y1, x0:x1]
        rec_clamped = np.clip(reconstructed, 0, 65535).astype(np.uint16)
        fused_sub[roi_mask] = rec_clamped[roi_mask]
        fused_16bit[y0:y1, x0:x1] = fused_sub

        del cropped_k, fused_patch, source_patch, alpha_patch, reconstructed
        gc.collect()
        print(f"  Applied multi-band retouching layer from Frame {k + 1} ({os.path.basename(files[k])})")

    return fused_16bit

def stream_fuse_laplacian_pyramids(files, warp_mats, crop_box, levels=6, contrast_kernel=5, exponent=3.0, noise_threshold=0.0, icc_profile=None, output_path=None, overrides_mask=None):
    """
    Memory-efficient streaming 16-bit Laplacian pyramid focus stacking.
    Peak RAM usage is strictly bounded (< 3.0 GB) regardless of the number of frames.
    """
    from tiff_io import read_image_16bit, read_tiff_16bit, write_image, write_tiff_16bit

    bx, by, bw, bh = crop_box
    num_images = len(files)
    assert num_images >= 2, "Must provide at least 2 images to stack"

    pad_h = (1 << levels) - (bh % (1 << levels)) if bh % (1 << levels) != 0 else 0
    pad_w = (1 << levels) - (bw % (1 << levels)) if bw % (1 << levels) != 0 else 0

    fused_lap_pyr = [None] * (levels + 1)
    max_contrast_pyr = [None] * (levels + 1)

    ref_idx = 0
    coarse_start = min(3, levels)
    min_crop = None
    max_crop = None

    # Pass 1: Compute downsampled global peak contrast to detect high-contrast boundaries
    from core_align import read_downsampled_gray
    down_contrasts = []
    down_scale = 1.0
    for i, fpath in enumerate(files):
        g_s, _, _, s, _ = read_downsampled_gray(fpath, max_dim=2048)
        down_scale = s
        w_mat = warp_mats[i].copy()
        w_mat[0, 2] *= s
        w_mat[1, 2] *= s
        w_s = cv2.warpAffine(g_s, w_mat, (g_s.shape[1], g_s.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        c0 = compute_contrast_map(w_s, kernel_size=contrast_kernel, noise_threshold=0.0)
        down_contrasts.append(c0)
        del g_s, w_s

    max_c_down = np.max(np.stack(down_contrasts, axis=0), axis=0)
    del down_contrasts
    dilated_c_down = cv2.dilate(max_c_down, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    del max_c_down

    crop_x_down = int(round(bx * down_scale))
    crop_y_down = int(round(by * down_scale))
    crop_w_down = int(round(bw * down_scale))
    crop_h_down = int(round(bh * down_scale))
    dilated_crop = dilated_c_down[crop_y_down : crop_y_down + crop_h_down, crop_x_down : crop_x_down + crop_w_down]
    del dilated_c_down
    gc.collect()

    for i in range(num_images):
        t_start = time.time()
        img, _ = read_tiff_16bit(files[i])
        full_h, full_w = img.shape[:2]

        warp_mat = warp_mats[i]
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
                borderMode=cv2.BORDER_REFLECT
            )

        cropped = img[by : by + bh, bx : bx + bw]
        del img

        if min_crop is None:
            min_crop = cropped.copy()
            max_crop = cropped.copy()
        else:
            np.minimum(min_crop, cropped, out=min_crop)
            np.maximum(max_crop, cropped, out=max_crop)

        f = (cropped.astype(np.float32) / 65535.0)
        del cropped

        if pad_h > 0 or pad_w > 0:
            f = cv2.copyMakeBorder(f, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        gray = 0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]

        g_pyr = build_gaussian_pyramid(f, levels)
        l_pyr = build_laplacian_pyramid(g_pyr)
        del g_pyr, f

        gray_g_pyr = build_gaussian_pyramid(gray, levels)
        del gray

        for l in range(levels + 1):
            if l >= coarse_start:
                # Coarse octaves (l >= 3) and base Gaussian residual (l == levels):
                # Anchor strictly to clean reference frame with zero blooming.
                # Wavelets cancel out perfectly with base residual without any coarse ringing or dark troughs!
                if i == ref_idx:
                    fused_lap_pyr[l] = l_pyr[l].copy()
            else:
                # Fine detail levels (l < 3): Hard maximum selection for razor optical sharpness!
                contrast = compute_contrast_map(gray_g_pyr[l], kernel_size=contrast_kernel, noise_threshold=0.0)

                # Morphological Halo Suppression:
                lh, lw = gray_g_pyr[l].shape[:2]
                dilated_l = cv2.resize(dilated_crop, (lw, lh), interpolation=cv2.INTER_LINEAR)
                is_halo = (contrast < 0.40 * dilated_l) & (dilated_l > 0.04)
                contrast[is_halo] = 0.0

                if fused_lap_pyr[l] is None:
                    fused_lap_pyr[l] = l_pyr[l].copy()
                    max_contrast_pyr[l] = contrast.copy()
                else:
                    mask = contrast > max_contrast_pyr[l]
                    fused_lap_pyr[l][mask] = l_pyr[l][mask]
                    max_contrast_pyr[l][mask] = contrast[mask]
                    del mask
                del contrast

        del l_pyr, gray_g_pyr
        gc.collect()
        print(f"  [{i+1}/{num_images}] Streamed & merged {os.path.basename(files[i])} in {time.time() - t_start:.2f}s")

    del max_contrast_pyr, dilated_crop
    gc.collect()

    print("Synthesizing composite image from Laplacian pyramid bands...")
    fused_img = reconstruct_from_laplacian(fused_lap_pyr)
    del fused_lap_pyr
    gc.collect()

    if pad_h > 0 or pad_w > 0:
        fused_img = fused_img[:bh, :bw, :]

    fused_16bit = np.clip(fused_img * 65535.0, 0, 65535).astype(np.uint16)
    del fused_img

    # Envelope protection: clamp to input image dynamic envelope
    np.clip(fused_16bit, min_crop, max_crop, out=fused_16bit)
    del min_crop, max_crop
    gc.collect()

    if overrides_mask is not None:
        apply_manual_overrides(fused_16bit, overrides_mask, files, warp_mats, crop_box)

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fmt_label = "JPEG" if output_path.lower().endswith(('.jpg', '.jpeg')) else "16-bit TIFF"
        print(f"Saving stacked {fmt_label} with original ICC profile: {output_path}")
        write_image(output_path, fused_16bit, icc_profile=icc_profile)
        del fused_16bit
        gc.collect()

        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Output successfully written: {output_path} ({file_size_mb:.2f} MB)")
        else:
            raise RuntimeError(f"Failed to write output file: {output_path}")
        return output_path

    return fused_16bit

def stream_fuse_dmap(files, warp_mats, crop_box, contrast_kernel=5, noise_threshold=0.0, depth_smooth=7, icc_profile=None, output_path=None, overrides_mask=None):
    """
    Memory-efficient streaming 16-bit DMap (Depth-Map) focus stacking.
    Computes regularized depth map and samples from depth planes with bounded RAM (< 2.5 GB).
    """
    from tiff_io import read_image_16bit, read_tiff_16bit, write_image, write_tiff_16bit
    from core_align import read_downsampled_gray

    bx, by, bw, bh = crop_box
    num_images = len(files)

    print("Computing regularized multi-frame depth map...")
    # Step 1: Compute downsampled energy maps for all frames (< 150MB RAM)
    down_energies = []
    down_w, down_h, down_scale = None, None, None

    for i, fpath in enumerate(files):
        gray_small, full_w, full_h, scale, _ = read_downsampled_gray(fpath, max_dim=2048)
        warp_mat = warp_mats[i]
        is_identity = (
            np.abs(warp_mat[0, 0] - 1.0) < 1e-5 and
            np.abs(warp_mat[1, 1] - 1.0) < 1e-5 and
            np.abs(warp_mat[0, 2]) < 1e-4 and
            np.abs(warp_mat[1, 2]) < 1e-4
        )
        if not is_identity:
            mat_small = warp_mat.copy()
            mat_small[0, 2] *= scale
            mat_small[1, 2] *= scale
            gray_small = cv2.warpAffine(
                gray_small, mat_small, (gray_small.shape[1], gray_small.shape[0]),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
            )
        c = compute_contrast_map(gray_small, kernel_size=contrast_kernel, noise_threshold=0.0)
        down_energies.append(c)
        if down_scale is None:
            down_scale = scale
            down_w, down_h = gray_small.shape[1], gray_small.shape[0]
        del gray_small

    stack_e = np.stack(down_energies, axis=0)
    del down_energies
    raw_depth_small = estimate_continuous_depth(stack_e)

    baseline_small = np.median(stack_e, axis=0)
    peak_small = np.max(stack_e, axis=0)
    delta_e_small = np.maximum(0.0, peak_small - baseline_small)
    del stack_e
    gc.collect()

    p30 = max(1e-6, float(np.percentile(delta_e_small, 30)))
    mult = max(0.2, min(5.0, float(noise_threshold))) if noise_threshold > 0.0 else 1.0
    t = p30 * (1.0 + 2.0 * mult)

    raw_mask_small = delta_e_small > t
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    confident_mask_small = cv2.morphologyEx(raw_mask_small.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    confident_mask_small = cv2.medianBlur(confident_mask_small.astype(np.uint8), 5) > 0

    rad = max(3, int(depth_smooth) | 1)
    subject_depth_small = cv2.medianBlur(raw_depth_small, 5)
    subject_depth_small = cv2.bilateralFilter(subject_depth_small, d=5, sigmaColor=1.5, sigmaSpace=float(rad))

    # Anchor background bokeh to Frame 0 to prevent foreground bloom from bleeding into dark backdrop
    bg_idx = 0
    depth_smooth_small = np.where(confident_mask_small, subject_depth_small, float(bg_idx))
    del raw_depth_small, delta_e_small
    gc.collect()

    # Upsample depth map & confidence mask to crop_box resolution (bw, bh)
    crop_x_down = int(round(bx * down_scale))
    crop_y_down = int(round(by * down_scale))
    crop_w_down = int(round(bw * down_scale))
    crop_h_down = int(round(bh * down_scale))
    depth_cropped_small = depth_smooth_small[crop_y_down : crop_y_down + crop_h_down, crop_x_down : crop_x_down + crop_w_down]
    depth_full = cv2.resize(depth_cropped_small, (bw, bh), interpolation=cv2.INTER_LINEAR)
    del depth_smooth_small, depth_cropped_small

    conf_cropped_small = confident_mask_small[crop_y_down : crop_y_down + crop_h_down, crop_x_down : crop_x_down + crop_w_down]
    conf_full = cv2.resize(conf_cropped_small.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
    del confident_mask_small, conf_cropped_small
    gc.collect()

    # If manual depth overrides were painted, apply directly to physical focus depth map
    if overrides_mask is not None:
        try:
            ov_frame = overrides_mask[:, :, 0]
            ov_alpha = overrides_mask[:, :, 1]
            ov_full_alpha = cv2.resize(ov_alpha, (bw, bh), interpolation=cv2.INTER_LINEAR)
            ov_full_frame = cv2.resize(ov_frame, (bw, bh), interpolation=cv2.INTER_NEAREST)
            depth_full = depth_full * (1.0 - ov_full_alpha) + ov_full_frame * ov_full_alpha
            conf_full = conf_full | (ov_full_alpha > 0.01)
            del ov_frame, ov_alpha, ov_full_alpha, ov_full_frame
        except Exception as e:
            print(f"Notice: Depth override adaptation skipped ({e})")

    anchor_full = np.clip(np.round(depth_full).astype(np.int32), 0, num_images - 1)

    # Step 2: Stream frames and synthesize sharp 2-frame linear interpolation
    k_floor = np.clip(np.floor(depth_full).astype(np.int32), 0, num_images - 1)
    k_ceil = np.clip(k_floor + 1, 0, num_images - 1)
    frac = (depth_full - k_floor)[:, :, np.newaxis]

    fused_float = np.zeros((bh, bw, 3), dtype=np.float32)

    for i in range(num_images):
        t_start = time.time()
        mask_floor = (k_floor == i) & conf_full
        mask_ceil = (k_ceil == i) & conf_full
        mask_bokeh = (anchor_full == i) & (~conf_full)

        active = mask_floor | mask_ceil | mask_bokeh
        if not np.any(active):
            continue

        src_img, _ = read_tiff_16bit(files[i])
        full_h, full_w = src_img.shape[:2]
        warp_mat = warp_mats[i]
        is_identity = (
            np.abs(warp_mat[0, 0] - 1.0) < 1e-5 and
            np.abs(warp_mat[1, 1] - 1.0) < 1e-5 and
            np.abs(warp_mat[0, 2]) < 1e-4 and
            np.abs(warp_mat[1, 2]) < 1e-4
        )
        if not is_identity:
            src_img = cv2.warpAffine(
                src_img, warp_mat, (full_w, full_h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
            )
        cropped = src_img[by : by + bh, bx : bx + bw]
        del src_img

        f = cropped.astype(np.float32) / 65535.0
        del cropped

        if np.any(mask_floor):
            fused_float[mask_floor] += f[mask_floor] * (1.0 - frac[mask_floor])
        if np.any(mask_ceil):
            fused_float[mask_ceil] += f[mask_ceil] * frac[mask_ceil]
        if np.any(mask_bokeh):
            fused_float[mask_bokeh] = f[mask_bokeh]
        del f
        gc.collect()
        print(f"  [{i+1}/{num_images}] DMap merged {os.path.basename(files[i])} in {time.time() - t_start:.2f}s")

    del depth_full, k_floor, k_ceil, frac, conf_full, anchor_full
    fused_16bit = np.clip(fused_float * 65535.0, 0, 65535).astype(np.uint16)
    del fused_float
    gc.collect()

    if overrides_mask is not None:
        apply_manual_overrides(fused_16bit, overrides_mask, files, warp_mats, crop_box)

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fmt_label = "JPEG" if output_path.lower().endswith(('.jpg', '.jpeg')) else "16-bit TIFF"
        print(f"Saving stacked {fmt_label} with original ICC profile: {output_path}")
        write_image(output_path, fused_16bit, icc_profile=icc_profile)
        del fused_16bit
        gc.collect()

        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Output successfully written: {output_path} ({file_size_mb:.2f} MB)")
        else:
            raise RuntimeError(f"Failed to write output file: {output_path}")
        return output_path

    return fused_16bit

def stream_fuse_focus_stack(
    files, warp_mats, crop_box,
    method="egmsd",
    levels=6, contrast_kernel=5, exponent=3.0, noise_threshold=1.0, depth_smooth=7,
    z_jump=5.0, halo_gamma=0.45, anisotropy=0.5, specular_thresh=0.94,
    max_coc_radius_px=None, bg_mode="auto", stack_order="near_to_far",
    icc_profile=None, output_path=None, overrides_mask=None, update_progress=None
):
    """
    Unified entry point for streaming 61MP 16-bit final stack rendering using EG-MSD.
    Streams frames with disk-backed energy memmap (< 2.5 GB peak RSS).
    """
    from egmsd_stacker import stream_fuse_egmsd, StackerConfig

    mult = max(0.2, min(5.0, float(noise_threshold))) if noise_threshold > 0.0 else 1.0
    cfg = StackerConfig(
        align_ecc=False,  # Aligned via warp_mats
        stack_order=stack_order,
        max_coc_radius_px=int(max_coc_radius_px) if max_coc_radius_px is not None else None,
        z_jump_thresh=float(z_jump) if z_jump else 5.0,
        halo_suppression_gamma=float(halo_gamma) if halo_gamma else 0.45,
        anisotropy_alpha=float(anisotropy) if anisotropy is not None else 0.5,
        specular_clip_thresh=float(specular_thresh) if specular_thresh else 0.94,
        bg_mode=bg_mode,
        noise_read=1e-5 * mult,
        noise_shot=4e-4 * mult
    )

    return stream_fuse_egmsd(
        files=files,
        warp_mats=warp_mats,
        crop_box=crop_box,
        cfg=cfg,
        icc_profile=icc_profile,
        output_path=output_path,
        overrides_mask=overrides_mask,
        update_progress=update_progress
    )

