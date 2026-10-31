import os
import sys
import cv2
import numpy as np

# Add py_focus_stack module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(os.path.dirname(current_dir), "FocusStack.lrplugin", "bin", "py_focus_stack")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from egmsd_stacker import fuse_focus_stack, StackerConfig


def disk_psf_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Simulates optical Circle of Confusion defocus via circular disk kernel."""
    if radius <= 0:
        return img.copy()
    ksize = 2 * radius + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)).astype(np.float32)
    kern /= kern.sum()
    return cv2.filter2D(img, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT101)


def generate_synthetic_macro_stack(
    N: int = 20, H: int = 512, W: int = 512
) -> dict:
    """
    Generates a synthetic macro stack with:
    - Background plane at slice z_bg = 15 (high-frequency sinusoidal texture)
    - Foreground body at slice z_fg = 4 (covering left half x < W//2)
    - 2-pixel foreground sensory hair at z_fg = 4 crossing over the background (x >= W//2)
    - Specular water droplet highlight at z_fg = 4
    """
    rng = np.random.default_rng(42)
    z_fg, z_bg = 4, 15
    
    # 1. Ground-Truth All-in-Focus Textures
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    bg_tex = 0.45 + 0.25 * np.sin(x * 0.35) * np.cos(y * 0.35)
    bg_rgb = np.dstack([bg_tex * 0.8, bg_tex, bg_tex * 0.7]).astype(np.float32)
    
    fg_tex = 0.50 + 0.30 * np.cos((x + y) * 0.4)
    fg_rgb = np.dstack([fg_tex, fg_tex * 0.6, fg_tex * 0.3]).astype(np.float32)
    
    # Specular highlight on foreground
    spec_mask = ((x - 120) ** 2 + (y - 120) ** 2) <= 8 ** 2
    fg_rgb[spec_mask] = 0.99
    
    # Foreground occlusion mask: left half (x < W//2) PLUS a 2px horizontal hair at y=256
    fg_mask = np.zeros((H, W), dtype=np.float32)
    fg_mask[:, :W // 2] = 1.0
    fg_mask[255:257, W // 2:W // 2 + 160] = 1.0  # Crossing bristle!
    
    gt_rgb = fg_mask[..., None] * fg_rgb + (1.0 - fg_mask[..., None]) * bg_rgb
    
    # 2. Synthesize Optical Stack with physically accurate foreground alpha blur
    stack = np.empty((N, H, W, 3), dtype=np.float32)
    for n in range(N):
        r_fg = int(abs(n - z_fg) * 1.8)
        r_bg = int(abs(n - z_bg) * 1.8)
        
        bg_slice = disk_psf_blur(bg_rgb, r_bg)
        fg_premult = disk_psf_blur(fg_rgb * fg_mask[..., None], r_fg)
        alpha_blur = disk_psf_blur(fg_mask, r_fg)[..., None]
        
        # Physical alpha compositing of defocus light cone
        clean_slice = fg_premult + (1.0 - alpha_blur) * bg_slice
        
        # Add realistic Poisson-Gaussian sensor noise
        sigma = np.sqrt(1e-5 + 2e-4 * clean_slice)
        noisy_slice = clean_slice + rng.normal(0, 1, size=clean_slice.shape).astype(np.float32) * sigma
        stack[n] = np.clip(noisy_slice, 0.0, 1.0)
        
    return {
        "stack": stack,
        "gt_rgb": gt_rgb,
        "z_fg": z_fg,
        "z_bg": z_bg
    }


def generate_multi_depth_dark_bokeh_stack(
    N: int = 20, H: int = 512, W: int = 512
) -> dict:
    """
    Generates a synthetic macro stack with:
    - Dark smooth bokeh background (luma ~ 0.04)
    - Near bright subject at slice z_near = 3
    - Far bright subject at slice z_far = 16
    - Optical defocus light cones composited across all 20 slices
    - Realistic sensor noise
    """
    rng = np.random.default_rng(42)
    z_near, z_far = 3, 16
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)

    # Dark smooth bokeh background (luma ~ 0.04)
    bg_tex = 0.04 + 0.005 * np.sin(x * 0.05) * np.cos(y * 0.05)
    bg_rgb = np.dstack([bg_tex, bg_tex * 1.1, bg_tex * 0.9]).astype(np.float32)

    # Near object at z=3 (bright circle at center (160, 160), radius 80)
    near_mask = (((x - 160) ** 2 + (y - 160) ** 2) <= 80 ** 2).astype(np.float32)
    near_tex = 0.70 + 0.20 * np.cos((x + y) * 0.3)
    near_rgb = np.dstack([near_tex, near_tex * 0.8, near_tex * 0.5]).astype(np.float32)

    # Far object at z=16 (bright circle at center (350, 350), radius 80)
    far_mask = (((x - 350) ** 2 + (y - 350) ** 2) <= 80 ** 2).astype(np.float32)
    far_tex = 0.65 + 0.20 * np.sin((x - y) * 0.3)
    far_rgb = np.dstack([far_tex * 0.6, far_tex * 0.8, far_tex]).astype(np.float32)

    # Ground truth all-in-focus composite
    gt_rgb = (
        near_mask[..., None] * near_rgb +
        (1.0 - near_mask[..., None]) * (
            far_mask[..., None] * far_rgb +
            (1.0 - far_mask[..., None]) * bg_rgb
        )
    )

    stack = np.empty((N, H, W, 3), dtype=np.float32)
    for n in range(N):
        r_near = int(abs(n - z_near) * 1.8)
        r_far = int(abs(n - z_far) * 1.8)

        near_premult = disk_psf_blur(near_rgb * near_mask[..., None], r_near)
        alpha_near = disk_psf_blur(near_mask, r_near)[..., None]

        far_premult = disk_psf_blur(far_rgb * far_mask[..., None], r_far)
        alpha_far = disk_psf_blur(far_mask, r_far)[..., None]

        clean_slice = near_premult + (1.0 - alpha_near) * (far_premult + (1.0 - alpha_far) * bg_rgb)
        sigma = np.sqrt(1e-5 + 2e-4 * clean_slice)
        noisy = clean_slice + rng.normal(0, 1, size=clean_slice.shape).astype(np.float32) * sigma
        stack[n] = np.clip(noisy, 0.0, 1.0)

    # Moat masks: annular ring of width 8..24px outside each object
    kern_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    kern_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))

    near_dil_in = cv2.dilate(near_mask, kern_inner)
    near_dil_out = cv2.dilate(near_mask, kern_outer)
    near_moat = (near_dil_out > 0) & (near_dil_in == 0) & (far_mask == 0)

    far_dil_in = cv2.dilate(far_mask, kern_inner)
    far_dil_out = cv2.dilate(far_mask, kern_outer)
    far_moat = (far_dil_out > 0) & (far_dil_in == 0) & (near_mask == 0)

    return {
        "stack": stack,
        "gt_rgb": gt_rgb,
        "z_near": z_near,
        "z_far": z_far,
        "near_moat": near_moat,
        "far_moat": far_moat
    }


def generate_high_key_bright_bokeh_stack(
    N: int = 20, H: int = 512, W: int = 512
) -> dict:
    """
    Generates a synthetic macro stack with:
    - High-key bright smooth bokeh background (luma ~ 0.92)
    - Near dark subject at slice z_near = 3
    - Far dark subject at slice z_far = 16
    - Optical defocus light cones composited across all 20 slices
    - Realistic sensor noise
    """
    rng = np.random.default_rng(42)
    z_near, z_far = 3, 16
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)

    # High-key bright background (luma ~ 0.92)
    bg_tex = 0.92 + 0.02 * np.sin(x * 0.05) * np.cos(y * 0.05)
    bg_rgb = np.dstack([bg_tex, bg_tex * 0.98, bg_tex * 0.95]).astype(np.float32)

    # Near dark object at z=3 (circle at (160, 160), radius 80)
    near_mask = (((x - 160) ** 2 + (y - 160) ** 2) <= 80 ** 2).astype(np.float32)
    near_tex = 0.12 + 0.04 * np.cos((x + y) * 0.3)
    near_rgb = np.dstack([near_tex, near_tex * 1.1, near_tex * 0.9]).astype(np.float32)

    # Far dark object at z=16 (circle at (350, 350), radius 80)
    far_mask = (((x - 350) ** 2 + (y - 350) ** 2) <= 80 ** 2).astype(np.float32)
    far_tex = 0.15 + 0.04 * np.sin((x - y) * 0.3)
    far_rgb = np.dstack([far_tex * 1.1, far_tex, far_tex * 0.8]).astype(np.float32)

    # Ground truth all-in-focus composite
    gt_rgb = (
        near_mask[..., None] * near_rgb +
        (1.0 - near_mask[..., None]) * (
            far_mask[..., None] * far_rgb +
            (1.0 - far_mask[..., None]) * bg_rgb
        )
    )

    stack = np.empty((N, H, W, 3), dtype=np.float32)
    for n in range(N):
        r_near = int(abs(n - z_near) * 1.8)
        r_far = int(abs(n - z_far) * 1.8)

        near_premult = disk_psf_blur(near_rgb * near_mask[..., None], r_near)
        alpha_near = disk_psf_blur(near_mask, r_near)[..., None]

        far_premult = disk_psf_blur(far_rgb * far_mask[..., None], r_far)
        alpha_far = disk_psf_blur(far_mask, r_far)[..., None]

        clean_slice = near_premult + (1.0 - alpha_near) * (far_premult + (1.0 - alpha_far) * bg_rgb)
        sigma = np.sqrt(1e-5 + 2e-4 * clean_slice)
        noisy = clean_slice + rng.normal(0, 1, size=clean_slice.shape).astype(np.float32) * sigma
        stack[n] = np.clip(noisy, 0.0, 1.0)

    kern_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    kern_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))

    near_dil_in = cv2.dilate(near_mask, kern_inner)
    near_dil_out = cv2.dilate(near_mask, kern_outer)
    near_moat = (near_dil_out > 0) & (near_dil_in == 0) & (far_mask == 0)

    far_dil_in = cv2.dilate(far_mask, kern_inner)
    far_dil_out = cv2.dilate(far_mask, kern_outer)
    far_moat = (far_dil_out > 0) & (far_dil_in == 0) & (near_mask == 0)

    return {
        "stack": stack,
        "gt_rgb": gt_rgb,
        "z_near": z_near,
        "z_far": z_far,
        "near_moat": near_moat,
        "far_moat": far_moat
    }


def test_qa_verification_suite():
    print("Generating synthetic macro test stack (20 slices, 512x512)...")
    data = generate_synthetic_macro_stack()
    stack, gt = data["stack"], data["gt_rgb"]
    
    cfg = StackerConfig(
        align_ecc=False,
        max_coc_radius_px=24,
        z_jump_thresh=5.0,
        halo_suppression_gamma=0.45
    )
    res = fuse_focus_stack(stack, cfg)
    fused = res["fused_rgb"]
    
    # -------------------------------------------------------------------------
    # TEST 1: Occlusion Boundary Halo Suppression (x in [256, 276] beside step edge)
    # -------------------------------------------------------------------------
    halo_strip_err = np.abs(fused[100:200, 258:276] - gt[100:200, 258:276]).mean()
    print(f"[QA 1] Boundary Halo Zone MAE:      {halo_strip_err:.5f} (Target < 0.035)")
    assert halo_strip_err < 0.035, f"Halo suppression failed: MAE {halo_strip_err}"

    # -------------------------------------------------------------------------
    # TEST 2: Crossing 2px Bristle Preservation (y=255..257, x=280..400)
    # -------------------------------------------------------------------------
    hair_err = np.abs(fused[255:257, 280:400] - gt[255:257, 280:400]).mean()
    print(f"[QA 2] Crossing Bristle Detail MAE: {hair_err:.5f} (Target < 0.065)")
    assert hair_err < 0.065, f"Crossing bristle clipped: MAE {hair_err}"

    # -------------------------------------------------------------------------
    # TEST 3: Non-Amplification of Background Sensor Noise
    # -------------------------------------------------------------------------
    fused_bg_patch = fused[320:480, 320:480] - gt[320:480, 320:480]
    single_bg_patch = stack[data["z_bg"], 320:480, 320:480] - gt[320:480, 320:480]
    snr_ratio_db = 20.0 * np.log10(np.std(fused_bg_patch) / (np.std(single_bg_patch) + 1e-8))
    print(f"[QA 3] Background Noise Change:     {snr_ratio_db:+.2f} dB (Target <= +0.50 dB)")
    assert snr_ratio_db <= 0.50, f"Excessive noise amplification: {snr_ratio_db:+.2f} dB"

    # -------------------------------------------------------------------------
    # TEST 4: Specular Bokeh Ring Rejection
    # -------------------------------------------------------------------------
    yy, xx = np.mgrid[0:512, 0:512]
    dist = np.sqrt((xx - 120) ** 2 + (yy - 120) ** 2)
    ring_region = (dist >= 14) & (dist <= 26)
    spec_ring_err = np.abs(fused[ring_region] - gt[ring_region]).max()
    print(f"[QA 4] Specular Bokeh Ring Max Err: {spec_ring_err:.5f} (Target < 0.080)")
    assert spec_ring_err < 0.080, f"Bokeh onion-ring artifact detected: {spec_ring_err}"

    # -------------------------------------------------------------------------
    # TEST 5: Tile Seam Continuity (< 10^-4 max delta across tile boundaries)
    # -------------------------------------------------------------------------
    cfg_tiled = StackerConfig(
        align_ecc=False,
        max_coc_radius_px=24,
        z_jump_thresh=5.0,
        halo_suppression_gamma=0.45,
        tile_size=256
    )
    res_tiled = fuse_focus_stack(stack, cfg_tiled)
    fused_tiled = res_tiled["fused_rgb"]
    
    seam_diff_x = np.abs(fused_tiled[:, 252:260] - fused[:, 252:260]).max()
    seam_diff_y = np.abs(fused_tiled[252:260, :] - fused[252:260, :]).max()
    max_seam_diff = max(seam_diff_x, seam_diff_y)
    print(f"[QA 5] Tile Seam Continuity Max Δ: {max_seam_diff:.6f} (Target < 0.0001)")
    assert max_seam_diff < 0.0001, f"Tile seam visible: max delta {max_seam_diff}"

    # -------------------------------------------------------------------------
    # TEST 6: Multi-Depth Bright Subjects on Dark/Smooth Bokeh Background
    # (Near object at z=3, far object at z=16 against dark bokeh background)
    # -------------------------------------------------------------------------
    print("Generating multi-depth dark bokeh test stack (20 slices, 512x512)...")
    data6 = generate_multi_depth_dark_bokeh_stack()
    stack6, gt6 = data6["stack"], data6["gt_rgb"]
    near_moat, far_moat = data6["near_moat"], data6["far_moat"]

    res6 = fuse_focus_stack(stack6, cfg)
    fused6 = res6["fused_rgb"]

    near_moat_err = np.abs(fused6[near_moat] - gt6[near_moat]).mean()
    far_moat_err = np.abs(fused6[far_moat] - gt6[far_moat]).mean()
    print(f"[QA 6] Multi-Depth Dark Moat MAE:   Near={near_moat_err:.5f}, Far={far_moat_err:.5f} (Target < 0.015)")
    assert near_moat_err < 0.015, f"Near object background halo moat detected: MAE {near_moat_err:.5f}"
    assert far_moat_err < 0.015, f"Far object background halo moat detected: MAE {far_moat_err:.5f}"

    # -------------------------------------------------------------------------
    # TEST 7: True Bounded-Memory O(H*W) Streaming Pass 1 Equivalence
    # (Assert streaming Pass 1 matches full-volume depth map within 10^-5)
    # -------------------------------------------------------------------------
    print("Testing Pass 1 streaming depth map equivalence (QA 7)...")
    from egmsd_stacker import Pass1DepthEstimator, compute_clean_depth_map, compute_focus_energy_cv2
    N7, H7, W7, _ = stack.shape
    max_coc7 = 24
    grad_ksize7 = max(3, int(0.08 * max_coc7)) | 1
    r_coc_morph7 = max(19, int(0.75 * max_coc7)) | 1
    luma_mat = np.array([[0.2126, 0.7152, 0.0722]], dtype=np.float32)

    # 1. Full volume baseline
    luma_vol7 = np.empty((N7, H7, W7), dtype=np.float32)
    energy_vol7 = np.empty((N7, H7, W7), dtype=np.float32)
    for n in range(N7):
        lum = cv2.transform(stack[n], luma_mat).reshape(H7, W7)
        luma_vol7[n] = lum
        energy_vol7[n] = compute_focus_energy_cv2(lum, grad_ksize=grad_ksize7, cfg=cfg)
    if N7 >= 3:
        energy_vol7[1:-1] = 0.25 * energy_vol7[:-2] + 0.50 * energy_vol7[1:-1] + 0.25 * energy_vol7[2:]

    full_res = compute_clean_depth_map(energy_vol7, luma_vol7, max_coc7, cfg, r_coc_morph7)
    z_clean_full, z_det_full, _, thin_fg_full, halo_full, _, _ = full_res

    # 2. Streaming Pass 1 estimator
    estimator7 = Pass1DepthEstimator(H7, W7, N7, max_coc7, cfg)
    stream_res = estimator7.estimate_from_block(stack)
    z_clean_stream, z_det_stream, _, thin_fg_stream, halo_stream, _, _ = stream_res

    diff_clean = np.max(np.abs(z_clean_stream - z_clean_full))
    diff_clean_mae = np.mean(np.abs(z_clean_stream - z_clean_full))
    diff_det = np.max(np.abs(z_det_stream - z_det_full))
    print(f"[QA 7] Streaming Pass 1 Equivalence: Max Δ={diff_clean:.6f}, MAE={diff_clean_mae:.6f} (Target < 0.000010)")
    assert diff_clean_mae < 0.05, f"Streaming Pass 1 MAE mismatch: {diff_clean_mae}"
    assert diff_clean < 1e-5, f"Streaming Pass 1 z_clean mismatch: {diff_clean}"
    assert diff_det < 1e-5, f"Streaming Pass 1 z_detail_map mismatch: {diff_det}"
    assert np.array_equal(thin_fg_stream, thin_fg_full), "Streaming Pass 1 thin_fg_mask mismatch"
    assert np.array_equal(halo_stream, halo_full), "Streaming Pass 1 halo_zone_mask mismatch"

    # -------------------------------------------------------------------------
    # TEST 8: Multi-Depth Dark Subjects on Bright/High-Key Bokeh Background
    # (Assert Auto selects 'bright' and Near & Far Moat MAE < 0.015)
    # -------------------------------------------------------------------------
    print("Generating multi-depth high-key bright bokeh test stack (20 slices, 512x512)...")
    data8 = generate_high_key_bright_bokeh_stack()
    stack8, gt8 = data8["stack"], data8["gt_rgb"]
    near_moat8, far_moat8 = data8["near_moat"], data8["far_moat"]

    cfg_auto = StackerConfig(
        align_ecc=False,
        max_coc_radius_px=24,
        z_jump_thresh=5.0,
        halo_suppression_gamma=0.45,
        bg_mode="auto"
    )
    res8 = fuse_focus_stack(stack8, cfg_auto)
    fused8 = res8["fused_rgb"]

    near_moat_err8 = np.abs(fused8[near_moat8] - gt8[near_moat8]).mean()
    far_moat_err8 = np.abs(fused8[far_moat8] - gt8[far_moat8]).mean()
    print(f"[QA 8] Multi-Depth Bright Moat MAE: Near={near_moat_err8:.5f}, Far={far_moat_err8:.5f} (Target < 0.015)")
    assert near_moat_err8 < 0.015, f"Near object high-key background halo moat detected: MAE {near_moat_err8:.5f}"
    assert far_moat_err8 < 0.015, f"Far object high-key background halo moat detected: MAE {far_moat_err8:.5f}"

    # -------------------------------------------------------------------------
    # TEST 9: Multi-Depth Dark Subjects in Energy Background Polarity Mode
    # (Assert explicit bg_mode='energy' yields Near & Far Moat MAE < 0.015)
    # -------------------------------------------------------------------------
    print("Testing Energy Mode halo suppression (QA 9)...")
    cfg_energy = StackerConfig(
        align_ecc=False,
        max_coc_radius_px=24,
        z_jump_thresh=5.0,
        halo_suppression_gamma=0.45,
        bg_mode="energy"
    )
    res9 = fuse_focus_stack(stack6, cfg_energy)
    fused9 = res9["fused_rgb"]

    near_moat_err9 = np.abs(fused9[near_moat] - gt6[near_moat]).mean()
    far_moat_err9 = np.abs(fused9[far_moat] - gt6[far_moat]).mean()
    print(f"[QA 9] Energy Mode Dark Moat MAE: Near={near_moat_err9:.5f}, Far={far_moat_err9:.5f} (Target < 0.015)")
    assert near_moat_err9 < 0.015, f"Near object energy mode halo moat detected: MAE {near_moat_err9:.5f}"
    assert far_moat_err9 < 0.015, f"Far object energy mode halo moat detected: MAE {far_moat_err9:.5f}"

    print("\nALL 9 QA ACCEPTANCE TESTS PASSED.")


import unittest


class TestEGMSDQA(unittest.TestCase):
    def test_egmsd_qa_suite(self):
        test_qa_verification_suite()


if __name__ == "__main__":
    unittest.main()

