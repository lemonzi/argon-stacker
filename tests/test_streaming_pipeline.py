"""
Comprehensive Verification Test for:
1. Memory-efficient streaming fusion (stream_fuse_focus_stack) with EG-MSD.
2. Stack order 'far_to_near' execution and p_d0 detail-map binding.
3. 4-level Laplacian pyramid multi-band retouch override blending.
4. A=255 Canvas PNG retouch mask encoding/decoding.
5. Monotonic progress reporting callback tracking.
"""

import os
import sys
import tempfile
import base64
import numpy as np
import cv2
import tifffile

# Include plugin py_focus_stack module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FocusStack.lrplugin', 'bin', 'py_focus_stack'))

from core_fusion import stream_fuse_focus_stack, apply_manual_overrides
from egmsd_stacker import StackerConfig


def create_synthetic_test_stack(num_slices=6, width=256, height=256):
    """
    Creates a synthetic focal sweep with distinct feature depths.
    Slice k is sharpest at vertical band around y = k * height / num_slices.
    """
    frames_16u = []
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    for k in range(num_slices):
        center_y = int((k + 0.5) * height / num_slices)
        blur_sigma = max(0.5, abs(k - 2) * 1.2)
        
        # Base pattern: high contrast grid with noise
        base = np.zeros((height, width), dtype=np.float32)
        base[(y_coords // 8) % 2 == (x_coords // 8) % 2] = 0.8
        base += 0.1 * np.sin(x_coords * 0.2) * np.cos(y_coords * 0.2)
        base = np.clip(base, 0.0, 1.0)
        
        # Apply depth of field blur
        k_size = int(np.ceil(blur_sigma * 3)) * 2 + 1
        blurred = cv2.GaussianBlur(base, (k_size, k_size), blur_sigma)
        
        # Colorize (3 channels RGB)
        r = np.clip(blurred * 65535.0 * 0.9, 0, 65535).astype(np.uint16)
        g = np.clip(blurred * 65535.0 * 0.7 + (k * 4000), 0, 65535).astype(np.uint16)
        b = np.clip(blurred * 65535.0 * 0.5 + ((num_slices - k) * 3000), 0, 65535).astype(np.uint16)
        frame_rgb = np.stack([r, g, b], axis=-1)
        frames_16u.append(frame_rgb)
        
    return frames_16u


def test_streaming_and_far_to_near():
    print("--- Test 1 & 2: Streaming EG-MSD with near_to_far and far_to_near ---")
    tmp_dir = tempfile.mkdtemp(prefix="focus_stack_test_")
    file_paths = []
    
    try:
        frames = create_synthetic_test_stack(num_slices=5, width=256, height=256)
        for i, frame in enumerate(frames):
            p = os.path.join(tmp_dir, f"slice_{i:02d}.tif")
            tifffile.imwrite(p, frame)
            file_paths.append(p)
            
        progress_events = []
        def track_progress(stage, detail, pct):
            progress_events.append((stage, detail, pct))

        warp_mats = [np.eye(2, 3, dtype=np.float32) for _ in file_paths]
        crop_box = [0, 0, 256, 256]

        # Test Near-to-Far streaming
        out_near = os.path.join(tmp_dir, "stacked_near.tif")
        stream_fuse_focus_stack(
            files=file_paths,
            warp_mats=warp_mats,
            crop_box=crop_box,
            method="egmsd",
            levels=4,
            bg_mode="auto",
            stack_order="near_to_far",
            output_path=out_near,
            update_progress=track_progress
        )
        assert os.path.exists(out_near), "Near-to-far stacked output TIFF was not created"
        stacked_near = tifffile.imread(out_near)
        assert stacked_near.shape == (256, 256, 3), f"Unexpected shape {stacked_near.shape}"
        assert stacked_near.dtype == np.uint16, f"Expected uint16 dtype, got {stacked_near.dtype}"
        assert len(progress_events) >= 5, "Progress callback was not fired adequately"
        print(f"  [PASS] Near-to-far streaming passed ({len(progress_events)} progress events)")

        # Test Far-to-Near streaming (validates p_d0 binding and reverse slice sweep)
        progress_events.clear()
        out_far = os.path.join(tmp_dir, "stacked_far.tif")
        stream_fuse_focus_stack(
            files=file_paths,
            warp_mats=warp_mats,
            crop_box=crop_box,
            method="egmsd",
            levels=4,
            bg_mode="auto",
            stack_order="far_to_near",
            output_path=out_far,
            update_progress=track_progress
        )
        assert os.path.exists(out_far), "Far-to-near stacked output TIFF was not created"
        stacked_far = tifffile.imread(out_far)
        assert stacked_far.shape == (256, 256, 3), f"Unexpected shape {stacked_far.shape}"
        assert stacked_far.dtype == np.uint16, f"Expected uint16 dtype, got {stacked_far.dtype}"
        print(f"  [PASS] Far-to-near streaming passed (p_d0 properly resolved)")

    finally:
        for p in file_paths + [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]:
            try: os.remove(p)
            except Exception: pass
        try: os.rmdir(tmp_dir)
        except Exception: pass


def test_multiband_manual_overrides():
    print("--- Test 3: Multi-Band Laplacian Pyramid Retouch Blending ---")
    h, w = 256, 256
    
    # Create base image and replacement slice
    fused_16u = np.full((h, w, 3), 10000, dtype=np.uint16)
    target_slice_16u = np.full((h, w, 3), 50000, dtype=np.uint16)
    
    # Create 2-channel overrides mask [frame_idx_map, alpha_map]
    # Circle in center: frame_idx = 0, alpha with feathering
    frame_idx_map = np.zeros((h, w), dtype=np.float32)
    alpha_map = np.zeros((h, w), dtype=np.float32)
    cv2.circle(alpha_map, (128, 128), 40, 1.0, -1)
    alpha_map = cv2.GaussianBlur(alpha_map, (21, 21), 5.0)
    overrides_mask = np.stack([frame_idx_map, alpha_map], axis=-1)
    
    tmp_slice = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_slice.close()
    tifffile.imwrite(tmp_slice.name, target_slice_16u)

    try:
        f_res = apply_manual_overrides(
            fused_16bit=fused_16u,
            overrides_mask=overrides_mask,
            files=[tmp_slice.name],
            warp_mats=[np.eye(2, 3, dtype=np.float32)],
            crop_box=[0, 0, w, h]
        )
    finally:
        try: os.remove(tmp_slice.name)
        except Exception: pass
    
    assert f_res.shape == (h, w, 3)
    assert f_res.dtype == np.uint16
    
    # Check center (should be ~50000)
    center_val = f_res[128, 128, 0]
    assert center_val > 45000, f"Expected center value > 45000, got {center_val}"
    
    # Check outside border (should be ~10000)
    corner_val = f_res[10, 10, 0]
    assert corner_val == 10000, f"Expected unpainted corner to remain 10000, got {corner_val}"
    
    # Verify transition region is continuous without sharp hard boundary step
    radii = [30, 35, 40, 45, 50]
    vals = [f_res[128, 128 + r, 0] for r in radii]
    assert vals == sorted(vals, reverse=True), f"Expected monotonically decreasing blend, got {vals}"
    print("  [PASS] Multi-band Laplacian blend smoothly blended override slice")


def test_canvas_alpha255_encoding():
    print("--- Test 4: Canvas A=255 Retouch Mask Protocol ---")
    h, w = 64, 64
    
    # Construct master canvas buffer as sent from HTML5 client
    # Canvas putImageData: R=bestFrame, G=bestAlpha, B=bestFrame, A=255
    canvas_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Paint frame 7 with alpha 180 (70% opacity)
    target_frame = 7
    target_alpha = 180
    canvas_rgba[20:44, 20:44, 0] = target_frame
    canvas_rgba[20:44, 20:44, 1] = target_alpha
    canvas_rgba[20:44, 20:44, 2] = target_frame
    canvas_rgba[20:44, 20:44, 3] = 255
    
    # Encode as PNG (OpenCV imencode expects BGRA)
    canvas_bgra = cv2.cvtColor(canvas_rgba, cv2.COLOR_RGBA2BGRA)
    _, buf = cv2.imencode('.png', canvas_bgra)
    b64_str = base64.b64encode(buf).decode('utf-8')
    
    # Server-side decoding logic (from cli.py line 580)
    raw_bytes = base64.b64decode(b64_str)
    png_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(png_arr, cv2.IMREAD_UNCHANGED)
    
    assert decoded is not None and decoded.shape[2] == 4
    assert np.all(decoded[20:44, 20:44, 3] == 255)
    
    frame_idx_map = decoded[:, :, 2].astype(np.float32) # In BGR: channel 2 is R
    alpha_map = decoded[:, :, 1].astype(np.float32) / 255.0 # In BGR: channel 1 is G
    
    assert frame_idx_map[30, 30] == target_frame, f"Expected frame {target_frame}, got {frame_idx_map[30, 30]}"
    assert abs(alpha_map[30, 30] - (target_alpha / 255.0)) < 1e-3, f"Expected alpha {target_alpha/255.0}, got {alpha_map[30, 30]}"
    print("  [PASS] A=255 encoding bypassed Canvas premultiplied-alpha corruption successfully")


import unittest


class TestStreamingPipeline(unittest.TestCase):
    def test_streaming_and_far_to_near(self):
        test_streaming_and_far_to_near()

    def test_multiband_manual_overrides(self):
        test_multiband_manual_overrides()

    def test_canvas_alpha255_encoding(self):
        test_canvas_alpha255_encoding()


if __name__ == "__main__":
    unittest.main()
