#!/usr/bin/env python3
"""
Argon Stacker (16-Bit Multi-Channel Engine)
CLI & Stacking Orchestrator with Streaming Architecture
Supports both Lightroom Classic plugin execution and standalone desktop usage.
"""
import sys
import os

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import re
import argparse
import base64
import io
import time
import gc
import subprocess
import tempfile
import cv2
import numpy as np
import tifffile
from PIL import Image, ImageCms

from tiff_io import read_image_16bit, write_image, read_tiff_16bit, write_tiff_16bit
from core_align import compute_alignment_matrix, read_downsampled_gray
from core_fusion import (
    fuse_focus_stack, stream_fuse_focus_stack, fuse_laplacian_pyramids, stream_fuse_laplacian_pyramids,
    compute_contrast_map, compute_robust_focus_statistics, compute_continuity_factor
)
from egmsd_stacker import compute_focus_energy_cv2, StackerConfig
from server_ui import run_interactive_server

SUPPORTED_EXTS = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.dng', '.arw', '.cr2', '.cr3', '.nef', '.orf', '.rw2')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_photo_capture_timestamp(filepath):
    """
    Extracts high-precision capture timestamp (DateTimeOriginal/DateTime + subseconds)
    from EXIF metadata across JPEG, TIFF, DNG, and RAW formats.
    Returns sortable string (e.g. '2026:10:11 12:39:20.164') or None.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import IFD
        with Image.open(filepath) as im:
            ex = im.getexif()
            if ex:
                ex_ifd = ex.get_ifd(IFD.Exif)
                dto = ex_ifd.get(36867) or ex_ifd.get(36868) or ex.get(306)
                if dto:
                    sub = str(ex_ifd.get(37521) or ex_ifd.get(37522) or ex_ifd.get(37520) or '0').strip('\x00')
                    return f"{str(dto).strip(chr(0))}.{sub.zfill(3)}"
    except Exception:
        pass

    try:
        import tifffile
        with tifffile.TiffFile(filepath) as tf:
            for page in tf.pages:
                for tag in page.tags:
                    if tag.code in (36867, 36868, 306) or tag.name in ('DateTimeOriginal', 'DateTimeDigitized', 'DateTime'):
                        val = str(tag.value).strip('\x00')
                        if val:
                            return val
    except Exception:
        pass

    return None

def check_sort_order_discrepancy(files):
    """
    Compares filename natural sort order vs EXIF photo taken timestamp sort order.
    Returns:
      discrepancy (bool): True if EXIF timestamps exist and disagree with filename sort.
      files_by_name (list): Files sorted by filename natural sort.
      files_by_exif (list): Files sorted by EXIF timestamp.
      details (dict): Structured conflict metadata for UI and logs.
    """
    if len(files) < 2:
        return False, list(files), list(files), {}

    files_by_name = sorted(files, key=natural_sort_key)
    timestamps = {}
    valid_count = 0
    for f in files_by_name:
        ts = get_photo_capture_timestamp(f)
        timestamps[f] = ts
        if ts is not None:
            valid_count += 1

    if valid_count < 2:
        return False, files_by_name, files_by_name, {}

    files_by_exif = sorted(
        files_by_name,
        key=lambda f: (
            timestamps[f] is None,
            timestamps[f] if timestamps[f] is not None else "",
            natural_sort_key(f)
        )
    )

    disagree = (files_by_name != files_by_exif)
    details = {
        "disagree": disagree,
        "count": len(files),
        "files_by_name": [
            {"path": f, "basename": os.path.basename(f), "timestamp": timestamps.get(f) or "No EXIF timestamp"}
            for f in files_by_name
        ],
        "files_by_exif": [
            {"path": f, "basename": os.path.basename(f), "timestamp": timestamps.get(f) or "No EXIF timestamp"}
            for f in files_by_exif
        ]
    }
    return disagree, files_by_name, files_by_exif, details

def discover_input_files(file_args):
    """
    Expands directories, wildcards, or handles lists of file paths.
    """
    if not file_args:
        return []

    if len(file_args) == 1 and os.path.isdir(file_args[0]):
        folder = os.path.abspath(file_args[0])
        candidates = []
        for fname in os.listdir(folder):
            if fname.startswith('.'):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS and not ('_stacked.' in fname):
                candidates.append(os.path.join(folder, fname))
        candidates.sort(key=natural_sort_key)
        return candidates

    matched = []
    for arg in file_args:
        if os.path.isdir(arg):
            for fname in os.listdir(arg):
                if not fname.startswith('.') and os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS and not ('_stacked.' in fname):
                    matched.append(os.path.join(arg, fname))
        elif os.path.isfile(arg):
            ext = os.path.splitext(arg)[1].lower()
            if ext in SUPPORTED_EXTS and not ('_stacked.' in os.path.basename(arg)):
                matched.append(arg)

    matched.sort(key=natural_sort_key)
    return matched

def native_choose_files_or_folder():
    """
    Presents a native macOS open dialog allowing the user to select either
    a folder containing bracketed macro photos, or multiple individual image files.
    """
    if sys.platform == 'darwin':
        try:
            from AppKit import NSOpenPanel, NSApplication, NSModalResponseOK
            app = NSApplication.sharedApplication()
            app.activateIgnoringOtherApps_(True)
            panel = NSOpenPanel.openPanel()
            panel.setTitle_("Argon Stacker — Select Photos or Folder")
            panel.setMessage_("Select a photoshoot folder, or choose individual bracketed macro photos:")
            panel.setCanChooseFiles_(True)
            panel.setCanChooseDirectories_(True)
            panel.setAllowsMultipleSelection_(True)
            panel.setResolvesAliases_(True)
            if panel.runModal() == NSModalResponseOK:
                urls = panel.URLs()
                paths = [str(u.path()) for u in urls if u.path()]
                if paths:
                    return paths
        except Exception as e:
            sys.stderr.write(f"Notice: Cocoa open panel skipped ({e}), falling back to AppleScript...\n")

        # Fallback to AppleScript choose folder
        try:
            cmd = ['osascript', '-e', 'tell application "System Events" to activate\nset chosen to (choose folder with prompt "Select folder containing bracketed macro photos:")\nreturn POSIX path of chosen']
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0 and p.stdout.strip():
                return [p.stdout.strip()]
        except Exception:
            pass

    return []

def align_frames(files, crop=True, update_status=None, is_cancelled=None):
    """
    PASS 1: Streaming Alignment Matrix Computation (RAM < 200MB)
    Computes subpixel ECC affine alignment matrices against the center reference frame.
    """
    ref_idx = len(files) // 2
    if update_status:
        update_status("Subpixel Alignment", f"Reading reference frame {ref_idx + 1} of {len(files)}...", 0.05)

    ref_gray_small, full_w, full_h, ref_scale, icc_profile = read_downsampled_gray(files[ref_idx], max_dim=2048)
    if icc_profile is None:
        _, _, _, _, icc_profile = read_downsampled_gray(files[0], max_dim=2048)

    warp_mats = [None] * len(files)
    warp_mats[ref_idx] = np.eye(2, 3, dtype=np.float32)

    mask_w = int(full_w * ref_scale)
    mask_h = int(full_h * ref_scale)
    combined_mask = np.ones((mask_h, mask_w), dtype=np.uint8)

    for i, fpath in enumerate(files):
        if is_cancelled and is_cancelled():
            return None

        if i == ref_idx:
            if update_status:
                prog = 0.05 + 0.50 * ((i + 1) / float(len(files)))
                update_status("Subpixel Alignment", f"Aligned {os.path.basename(fpath)} ({i + 1}/{len(files)})", prog)
            continue

        if update_status:
            prog = 0.05 + 0.50 * ((i + 1) / float(len(files)))
            update_status("Subpixel Alignment", f"Aligning {os.path.basename(fpath)} ({i + 1}/{len(files)})...", prog)

        src_gray_small, _, _, src_scale, _ = read_downsampled_gray(fpath, max_dim=2048)
        warp_mat = compute_alignment_matrix(ref_gray_small, src_gray_small, max_ecc_res=2048)
        del src_gray_small
        gc.collect()

        warp_mat[0, 2] /= ref_scale
        warp_mat[1, 2] /= ref_scale
        warp_mats[i] = warp_mat

        mat_small = warp_mat.copy()
        mat_small[0, 2] *= ref_scale
        mat_small[1, 2] *= ref_scale
        mask = np.ones((mask_h, mask_w), dtype=np.uint8)
        warped_mask = cv2.warpAffine(
            mask, mat_small, (mask_w, mask_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        combined_mask = cv2.bitwise_and(combined_mask, warped_mask)
        del mask, warped_mask

    del ref_gray_small
    gc.collect()

    sx, sy, sbw, sbh = cv2.boundingRect(combined_mask)
    del combined_mask
    gc.collect()

    if crop:
        bx = max(0, int(round(sx / ref_scale)))
        by = max(0, int(round(sy / ref_scale)))
        bw = min(full_w - bx, int(round(sbw / ref_scale)))
        bh = min(full_h - by, int(round(sbh / ref_scale)))
    else:
        bx, by, bw, bh = 0, 0, full_w, full_h

    crop_box = (bx, by, bw, bh)
    return warp_mats, crop_box, (full_w, full_h), icc_profile

def build_preview_engine(files, warp_mats, crop_box, full_dim, icc_profile, output_path=None, update_status=None, is_cancelled=None, default_output_path=None):
    """
    PASS 2 (Interactive): Extracts preview frames, initializes color management,
    and constructs get_preview, get_frame, and render_final callbacks.
    """
    default_output_path = default_output_path or output_path
    if not default_output_path and len(files) > 0:
        first_dir = os.path.dirname(os.path.abspath(files[0]))
        first_stem = os.path.splitext(os.path.basename(files[0]))[0]
        first_ext = os.path.splitext(files[0])[1].lower()
        target_ext = '.jpg' if first_ext in ('.jpg', '.jpeg') else '.tif'
        default_output_path = os.path.join(first_dir, f"{first_stem}_stacked{target_ext}")

    full_w, full_h = full_dim
    bx, by, bw, bh = crop_box

    if update_status:
        update_status("Preparing Previews", f"Extracting 16-bit multi-scale preview frames...", 0.60)

    # Low-res preview scale (max 960px)
    preview_scale = min(1.0, 960.0 / float(max(bh, bw)))
    pw = int(bw * preview_scale)
    ph = int(bh * preview_scale)

    preview_frames = []
    for idx, fpath in enumerate(files):
        if is_cancelled and is_cancelled():
            return None, None, None

        if update_status:
            prog = 0.60 + 0.25 * ((idx + 1) / float(len(files)))
            update_status("Preparing Previews", f"Downscaling {os.path.basename(fpath)} ({idx + 1}/{len(files)})...", prog)

        img, _ = read_tiff_16bit(fpath)
        warp_mat = warp_mats[idx]
        is_identity = (
            np.abs(warp_mat[0, 0] - 1.0) < 1e-5 and
            np.abs(warp_mat[1, 1] - 1.0) < 1e-5 and
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
        preview_frames.append(cv2.resize(cropped, (pw, ph), interpolation=cv2.INTER_AREA))
        del cropped
        gc.collect()

    # Precompute statistical focus model across preview frames for zero-lag Focus Peaking
    preview_energies = []
    for pf in preview_frames:
        pf_f = pf.astype(np.float32) / 65535.0
        g = 0.2126 * pf_f[:, :, 0] + 0.7152 * pf_f[:, :, 1] + 0.0722 * pf_f[:, :, 2]
        c = compute_focus_energy_cv2(g, grad_ksize=3, cfg=StackerConfig())
        preview_energies.append(c)
    preview_energy_stack = np.stack(preview_energies, axis=0)
    del preview_energies
    preview_min_e = np.min(preview_energy_stack, axis=0)

    preview_z_global, preview_baseline, preview_sigma_reg, preview_k_star = compute_robust_focus_statistics(
        preview_energy_stack, spatial_smooth_kernel=5, tau_continuity=0.15, continuity_check=True
    )
    preview_alpha_cont = compute_continuity_factor(preview_energy_stack, preview_baseline, preview_k_star, tau=0.15)
    preview_e_peak = np.take_along_axis(preview_energy_stack, preview_k_star[np.newaxis, :, :], axis=0)[0]

    tif_handles = []
    memmap_pages = []
    temp_memmap_files = []

    for f in files:
        if f.lower().endswith(('.tif', '.tiff')):
            try:
                t = tifffile.TiffFile(f)
                tif_handles.append(t)
                try:
                    memmap_pages.append(t.pages[0].asarray(out='memmap'))
                except Exception:
                    memmap_pages.append(t.pages[0].asarray())
                continue
            except Exception:
                pass

        # For JPEG/PNG, create virtual memory backed disk memmap
        data, _ = read_tiff_16bit(f)
        temp_file = tempfile.NamedTemporaryFile(prefix='fs_frame_', suffix='.dat', delete=False)
        temp_memmap_files.append(temp_file.name)
        temp_file.close()

        mm = np.memmap(temp_file.name, dtype='uint16', mode='w+', shape=data.shape)
        mm[:] = data
        mm.flush()
        del data
        mm_ro = np.memmap(temp_file.name, dtype='uint16', mode='r', shape=mm.shape)
        memmap_pages.append(mm_ro)
        del mm
        gc.collect()

    def extract_aligned_crop(m, warp_mat, target_x, target_y, target_w, target_h):
        full_h, full_w = m.shape[:2]
        is_identity = (
            np.abs(warp_mat[0, 0] - 1.0) < 1e-5 and
            np.abs(warp_mat[1, 1] - 1.0) < 1e-5 and
            np.abs(warp_mat[0, 2]) < 1e-4 and
            np.abs(warp_mat[1, 2]) < 1e-4
        )
        if is_identity:
            return m[target_y : target_y + target_h, target_x : target_x + target_w].copy()

        M_inv = cv2.invertAffineTransform(warp_mat)
        corners_target = np.array([
            [target_x, target_y],
            [target_x + target_w, target_y],
            [target_x + target_w, target_y + target_h],
            [target_x, target_y + target_h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        corners_source = cv2.transform(corners_target, M_inv).reshape(-1, 2)

        pad = 20
        sx0 = max(0, int(np.floor(corners_source[:, 0].min())) - pad)
        sy0 = max(0, int(np.floor(corners_source[:, 1].min())) - pad)
        sx1 = min(full_w, int(np.ceil(corners_source[:, 0].max())) + pad)
        sy1 = min(full_h, int(np.ceil(corners_source[:, 1].max())) + pad)

        patch = m[sy0:sy1, sx0:sx1]

        M_patch = warp_mat.copy()
        M_patch[0, 2] = warp_mat[0, 2] - target_x + warp_mat[0, 0] * sx0 + warp_mat[0, 1] * sy0
        M_patch[1, 2] = warp_mat[1, 2] - target_y + warp_mat[1, 0] * sx0 + warp_mat[1, 1] * sy0

        return cv2.warpAffine(patch, M_patch, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    # Color Management: ProPhoto RGB / wide-gamut -> sRGB for display
    preview_color_transform = None
    try:
        dst_prof = ImageCms.createProfile('sRGB')
        if icc_profile is not None:
            try:
                src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                preview_color_transform = ImageCms.buildTransform(
                    src_prof, dst_prof, 'RGB', 'RGB', renderingIntent=ImageCms.Intent.PERCEPTUAL
                )
            except Exception:
                pass
        if preview_color_transform is None and len(files) > 0 and not files[0].lower().endswith(('.jpg', '.jpeg', '.png')):
            for p in ['/System/Library/ColorSync/Profiles/ROMM RGB.icc', '/System/Library/ColorSync/Profiles/AdobeRGB1998.icc']:
                if os.path.exists(p):
                    try:
                        src_prof = ImageCms.ImageCmsProfile(p)
                        preview_color_transform = ImageCms.buildTransform(
                            src_prof, dst_prof, 'RGB', 'RGB', renderingIntent=ImageCms.Intent.PERCEPTUAL
                        )
                        break
                    except Exception:
                        pass
    except Exception:
        pass

    def parse_zoom(mode_str):
        if mode_str == "fit":
            return None
        if mode_str in ("roi", "100%", "100", "1.0"):
            return 1.0
        if mode_str in ("50%", "50", "0.5"):
            return 0.5
        if mode_str in ("200%", "200", "2.0"):
            return 2.0
        try:
            val = float(str(mode_str).replace("%", ""))
            if val > 10.0:
                val = val / 100.0
            return max(0.1, min(10.0, val))
        except Exception:
            return 1.0

    _preview_cache = {}

    def get_preview(
        method="egmsd", levels=6, kernel=5, exponent=5.0, noise_threshold=1.0, depth_smooth=7,
        z_jump=5.0, halo_gamma=0.45, anisotropy=0.5, specular_thresh=0.94,
        bg_mode="auto",
        mode="fit", cx=0.5, cy=0.5,
        stack_order="near_to_far",
        vw=800, vh=800
    ):
        cx = min(1.0, max(0.0, float(cx)))
        cy = min(1.0, max(0.0, float(cy)))

        cache_key = (
            method, round(float(z_jump), 2), round(float(halo_gamma), 2), round(float(anisotropy), 2),
            round(float(specular_thresh), 3), round(float(noise_threshold), 4),
            str(bg_mode), str(stack_order),
            "fit" if mode == "fit" else (str(mode), round(cx, 3), round(cy, 3), int(vw), int(vh))
        )

        if mode == "fit":
            crop_rect = [0.0, 0.0, 1.0, 1.0]
            m_label = "EG-MSD Multi-Scale"
            info = f"{len(files)} frames | {m_label} Preview ({pw}x{ph}) | 16-Bit Stack: {bw}x{bh}"
            target_frames = preview_frames
        else:
            zoom_factor = parse_zoom(mode)
            crop_size_w = min(int(round(float(vw) / zoom_factor)), bw)
            crop_size_h = min(int(round(float(vh) / zoom_factor)), bh)
            off_x = int(round(cx * (bw - crop_size_w)))
            off_y = int(round(cy * (bh - crop_size_h)))
            target_x = bx + off_x
            target_y = by + off_y
            crop_rect = [off_x / float(bw), off_y / float(bh), crop_size_w / float(bw), crop_size_h / float(bh)]

            disp_w = int(round(crop_size_w * zoom_factor))
            disp_h = int(round(crop_size_h * zoom_factor))

            pct_x = int(round(cx * 100))
            pct_y = int(round(cy * 100))
            m_label = "EG-MSD Multi-Scale"
            zoom_pct = int(round(zoom_factor * 100))
            info = f"{len(files)} frames | {m_label} {zoom_pct}% Zoom ({pct_x}%, {pct_y}%) | FOV: {crop_size_w}x{crop_size_h}"

            if cache_key in _preview_cache:
                return _preview_cache[cache_key]

            target_frames = [
                extract_aligned_crop(m, w, target_x, target_y, crop_size_w, crop_size_h)
                for m, w in zip(memmap_pages, warp_mats)
            ]
            if (disp_w, disp_h) != (crop_size_w, crop_size_h):
                interp = cv2.INTER_AREA if zoom_factor < 1.0 else cv2.INTER_LINEAR
                target_frames = [
                    cv2.resize(f, (disp_w, disp_h), interpolation=interp)
                    for f in target_frames
                ]

        if cache_key in _preview_cache:
            return _preview_cache[cache_key]

        zoom_factor = parse_zoom(mode) if mode != "fit" else 1.0
        max_coc_preview = int(round(0.007 * max(bw, bh) * zoom_factor)) if mode != "fit" else None

        frames_for_fuse = target_frames

        fuse_res = fuse_focus_stack(
            frames_for_fuse,
            method="egmsd",
            levels=min(levels, 4),
            contrast_kernel=kernel,
            exponent=exponent,
            noise_threshold=noise_threshold,
            depth_smooth=depth_smooth,
            z_jump=z_jump,
            halo_gamma=halo_gamma,
            anisotropy=anisotropy,
            specular_thresh=specular_thresh,
            max_coc_radius_px=max_coc_preview,
            bg_mode=bg_mode,
            stack_order=stack_order,
            return_depth_map=True
        )

        if len(fuse_res) == 4:
            fused, depth_8u, saliency_8u, eff_bg = fuse_res
        elif len(fuse_res) == 3:
            fused, depth_8u, saliency_8u = fuse_res
            eff_bg = bg_mode
        else:
            fused, depth_8u = fuse_res
            saliency_8u = depth_8u
            eff_bg = bg_mode

        fused_8u = (fused >> 8).astype(np.uint8)
        if preview_color_transform is not None:
            try:
                pil_img = Image.fromarray(fused_8u, 'RGB')
                transformed_img = ImageCms.applyTransform(pil_img, preview_color_transform)
                fused_8u = np.asarray(transformed_img)
            except Exception:
                pass

        fused_bgr = cv2.cvtColor(fused_8u, cv2.COLOR_RGB2BGR)
        _, buf_img = cv2.imencode('.jpg', fused_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_b64 = base64.b64encode(buf_img).decode('utf-8')

        _, buf_depth = cv2.imencode('.jpg', depth_8u, [cv2.IMWRITE_JPEG_QUALITY, 85])
        depth_b64 = base64.b64encode(buf_depth).decode('utf-8')

        _, buf_sal = cv2.imencode('.jpg', saliency_8u, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saliency_b64 = base64.b64encode(buf_sal).decode('utf-8')

        result = (img_b64, depth_b64, saliency_b64, info, crop_rect, eff_bg)
        _preview_cache[cache_key] = result
        return result

    def get_frame(index, mode="fit", cx=0.5, cy=0.5, peaking=False, vw=800, vh=800):
        index = max(0, min(len(files) - 1, int(index)))
        if mode == "fit":
            f_img = preview_frames[index]
        else:
            zoom_factor = parse_zoom(mode)
            crop_size_w = min(int(round(float(vw) / zoom_factor)), bw)
            crop_size_h = min(int(round(float(vh) / zoom_factor)), bh)
            off_x = int(round(cx * (bw - crop_size_w)))
            off_y = int(round(cy * (bh - crop_size_h)))
            target_x = bx + off_x
            target_y = by + off_y
            disp_w = int(round(crop_size_w * zoom_factor))
            disp_h = int(round(crop_size_h * zoom_factor))
            f_img = extract_aligned_crop(memmap_pages[index], warp_mats[index], target_x, target_y, crop_size_w, crop_size_h)
            if (disp_w, disp_h) != (f_img.shape[1], f_img.shape[0]):
                interp = cv2.INTER_AREA if zoom_factor < 1.0 else cv2.INTER_LINEAR
                f_img = cv2.resize(f_img, (disp_w, disp_h), interpolation=interp)

        if peaking:
            f_float = f_img.astype(np.float32) / 65535.0
            gray = 0.2126 * f_float[:, :, 0] + 0.7152 * f_float[:, :, 1] + 0.0722 * f_float[:, :, 2]

            if mode == "fit":
                e_i = preview_energy_stack[index]
                p99 = max(1e-4, float(np.percentile(preview_e_peak, 99)))
                elev_i = e_i - preview_min_e
                peak_mask = (
                    (e_i > max(3e-4, p99 * 0.035)) &
                    (elev_i > max(1.5e-4, p99 * 0.015)) &
                    (np.abs(index - preview_k_star) <= 1) &
                    (e_i >= 0.70 * preview_e_peak)
                )
            else:
                rx = int(round((off_x / float(bw)) * pw))
                ry = int(round((off_y / float(bh)) * ph))
                rw = max(1, int(round((crop_size_w / float(bw)) * pw)))
                rh = max(1, int(round((crop_size_h / float(bh)) * ph)))

                roi_k = cv2.resize(preview_k_star[ry : ry + rh, rx : rx + rw], (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                roi_e_peak = cv2.resize(preview_e_peak[ry : ry + rh, rx : rx + rw], (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                roi_min_e = cv2.resize(preview_min_e[ry : ry + rh, rx : rx + rw], (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                c = compute_focus_energy_cv2(gray, grad_ksize=3, cfg=StackerConfig())
                p99 = max(1e-4, float(np.percentile(roi_e_peak, 99)))
                elev_c = c - roi_min_e
                peak_mask = (
                    (c > max(3e-4, p99 * 0.035)) &
                    (elev_c > max(1.5e-4, p99 * 0.015)) &
                    (np.abs(index - roi_k) <= 1) &
                    (c >= 0.70 * roi_e_peak)
                )

            # Grayscale base + neon green focus peaking
            gray_8u = (gray * 255.0).astype(np.uint8)
            peak_bgr = cv2.cvtColor(gray_8u, cv2.COLOR_GRAY2BGR)
            peak_bgr[peak_mask] = [30, 240, 50]  # Bright neon green
            _, buf = cv2.imencode('.jpg', peak_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return base64.b64encode(buf).decode('utf-8')

        f_8u = (f_img >> 8).astype(np.uint8)
        if preview_color_transform is not None:
            try:
                pil_img = Image.fromarray(f_8u, 'RGB')
                transformed_img = ImageCms.applyTransform(pil_img, preview_color_transform)
                f_8u = np.asarray(transformed_img)
            except Exception:
                pass
        f_bgr = cv2.cvtColor(f_8u, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', f_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode('utf-8')

    def render_final(
        method="egmsd", levels=6, kernel=5, exponent=5.0, noise_threshold=1.0, depth_smooth=7,
        z_jump=5.0, halo_gamma=0.45, anisotropy=0.5, specular_thresh=0.94,
        bg_mode="auto",
        overrides=None, output_path=None,
        stack_order="near_to_far",
        update_progress=None
    ):
        for t in list(tif_handles):
            try: t.close()
            except Exception: pass
        tif_handles.clear()

        for tmp_p in list(temp_memmap_files):
            try:
                if os.path.exists(tmp_p):
                    os.remove(tmp_p)
            except Exception: pass
        temp_memmap_files.clear()
        gc.collect()

        target_output = output_path or default_output_path
        if not target_output and len(files) > 0:
            first_dir = os.path.dirname(os.path.abspath(files[0]))
            first_stem = os.path.splitext(os.path.basename(files[0]))[0]
            first_ext = os.path.splitext(files[0])[1].lower()
            target_ext = '.jpg' if first_ext in ('.jpg', '.jpeg') else '.tif'
            target_output = os.path.join(first_dir, f"{first_stem}_stacked{target_ext}")

        overrides_mask = None
        if overrides:
            try:
                if "," in overrides:
                    overrides = overrides.split(",", 1)[1]
                raw_bytes = base64.b64decode(overrides)
                png_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(png_arr, cv2.IMREAD_UNCHANGED)
                if decoded is not None and len(decoded.shape) == 3 and decoded.shape[2] >= 2:
                    if decoded.shape[2] >= 4 and np.all(decoded[:, :, 3] == 255):
                        frame_idx_map = decoded[:, :, 2].astype(np.float32)
                        alpha_map = decoded[:, :, 1].astype(np.float32) / 255.0
                    else:
                        frame_idx_map = decoded[:, :, 2].astype(np.float32)
                        alpha_map = (decoded[:, :, 3].astype(np.float32) / 255.0)
                    overrides_mask = np.stack([frame_idx_map, alpha_map], axis=-1)
                    print(f"Decoded retouching mask: {overrides_mask.shape} (active paint pixels: {np.sum(alpha_map > 0.01)})")
            except Exception as e:
                print(f"Warning: Could not decode manual overrides ({e})")

        m_name = f"Edge-Guided Multi-Scale Decomposition (EG-MSD, bg_mode={bg_mode}, z_jump={z_jump}, halo_gamma={halo_gamma}, anisotropy={anisotropy}, stack_order={stack_order})"
        fmt_label = "JPEG" if target_output.lower().endswith(('.jpg', '.jpeg')) else "16-Bit TIFF"
        print(f"\nRendering final full-resolution {fmt_label} stack via {m_name} (noise_threshold={noise_threshold})...")
        t_fuse = time.time()

        stream_fuse_focus_stack(
            files=files,
            warp_mats=warp_mats,
            crop_box=crop_box,
            method="egmsd",
            levels=levels,
            contrast_kernel=kernel,
            exponent=exponent,
            noise_threshold=noise_threshold,
            depth_smooth=depth_smooth,
            z_jump=z_jump,
            halo_gamma=halo_gamma,
            anisotropy=anisotropy,
            specular_thresh=specular_thresh,
            bg_mode=bg_mode,
            stack_order=stack_order,
            icc_profile=icc_profile,
            output_path=target_output,
            overrides_mask=overrides_mask,
            update_progress=update_progress
        )
        print(f"Streaming fusion complete in {time.time() - t_fuse:.2f}s")
        return target_output

    if update_status:
        update_status("Pre-rendering", "Synthesizing live EG-MSD preview & depth map...", 0.90)

    # Pre-render initial fit and 100% center crop for instant 0ms display
    get_preview(method="egmsd", z_jump=5.0, halo_gamma=0.45, anisotropy=0.5, specular_thresh=0.94, noise_threshold=1.0, mode="fit")
    get_preview(method="egmsd", z_jump=5.0, halo_gamma=0.45, anisotropy=0.5, specular_thresh=0.94, noise_threshold=1.0, mode="100%", cx=0.5, cy=0.5)

    if update_status:
        update_status("Ready", f"All {len(files)} frames aligned and pre-rendered", 1.0)

    return get_preview, get_frame, render_final

def build_worker_for_files(files, requested_output=None, crop=True):
    first_ext = os.path.splitext(files[0])[1].lower()
    is_jpeg_source = first_ext in ('.jpg', '.jpeg')
    target_ext = '.jpg' if is_jpeg_source else '.tif'

    if not requested_output:
        first_dir = os.path.dirname(os.path.abspath(files[0]))
        first_stem = os.path.splitext(os.path.basename(files[0]))[0]
        output_path = os.path.join(first_dir, f"{first_stem}_stacked{target_ext}")
    else:
        base, cur_ext = os.path.splitext(requested_output)
        if is_jpeg_source and cur_ext.lower() not in ('.jpg', '.jpeg'):
            output_path = f"{base}{target_ext}"
        elif (not is_jpeg_source) and cur_ext.lower() in ('.jpg', '.jpeg'):
            output_path = f"{base}.tif"
        else:
            output_path = requested_output

    def interactive_worker(update_status, is_cancelled):
        align_res = align_frames(files, crop=crop, update_status=update_status, is_cancelled=is_cancelled)
        if align_res is None or (is_cancelled and is_cancelled()):
            return None, None
        warp_mats, crop_box, full_dim, icc_profile = align_res

        get_prev, get_frm, render_fin = build_preview_engine(
            files, warp_mats, crop_box, full_dim, icc_profile,
            output_path=output_path,
            update_status=update_status,
            is_cancelled=is_cancelled
        )
        return (get_prev, get_frm, len(files), [os.path.basename(f) for f in files]), render_fin

    return interactive_worker, output_path

def main():
    parser = argparse.ArgumentParser(description="Argon Stacker — 16-Bit Multi-Channel Engine")
    parser.add_argument("files", nargs="*", help="Input image files or directory containing bracketed photos")
    parser.add_argument("--output", "-o", default=None, help="Output stacked 16-bit TIFF or JPEG path")
    parser.add_argument("--interactive", action="store_true", default=None, help="Launch interactive native desktop GUI")
    parser.add_argument("--batch", action="store_true", help="Run batch headless fusion without opening GUI")
    parser.add_argument("--method", choices=["egmsd"], default="egmsd", help="Fusion method: egmsd (Edge-Guided Multi-Scale Decomposition)")
    parser.add_argument("--stack-order", choices=["near_to_far", "far_to_near"], default="near_to_far", help="Stack slice direction: near_to_far (standard) or far_to_near (reverse)")
    parser.add_argument("--z-jump", type=float, default=5.0, help="Occlusion Z-jump threshold (default: 5.0)")
    parser.add_argument("--halo-gamma", type=float, default=0.45, help="Asymmetric halo suppression gamma (default: 0.45)")
    parser.add_argument("--anisotropy", type=float, default=0.5, help="Crossing bristle anisotropy boost (default: 0.5)")
    parser.add_argument("--specular-thresh", type=float, default=0.94, help="Specular bokeh ring threshold (default: 0.94)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Bokeh noise floor multiplier (default: 1.0)")
    parser.add_argument("--bg-mode", choices=["auto", "dark", "bright", "energy"], default="auto", help="Background polarity mode: auto, dark, bright, or energy (default: auto)")
    parser.add_argument("--crop", action="store_true", default=True, help="Crop output to valid overlapping area (removes warped edges)")
    parser.add_argument("--no-crop", dest="crop", action="store_false")
    parser.add_argument("--sort-by", choices=["auto", "name", "exif"], default="auto", help="Frame ordering mode: auto (prompt if filename and EXIF disagree), name (sort by filename), exif (sort by capture timestamp)")
    parser.add_argument("--lightroom", action="store_true", default=False, help="Lightroom Classic plugin export mode")

    args = parser.parse_args()

    files = discover_input_files(args.files)

    # Determine interactive mode: default to True unless --batch is explicitly requested
    is_interactive = True if not args.batch else False
    if args.interactive is False:
        is_interactive = False

    # Standalone Launch with No CLI Files: Prompt directly with native window picker
    if len(files) < 2:
        if not is_interactive:
            print("Error: Must provide at least 2 photos to stack in headless batch mode.")
            print("Usage: python3 -m py_focus_stack.cli /path/to/photos_folder/ [or list of files]")
            sys.exit(1)

        print("============================================================")
        print("Argon Stacker — Interactive Workspace")
        print("No files provided via CLI. Opening native window picker...")
        print("============================================================")

        chosen = native_choose_files_or_folder()
        if not chosen:
            print("No folder or photos selected. Stacking session cancelled.")
            sys.exit(0)

        files = discover_input_files(chosen)
        if len(files) < 2:
            print(f"Error: Selected path contains {len(files)} supported photo(s). Need at least 2 bracketed photos to stack.")
            sys.exit(1)

        print(f"Selected {len(files)} bracketed photos from: {chosen[0]}")

    # Inspect EXIF photo taken timestamp sort order vs filename sort order
    disagree, files_by_name, files_by_exif, sort_details = check_sort_order_discrepancy(files)
    order_conflict = None
    on_resolve_order = None

    if args.sort_by == "exif":
        files = files_by_exif
        print(f"Notice: Applied EXIF capture timestamp ordering across {len(files)} frames.")
    elif args.sort_by == "name":
        files = files_by_name
    elif disagree:
        if not is_interactive:
            print("\n⚠️  WARNING: EXIF photo taken timestamps and filename sort order disagree!")
            print("----------------------------------------------------------------------")
            print(f"{'#':<4} {'Filename Order':<32} | {'EXIF Timestamp Order':<32}")
            print("----------------------------------------------------------------------")
            for i in range(min(5, len(files_by_name))):
                fn = f"{os.path.basename(files_by_name[i])} ({sort_details['files_by_name'][i]['timestamp']})"
                fe = f"{os.path.basename(files_by_exif[i])} ({sort_details['files_by_exif'][i]['timestamp']})"
                print(f"{i+1:<4} {fn:<32} | {fe:<32}")
            if len(files_by_name) > 5:
                print(f"... and {len(files_by_name) - 5} more frames.")
            print("----------------------------------------------------------------------")
            if sys.stdin.isatty():
                try:
                    ans = input("How would you like to order the files? [1 = EXIF Timestamp (Recommended), 2 = Filename Order] (default 1): ").strip()
                    if ans == "2":
                        files = files_by_name
                        print("Applying filename order.")
                    else:
                        files = files_by_exif
                        print("Applying EXIF capture timestamp order.")
                except (KeyboardInterrupt, EOFError):
                    sys.exit(0)
            else:
                print("Notice: Non-interactive batch mode. Defaulting to EXIF capture timestamp order.")
                files = files_by_exif
        else:
            order_conflict = sort_details
            files = files_by_name
            def on_resolve_order(choice):
                chosen_files = files_by_exif if choice == 'exif' else files_by_name
                w_fn, _ = build_worker_for_files(chosen_files, requested_output=args.output, crop=args.crop)
                return w_fn
    else:
        files = files_by_name

    interactive_worker, resolved_output = build_worker_for_files(files, requested_output=args.output, crop=args.crop)
    args.output = resolved_output

    first_ext = os.path.splitext(files[0])[1].lower()
    is_jpeg_source = first_ext in ('.jpg', '.jpeg')
    fmt_label = "JPEG" if is_jpeg_source else "16-Bit TIFF"
    print(f"============================================================")
    print(f"Argon Stacker — {fmt_label} Fusion Engine")
    print(f"Found {len(files)} bracketed {fmt_label} frames in sequence")
    print(f"Target Output: {args.output}")
    print(f"Mode: {'Interactive Native GUI' if is_interactive else 'Headless Batch Render'}")
    print(f"============================================================")

    t0 = time.time()
    is_lightroom = getattr(args, 'lightroom', False) or (os.environ.get("FOCUS_STACK_LIGHTROOM") == "1") or bool(args.output and ".focus_stack_tmp" in args.output)

    if is_interactive:
        cancelled = run_interactive_server(
            interactive_worker,
            start_port=8765,
            is_dropzone=False,
            is_lightroom=is_lightroom,
            default_output_path=args.output,
            order_conflict=order_conflict,
            on_resolve_order=on_resolve_order
        )
        if cancelled:
            print("Interactive session cancelled by user.")
            sys.exit(2)
        print("Interactive session finished successfully.")
        return

    # Headless / Batch Mode
    align_res = align_frames(files, crop=args.crop)
    warp_mats, crop_box, (full_w, full_h), icc_profile = align_res

    print(f"\n[Pass 2/2] Streaming 16-bit EG-MSD focus stack (bg_mode={args.bg_mode}, stack_order={args.stack_order}, z_jump={args.z_jump}, halo_gamma={args.halo_gamma}, anisotropy={args.anisotropy}, threshold={args.threshold})...")
    t_fuse = time.time()
    stream_fuse_focus_stack(
        files=files,
        warp_mats=warp_mats,
        crop_box=crop_box,
        method="egmsd",
        noise_threshold=args.threshold,
        z_jump=args.z_jump,
        halo_gamma=args.halo_gamma,
        anisotropy=args.anisotropy,
        specular_thresh=args.specular_thresh,
        bg_mode=args.bg_mode,
        stack_order=args.stack_order,
        icc_profile=icc_profile,
        output_path=args.output
    )
    print(f"Fusion complete in {time.time() - t_fuse:.2f}s")
    print(f"============================================================")
    print(f"✅ Stacking completed successfully in {time.time() - t0:.2f}s!")
    print(f"Output saved to: {args.output}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
