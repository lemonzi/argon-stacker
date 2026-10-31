"""
Multi-Format Image I/O Module (TIFF & JPEG) with 16-bit Depth & ICC Profile Preservation
"""
import os
import tifffile
import numpy as np
from PIL import Image

def read_image_16bit(filepath):
    """
    Reads an image (TIFF, JPEG, PNG, etc.) ensuring 16-bit RGB output and preserving ICC profile.
    Returns:
        image: np.ndarray with dtype uint16 and shape (H, W, 3)
        icc_profile: bytes or None (raw ICC profile bytes)
    """
    ext = os.path.splitext(filepath)[1].lower()
    icc_profile = None

    # Native macOS Camera RAW decoding (Sony ARW, Canon CR2/CR3, Nikon NEF, DNG, Fuji RAF, Olympus ORF)
    RAW_EXTS = ('.arw', '.cr2', '.cr3', '.nef', '.dng', '.raw', '.rw2', '.orf', '.raf', '.pef')
    if ext in RAW_EXTS:
        try:
            import subprocess
            import tempfile
            temp_tif = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_tif_path = temp_tif.name
            temp_tif.close()
            cmd = ['/usr/bin/sips', '-s', 'format', 'tiff', filepath, '--out', temp_tif_path]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0 and os.path.exists(temp_tif_path) and os.path.getsize(temp_tif_path) > 0:
                data, icc_profile = read_image_16bit(temp_tif_path)
                try:
                    os.unlink(temp_tif_path)
                except Exception:
                    pass
                return data, icc_profile
            try:
                os.unlink(temp_tif_path)
            except Exception:
                pass
        except Exception as e:
            pass

    if ext in ('.tif', '.tiff'):
        try:
            with tifffile.TiffFile(filepath) as tif:
                page = tif.pages[0]
                data = page.asarray()

                # Read ICC profile if present
                if hasattr(page, 'tags') and 'InterColorProfile' in page.tags:
                    icc_profile = page.tags['InterColorProfile'].value
                elif hasattr(page, 'tags') and 34675 in page.tags:
                    icc_profile = page.tags[34675].value

                # Ensure 16-bit representation
                if data.dtype == np.uint8:
                    data = (data.astype(np.uint16) * 257)
                elif data.dtype != np.uint16:
                    data = data.astype(np.uint16)

                # Ensure 3-channel RGB
                if data.ndim == 3 and data.shape[2] == 4:
                    data = data[:, :, :3]
                elif data.ndim == 2:
                    data = np.stack([data, data, data], axis=-1)

                return data, icc_profile
        except Exception:
            pass

    # JPEG, PNG, or TIFF fallback via PIL (preserves embedded ICC profile from EXIF/APP2 markers)
    try:
        with Image.open(filepath) as pil_img:
            icc_profile = pil_img.info.get('icc_profile', None)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            arr = np.asarray(pil_img)
            if arr.dtype == np.uint8:
                data = (arr.astype(np.uint16) * 257)
            elif arr.dtype != np.uint16:
                data = arr.astype(np.uint16)
            else:
                data = arr
            return data, icc_profile
    except Exception:
        pass

    # Final fallback via OpenCV
    import cv2
    bgr = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Could not read image: {filepath}")
    if bgr.ndim == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)

    if rgb.dtype == np.uint8:
        rgb = (rgb.astype(np.uint16) * 257)
    elif rgb.dtype != np.uint16:
        rgb = rgb.astype(np.uint16)
    return rgb, None

# Backwards compatibility alias
read_tiff_16bit = read_image_16bit

def write_jpeg(filepath, data, icc_profile=None, quality=98):
    """
    Writes a high-fidelity 8-bit JPEG preserving ICC color profile.
    Uses 4:4:4 chroma subsampling (subsampling=0) to prevent chroma blurring
    across sharp micro-contrast boundaries and macro fine structures.
    """
    if data.dtype == np.uint16:
        data_8u = np.clip(np.round(data / 257.0), 0, 255).astype(np.uint8)
    else:
        data_8u = np.clip(data, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(data_8u)
    save_kwargs = {
        'quality': quality,
        'subsampling': 0, # 4:4:4 chroma subsampling (zero chroma smearing)
        'optimize': True
    }
    if icc_profile:
        save_kwargs['icc_profile'] = icc_profile

    out_dir = os.path.dirname(os.path.abspath(filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pil_img.save(filepath, 'JPEG', **save_kwargs)

def write_tiff_16bit(filepath, data, icc_profile=None, compression='deflate', predictor=True):
    """
    Writes a 16-bit TIFF image preserving ICC profile.
    Uses Adobe Deflate with Horizontal Differencing (predictor=True)
    to achieve 2.5x-3.5x lossless compression on 16-bit continuous tone images,
    reducing 61MP files from ~360MB down to ~90-130MB with zero loss of fidelity.
    """
    # If a JPEG output path is passed, automatically redirect to write_jpeg
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        write_jpeg(filepath, data, icc_profile=icc_profile)
        return

    if data.dtype != np.uint16:
        data = np.clip(data, 0, 65535).astype(np.uint16)

    extratags = []
    if icc_profile is not None:
        # Tag 34675 (0x8773): InterColorProfile, type 7 (UNDEFINED / BYTE)
        extratags.append((34675, 'B', len(icc_profile), icc_profile, False))

    out_dir = os.path.dirname(os.path.abspath(filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tifffile.imwrite(
        filepath,
        data,
        photometric='rgb' if data.ndim == 3 else 'minisblack',
        metadata=None,
        resolution=(300, 300),
        extratags=extratags if extratags else None,
        compression=compression,
        predictor=predictor,
        compressionargs={'level': 6} if compression == 'deflate' else None
    )

def write_image(filepath, data, icc_profile=None, compression='deflate', predictor=True, jpeg_quality=98):
    """
    Unified image writer that automatically formats output based on filepath extension:
    - .jpg / .jpeg: High-fidelity 8-bit JPEG with 4:4:4 chroma and embedded ICC profile.
    - .tif / .tiff: High-fidelity 16-bit TIFF with Deflate compression and embedded ICC profile.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        write_jpeg(filepath, data, icc_profile=icc_profile, quality=jpeg_quality)
    else:
        write_tiff_16bit(filepath, data, icc_profile=icc_profile, compression=compression, predictor=predictor)
