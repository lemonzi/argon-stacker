"""
Subpixel Geometric Alignment Module for Focus Stacking
Uses OpenCV Enhanced Correlation Coefficient (ECC) with Affine motion model.
"""
import cv2
import numpy as np

def read_downsampled_gray(filepath, max_dim=2048):
    """
    Reads an image from disk and returns downscaled float32 grayscale, full dimensions, and scale factor.
    Loads only one image at a time and immediately frees the full-res array to keep RAM minimal (<150MB).
    """
    from tiff_io import read_tiff_16bit
    img, icc = read_tiff_16bit(filepath)
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        sw, sh = int(w * scale), int(h * scale)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = img
        scale = 1.0
    del img

    # Convert to float32 grayscale (0.0 to 1.0)
    g = (0.2126 * (small[:, :, 0].astype(np.float32) / 65535.0) +
         0.7152 * (small[:, :, 1].astype(np.float32) / 65535.0) +
         0.0722 * (small[:, :, 2].astype(np.float32) / 65535.0))
    del small
    return g, w, h, scale, icc

def compute_alignment_matrix(ref_gray, src_gray, max_ecc_res=2048, num_iterations=50, eps=1e-4):
    """
    Computes 2x3 Affine transformation matrix from src to ref.
    Both ref_gray and src_gray should be 2D float32 arrays (0.0 to 1.0).
    """
    h, w = ref_gray.shape
    scale = 1.0
    max_dim = max(h, w)
    
    if max_dim > max_ecc_res:
        scale = max_ecc_res / float(max_dim)
        ref_small = cv2.resize(ref_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        src_small = cv2.resize(src_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        ref_small = ref_gray
        src_small = src_gray

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iterations, eps)

    try:
        _, warp_matrix = cv2.findTransformECC(
            src_small, ref_small, warp_matrix,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5
        )
    except cv2.error as e:
        # Fallback to translation if affine fails to converge
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            _, warp_matrix = cv2.findTransformECC(
                src_small, ref_small, warp_matrix,
                motionType=cv2.MOTION_TRANSLATION,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5
            )
        except Exception:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Scale translation elements back to full resolution
    if scale != 1.0:
        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale

    return warp_matrix

def warp_image_16bit(img, warp_matrix, output_size=None):
    """
    Warps a 16-bit RGB or grayscale image using bicubic interpolation.
    Returns:
        warped: warped image
        valid_mask: uint8 binary mask of valid sensor pixels (no reflected border)
    """
    h, w = img.shape[:2]
    out_w, out_h = output_size if output_size else (w, h)

    warped = cv2.warpAffine(
        img, warp_matrix, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )

    mask = np.ones((h, w), dtype=np.uint8)
    warped_mask = cv2.warpAffine(
        mask, warp_matrix, (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped, warped_mask
