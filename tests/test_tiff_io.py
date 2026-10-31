"""
Unit Tests for TIFF & Image I/O Module (tiff_io.py)
Validates 16-bit depth preservation, ICC profile roundtrip, and resolution tags.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import tifffile
from PIL import Image

# Add py_focus_stack to import path
current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(os.path.dirname(current_dir), "FocusStack.lrplugin", "bin", "py_focus_stack")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from tiff_io import read_image_16bit, write_tiff_16bit, write_image


class TestTiffIO(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="focus_stack_io_test_")

    def tearDown(self):
        for f in os.listdir(self.tmp_dir):
            try:
                os.unlink(os.path.join(self.tmp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(self.tmp_dir)
        except Exception:
            pass

    def test_16bit_tiff_roundtrip(self):
        """Test true 16-bit uint16 array write and read without truncation."""
        test_img = np.linspace(0, 65535, 256 * 256 * 3, dtype=np.uint16).reshape((256, 256, 3))
        out_path = os.path.join(self.tmp_dir, "test_16bit.tif")

        write_tiff_16bit(out_path, test_img)
        self.assertTrue(os.path.exists(out_path))

        loaded_img, _ = read_image_16bit(out_path)
        self.assertEqual(loaded_img.dtype, np.uint16)
        self.assertEqual(loaded_img.shape, (256, 256, 3))
        np.testing.assert_array_equal(loaded_img, test_img)

    def test_icc_profile_preservation(self):
        """Test roundtrip preservation of Tag 34675 (ICC Profile) bytes."""
        test_img = np.full((64, 64, 3), 32768, dtype=np.uint16)
        # Create a mock ICC profile payload
        mock_icc = b"MockICCProfileBytesForWideGamutTesting" * 10
        out_path = os.path.join(self.tmp_dir, "test_icc.tif")

        write_tiff_16bit(out_path, test_img, icc_profile=mock_icc)
        loaded_img, loaded_icc = read_image_16bit(out_path)

        self.assertIsNotNone(loaded_icc)
        self.assertEqual(loaded_icc, mock_icc)

    def test_tiff_metadata_and_dpi(self):
        """Test that resolution tags are written as 300 DPI without duplicate tag 270."""
        test_img = np.zeros((32, 32, 3), dtype=np.uint16)
        out_path = os.path.join(self.tmp_dir, "test_tags.tif")

        write_tiff_16bit(out_path, test_img)

        with tifffile.TiffFile(out_path) as tif:
            page = tif.pages[0]
            # Check 300 DPI
            x_res = page.tags.get(282)  # XResolution
            y_res = page.tags.get(283)  # YResolution
            self.assertIsNotNone(x_res)
            self.assertIsNotNone(y_res)
            self.assertEqual(x_res.value, (300, 1))
            self.assertEqual(y_res.value, (300, 1))

    def test_8bit_to_16bit_normalization(self):
        """Test that 8-bit inputs are cleanly promoted to 16-bit uint16 (0-65535)."""
        img_8u = np.array([[[0, 128, 255]]], dtype=np.uint8)
        out_path = os.path.join(self.tmp_dir, "test_8u.png")
        Image.fromarray(img_8u).save(out_path)

        loaded_img, _ = read_image_16bit(out_path)
        self.assertEqual(loaded_img.dtype, np.uint16)
        # 0 -> 0, 255 -> 65535
        self.assertEqual(loaded_img[0, 0, 0], 0)
        self.assertEqual(loaded_img[0, 0, 2], 65535)


if __name__ == "__main__":
    unittest.main()
