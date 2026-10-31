#!/usr/bin/env python3
"""
Reset TIFF EXIF Orientation tag (tag 274 / 0x0112) to 1 (normal).
This prevents OpenCV's imread from crashing on Apple Silicon / macOS with:
"Internal imread issue: 'original_ptr == real_mat.data' must be 'true'"
"""
import sys
import struct

def fix_orientation(filepath):
    try:
        with open(filepath, 'r+b') as f:
            header = f.read(8)
            if len(header) < 8:
                return
            if header[:2] == b'MM':
                endian = '>'
            elif header[:2] == b'II':
                endian = '<'
            else:
                return

            magic = struct.unpack(endian + 'H', header[2:4])[0]
            if magic != 42:
                return  # Only standard TIFF

            ifd_offset = struct.unpack(endian + 'I', header[4:8])[0]
            while ifd_offset != 0:
                f.seek(ifd_offset)
                num_entries_b = f.read(2)
                if len(num_entries_b) < 2:
                    break
                num_entries = struct.unpack(endian + 'H', num_entries_b)[0]
                for _ in range(num_entries):
                    entry_offset = f.tell()
                    entry = f.read(12)
                    if len(entry) < 12:
                        break
                    tag = struct.unpack(endian + 'H', entry[:2])[0]
                    if tag == 274:  # Orientation tag
                        val = struct.unpack(endian + 'H', entry[8:10])[0]
                        if val != 1:
                            f.seek(entry_offset + 8)
                            f.write(struct.pack(endian + 'H', 1))
                next_ifd_b = f.read(4)
                if len(next_ifd_b) < 4:
                    break
                ifd_offset = struct.unpack(endian + 'I', next_ifd_b)[0]
    except Exception as e:
        sys.stderr.write(f"Warning: could not check orientation for {filepath}: {e}\n")

if __name__ == '__main__':
    for path in sys.argv[1:]:
        fix_orientation(path)
