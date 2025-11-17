"""Tests for vol reader/writer modules"""
import os
import tempfile
import struct
import numpy as np
from usal_diaphragm.vol.fio import vol_reader, vol_writer
from usal_diaphragm.vol import vol_obj, read_4d_vol


def create_test_vol_file(filepath, n_frames=1):
    """Create a test .vol file"""
    ni, nj, nk = 10, 8, 6
    frame_size = ni * nj * nk

    with open(filepath, "wb") as fid:
        # Write header
        fid.write("KRETZFILE 1.0   ".encode("utf-8"))

        # Write dimensions
        fid.write(bytes.fromhex("00C00100"))
        fid.write(struct.pack("<Q", 8))
        fid.write(struct.pack("<Q", ni))

        fid.write(bytes.fromhex("00C00200"))
        fid.write(struct.pack("<Q", 8))
        fid.write(struct.pack("<Q", nj))

        fid.write(bytes.fromhex("00C00300"))
        fid.write(struct.pack("<Q", 8))
        fid.write(struct.pack("<Q", nk))

        # Write properties
        fid.write(bytes.fromhex("00C20100"))  # offset1
        fid.write(struct.pack("<Q", 8))
        fid.write(struct.pack("d", 1.5))

        fid.write(bytes.fromhex("00C20200"))  # offset2
        fid.write(struct.pack("<Q", 8))
        fid.write(struct.pack("d", -0.5))

        fid.write(bytes.fromhex("00C10100"))  # rad_res
        fid.write(struct.pack("<Q", 8))
        fid.write(struct.pack("d", 0.001))

        fid.write(bytes.fromhex("10002200"))  # cartesian_spacing
        fid.write(struct.pack("<Q", 8))
        fid.write(struct.pack("d", 0.5))

        # Write angles
        theta_data = b"".join([struct.pack("d", 1.0 + i * 0.1) for i in range(nj)])
        fid.write(bytes.fromhex("00C30200"))  # theta_angles
        fid.write(struct.pack("<Q", len(theta_data)))
        fid.write(theta_data)

        phi_data = b"".join([struct.pack("d", 1.5 - i * 0.1) for i in range(nk)])
        fid.write(bytes.fromhex("00C30100"))  # phi_angles
        fid.write(struct.pack("<Q", len(phi_data)))
        fid.write(phi_data)

        # Write frame data
        frame_data = bytes([i % 256 for i in range(frame_size)])
        fid.write(bytes.fromhex("00D00100"))
        fid.write(struct.pack("<Q", frame_size))
        fid.write(frame_data)

        # Write sequence data if multiple frames
        if n_frames > 1:
            seq_data = frame_data * n_frames
            fid.write(bytes.fromhex("00D60100"))
            fid.write(struct.pack("<Q", len(seq_data)))
            fid.write(seq_data)


def test_vol_reader_basic():
    """Test basic vol file reading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.vol")
        create_test_vol_file(filepath)

        reader = vol_reader.VolReader(filepath)
        vol = reader.read()

        assert vol is not None
        assert isinstance(vol, vol_obj.VolObj)


def test_vol_reader_dimensions():
    """Test reading volume dimensions"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.vol")
        create_test_vol_file(filepath)

        reader = vol_reader.VolReader(filepath)
        vol = reader.read()

        nk, nj, ni = vol.frame_dimensions()
        assert ni == 10
        assert nj == 8
        assert nk == 6


def test_vol_reader_properties():
    """Test reading volume properties"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.vol")
        create_test_vol_file(filepath)

        reader = vol_reader.VolReader(filepath)
        vol = reader.read()

        props = vol.volume_properties()
        assert abs(props["offset1"] - 1.5) < 1e-6
        assert abs(props["offset2"] - (-0.5)) < 1e-6
        assert abs(props["rad_res"] - 0.001) < 1e-9


def test_vol_reader_multiframe():
    """Test reading multi-frame volume"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.vol")
        create_test_vol_file(filepath, n_frames=3)

        reader = vol_reader.VolReader(filepath)
        vol = reader.read()

        assert vol.n_frames() == 3


def test_vol_writer_basic():
    """Test basic vol file writing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First create a vol object
        read_path = os.path.join(tmpdir, "input.vol")
        create_test_vol_file(read_path, n_frames=2)

        reader = vol_reader.VolReader(read_path)
        vol = reader.read()

        # Write it out
        write_path = os.path.join(tmpdir, "output.vol")
        writer = vol_writer.VolWriter(write_path)
        writer.write_one(vol, frame=0)

        # Check file was created
        expected_file = os.path.join(tmpdir, "output-000.vol")
        assert os.path.exists(expected_file)


def test_vol_writer_all_frames():
    """Test writing all frames"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a multi-frame vol
        read_path = os.path.join(tmpdir, "input.vol")
        create_test_vol_file(read_path, n_frames=3)

        reader = vol_reader.VolReader(read_path)
        vol = reader.read()

        # Write all frames
        write_path = os.path.join(tmpdir, "output.vol")
        writer = vol_writer.VolWriter(write_path)
        writer.write_all(vol)

        # Check all files were created
        assert os.path.exists(os.path.join(tmpdir, "output-000.vol"))
        assert os.path.exists(os.path.join(tmpdir, "output-001.vol"))
        assert os.path.exists(os.path.join(tmpdir, "output-002.vol"))


def test_vol_writer_custom_destdir():
    """Test writing to custom destination directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        read_path = os.path.join(tmpdir, "input.vol")
        create_test_vol_file(read_path, n_frames=1)

        reader = vol_reader.VolReader(read_path)
        vol = reader.read()

        # Create custom destination
        destdir = os.path.join(tmpdir, "output_dir")
        os.makedirs(destdir)

        write_path = os.path.join(tmpdir, "test.vol")
        writer = vol_writer.VolWriter(write_path, destdir=destdir)
        writer.write_one(vol, frame=0)

        # Check file in custom dir
        assert os.path.exists(os.path.join(destdir, "test-000.vol"))


def test_read_4d_vol_from_file():
    """Test reading 4D vol from single file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.vol")
        # Create large enough file (>100MB requirement bypassed by setting smaller data)
        # Actually, let's create a file that's big enough
        ni, nj, nk = 200, 100, 50
        n_frames = 11  # ~110MB
        frame_size = ni * nj * nk

        with open(filepath, "wb") as fid:
            fid.write("KRETZFILE 1.0   ".encode("utf-8"))

            # Write dimensions
            for hex_code, val in [("00C00100", ni), ("00C00200", nj), ("00C00300", nk)]:
                fid.write(bytes.fromhex(hex_code))
                fid.write(struct.pack("<Q", 8))
                fid.write(struct.pack("<Q", val))

            # Write minimal properties
            for hex_code, val in [("00C20100", 1.5), ("00C20200", -0.5),
                                   ("00C10100", 0.001), ("10002200", 0.5)]:
                fid.write(bytes.fromhex(hex_code))
                fid.write(struct.pack("<Q", 8))
                fid.write(struct.pack("d", val))

            # Write angles
            theta_data = b"".join([struct.pack("d", 1.0 + i * 0.01) for i in range(nj)])
            fid.write(bytes.fromhex("00C30200"))
            fid.write(struct.pack("<Q", len(theta_data)))
            fid.write(theta_data)

            phi_data = b"".join([struct.pack("d", 1.5 - i * 0.01) for i in range(nk)])
            fid.write(bytes.fromhex("00C30100"))
            fid.write(struct.pack("<Q", len(phi_data)))
            fid.write(phi_data)

            # Write frame data
            frame_data = bytes([i % 256 for i in range(frame_size)])
            fid.write(bytes.fromhex("00D00100"))
            fid.write(struct.pack("<Q", frame_size))
            fid.write(frame_data)

            # Write sequence
            seq_data = frame_data * n_frames
            fid.write(bytes.fromhex("00D60100"))
            fid.write(struct.pack("<Q", len(seq_data)))
            fid.write(seq_data)

        vol_arrays, vol_props = read_4d_vol.read_4d_vol_from(filepath)

        assert vol_arrays is not None
        assert vol_props is not None
        assert len(vol_arrays) == n_frames


def test_read_4d_vol_from_directory():
    """Test reading 4D vol from directory of 3D volumes"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory with multiple .vol files
        for i in range(3):
            filepath = os.path.join(tmpdir, f"frame-{i:03d}.vol")
            create_test_vol_file(filepath, n_frames=1)

        vol_arrays, vol_props = read_4d_vol.read_4d_vol_from(tmpdir)

        assert vol_arrays is not None
        assert vol_props is not None
        assert len(vol_arrays) == 3


def test_read_4d_vol_invalid_input():
    """Test reading from invalid input"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a non-.vol file
        filepath = os.path.join(tmpdir, "test.txt")
        with open(filepath, "w") as f:
            f.write("not a vol file")

        vol_arrays, vol_props = read_4d_vol.read_4d_vol_from(filepath)

        # Should return None for invalid input
        assert vol_props is None


def test_read_4d_vol_small_file():
    """Test that small .vol files are ignored"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "small.vol")
        create_test_vol_file(filepath, n_frames=1)  # This will be < 100MB

        vol_arrays, vol_props = read_4d_vol.read_4d_vol_from(filepath)

        # Small files should be ignored
        assert vol_props is None
