"""Tests for vol_obj module"""
import struct
import numpy as np
from usal_diaphragm.vol.vol_obj import VolObj


def create_test_data_dict():
    """Create a test data dictionary for VolObj"""
    ni, nj, nk = 10, 8, 6
    frame_size = ni * nj * nk

    # Create dummy frame data
    frame_data = bytes([i % 256 for i in range(frame_size)])

    # Create sequence data (3 frames)
    n_frames = 3
    seq_data = frame_data * n_frames

    data_dict = {
        "frame_data": (struct.pack("<Q", frame_size), frame_data),
        "sequence_data": (struct.pack("<Q", frame_size * n_frames), seq_data),
        "dim_i": (8, struct.pack("<Q", ni)),
        "dim_j": (8, struct.pack("<Q", nj)),
        "dim_k": (8, struct.pack("<Q", nk)),
        "offset1": (8, struct.pack("d", 1.5)),
        "offset2": (8, struct.pack("d", -0.5)),
        "rad_res": (8, struct.pack("d", 0.001)),
        "cartesian_spacing": (8, struct.pack("d", 0.5)),
        "theta_angles": (8 * nj, b"".join([struct.pack("d", 1.0 + i * 0.1) for i in range(nj)])),
        "phi_angles": (8 * nk, b"".join([struct.pack("d", 1.5 - i * 0.1) for i in range(nk)])),
    }
    return data_dict


def test_vol_obj_creation():
    """Test VolObj creation"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)
    assert vol is not None


def test_vol_obj_frame_dimensions():
    """Test getting frame dimensions"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    nk, nj, ni = vol.frame_dimensions()
    assert ni == 10
    assert nj == 8
    assert nk == 6


def test_vol_obj_n_frames():
    """Test getting number of frames"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    n_frames = vol.n_frames()
    assert n_frames == 3


def test_vol_obj_frame_data_default():
    """Test getting default frame data"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    frame = vol.frame_data()
    assert frame is not None
    assert len(frame) == 10 * 8 * 6


def test_vol_obj_frame_data_by_index():
    """Test getting frame data by index"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    frame0 = vol.frame_data(0)
    frame1 = vol.frame_data(1)
    frame2 = vol.frame_data(2)

    assert len(frame0) == 10 * 8 * 6
    assert len(frame1) == 10 * 8 * 6
    assert len(frame2) == 10 * 8 * 6


def test_vol_obj_frame_data_out_of_bounds():
    """Test that out of bounds frame index raises IndexError"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    try:
        vol.frame_data(10)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


def test_vol_obj_volume_properties():
    """Test getting volume properties"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    props = vol.volume_properties()

    assert props["dim_i"] == 10
    assert props["dim_j"] == 8
    assert props["dim_k"] == 6
    assert abs(props["offset1"] - 1.5) < 1e-6
    assert abs(props["offset2"] - (-0.5)) < 1e-6
    assert abs(props["rad_res"] - 0.001) < 1e-9
    assert abs(props["cartesian_spacing"] - 0.5) < 1e-6


def test_vol_obj_theta_angles():
    """Test getting theta angles"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    theta = vol.theta_angles()
    assert theta is not None


def test_vol_obj_phi_angles():
    """Test getting phi angles"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    phi = vol.phi_angles()
    assert phi is not None


def test_vol_obj_items():
    """Test getting items from data dict"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    items = vol.items()
    assert items is not None
    assert len(list(items)) > 0


def test_vol_obj_frame_size():
    """Test getting frame size"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    size = vol.frame_size()
    assert size is not None


def test_vol_obj_sequence_size():
    """Test getting sequence size"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    size = vol.sequence_size()
    assert size is not None


def test_vol_obj_without_sequence_data():
    """Test VolObj without sequence data"""
    data_dict = create_test_data_dict()
    del data_dict["sequence_data"]

    vol = VolObj(data_dict)
    n_frames = vol.n_frames()

    # Should default to 1 frame
    assert n_frames == 1


def test_vol_obj_angle_lists():
    """Test that angle lists are properly decoded"""
    data_dict = create_test_data_dict()
    vol = VolObj(data_dict)

    props = vol.volume_properties()

    assert len(props["theta_angles"]) == 8
    assert len(props["phi_angles"]) == 6

    # Check values are approximately correct
    assert abs(props["theta_angles"][0] - 1.0) < 1e-6
    assert abs(props["phi_angles"][0] - 1.5) < 1e-6
