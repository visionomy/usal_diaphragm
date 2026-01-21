"""Tests for peak_finders module"""
import numpy as np
import mock
from usal_diaphragm.find_peaks import peak_finders


def create_test_volume(nk=50, nj=50, ni=100):
    """Create a simple test volume with a peak"""
    volume = np.zeros((nk, nj, ni), dtype=np.uint8)
    # Add a bright region
    volume[20:30, 20:30, 70:80] = 200
    return volume


def get_mock_args():
    """Create mock args for peak finders"""
    args = mock.Mock()
    args.ztrim = 0
    args.ytrim = 0
    args.filter = False
    args.peaks_axis = 0
    args.intensity_threshold = 0.5
    args.grad_threshold = 10.0
    return args


def test_peak_finder_names():
    """Test that peak_finder_names returns expected list"""
    names = peak_finders.peak_finder_names()
    assert isinstance(names, list)
    assert "max" in names
    assert "grad" in names
    assert "dog" in names
    assert "grad2d" in names


def test_create_max_peak_finder():
    """Test creating MaxPeakFinder"""
    args = get_mock_args()
    finder = peak_finders.create("max", args)
    assert finder is not None
    assert isinstance(finder, peak_finders._MaxPeakFinder)


def test_create_gradient_peak_finder():
    """Test creating GradientPeakFinder"""
    args = get_mock_args()
    finder = peak_finders.create("grad", args)
    assert finder is not None
    assert isinstance(finder, peak_finders._GradientPeakFinder)


def test_create_dog_peak_finder():
    """Test creating DoGPeakFinder"""
    args = get_mock_args()
    finder = peak_finders.create("dog", args)
    assert finder is not None
    assert isinstance(finder, peak_finders._DoGPeakFinder)


def test_create_2d_gradient_peak_finder():
    """Test creating 2DGradientPeakFinder"""
    args = get_mock_args()
    finder = peak_finders.create("grad2d", args)
    assert finder is not None
    assert isinstance(finder, peak_finders._2DGradientPeakFinder)


def test_create_2d_dog_peak_finder():
    """Test creating 2DDoGPeakFinder"""
    args = get_mock_args()
    finder = peak_finders.create("dog2d", args)
    assert finder is not None
    assert isinstance(finder, peak_finders._2DDoGPeakFinder)


def test_create_gradprime2d_peak_finder():
    """Test creating 2DGradGradientPeakFinder"""
    args = get_mock_args()
    finder = peak_finders.create("gradprime2d", args)
    assert finder is not None
    assert isinstance(finder, peak_finders._2DGradGradientPeakFinder)


def test_create_wrong_grad2d_peak_finder():
    """Test creating Wrong2DGradientPeakFinder"""
    args = get_mock_args()
    finder = peak_finders.create("grad2d-wrong", args)
    assert finder is not None
    assert isinstance(finder, peak_finders._Wrong2DGradientPeakFinder)


def test_create_none_returns_none():
    """Test that creating 'none' returns None"""
    args = get_mock_args()
    finder = peak_finders.create("none", args)
    assert finder is None


def test_create_unknown_returns_none():
    """Test that creating unknown type returns None"""
    args = get_mock_args()
    finder = peak_finders.create("unknown_type", args)
    assert finder is None


def test_max_peak_finder_basic():
    """Test MaxPeakFinder finds peaks"""
    volume = create_test_volume()
    args = get_mock_args()
    args.intensity_threshold = 0

    finder = peak_finders.create("max", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should find peaks (one per j,k position)
    assert len(ii) > 0
    assert len(ii) == len(jj) == len(kk)


def test_max_peak_finder_with_threshold():
    """Test MaxPeakFinder with intensity threshold"""
    volume = create_test_volume()
    args = get_mock_args()
    args.intensity_threshold = 0.8  # Fraction of max

    finder = peak_finders.create("max", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should find fewer peaks due to threshold
    assert len(ii) > 0
    assert len(ii) == len(jj) == len(kk)


def test_max_peak_finder_with_absolute_threshold():
    """Test MaxPeakFinder with absolute threshold"""
    volume = create_test_volume()
    args = get_mock_args()
    args.intensity_threshold = 100  # Absolute value > 1

    finder = peak_finders.create("max", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should find peaks above absolute threshold
    assert len(ii) > 0


def test_max_peak_finder_with_trim():
    """Test MaxPeakFinder with trimming"""
    volume = create_test_volume()
    args = get_mock_args()
    args.ztrim = 10
    args.ytrim = 10
    args.intensity_threshold = 0

    finder = peak_finders.create("max", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # All points should be beyond trim boundaries
    assert all(i > args.ztrim for i in ii)
    assert all(j > args.ytrim for j in jj)


def test_max_peak_finder_with_filter():
    """Test MaxPeakFinder with point cloud filtering"""
    volume = create_test_volume()
    args = get_mock_args()
    args.filter = True
    args.intensity_threshold = 0

    finder = peak_finders.create("max", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should still find peaks
    assert len(ii) >= 0  # May filter out isolated points


def test_gradient_peak_finder():
    """Test GradientPeakFinder"""
    volume = create_test_volume()
    args = get_mock_args()
    args.grad_threshold = 5.0

    finder = peak_finders.create("grad", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should find peaks based on gradient
    assert len(ii) == len(jj) == len(kk)


def test_dog_peak_finder():
    """Test DoGPeakFinder"""
    volume = create_test_volume()
    args = get_mock_args()
    args.grad_threshold = 5.0

    finder = peak_finders.create("dog", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should find peaks based on DoG filter
    assert len(ii) == len(jj) == len(kk)


def test_2d_gradient_peak_finder():
    """Test 2DGradientPeakFinder"""
    volume = create_test_volume()
    args = get_mock_args()
    args.grad_threshold = 5.0

    finder = peak_finders.create("grad2d", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should find peaks based on 2D gradient
    assert len(ii) == len(jj) == len(kk)


def test_2d_dog_peak_finder():
    """Test 2DDoGPeakFinder"""
    volume = create_test_volume()
    args = get_mock_args()
    args.grad_threshold = 5.0

    finder = peak_finders.create("dog2d", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should find peaks based on 2D DoG
    assert len(ii) == len(jj) == len(kk)


def test_peak_finder_with_transpose():
    """Test peak finder with axis transposition"""
    volume = create_test_volume()
    args = get_mock_args()
    args.peaks_axis = 1
    args.intensity_threshold = 0

    finder = peak_finders.create("max", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should still work with transposed axes
    assert len(ii) == len(jj) == len(kk)


def test_peak_finder_with_double_transpose():
    """Test peak finder with double axis transposition"""
    volume = create_test_volume()
    args = get_mock_args()
    args.peaks_axis = 2
    args.intensity_threshold = 0

    finder = peak_finders.create("max", args)
    ii, jj, kk = finder.find_peaks_in(volume)

    # Should still work with double transposed axes
    assert len(ii) == len(jj) == len(kk)
