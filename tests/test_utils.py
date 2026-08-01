"""Tests for mega_table.utils public functions."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from mega_table.utils import deproject, reduce_image_input
from conftest import make_simple_header


# ======================================================================
# deproject
# ======================================================================


def test_deproject_center() -> None:
    r, _ = deproject(
        center_coord=(180.0, 0.0),
        ra=np.array([180.0]),
        dec=np.array([0.0]),
    )
    assert r[0] == pytest.approx(0.0, abs=1e-12)


def test_deproject_skycoord_center() -> None:
    center = SkyCoord(180.0 * u.deg, 0.0 * u.deg)
    r, _ = deproject(
        center_coord=center,
        ra=np.array([180.0]),
        dec=np.array([0.0]),
    )
    assert r[0] == pytest.approx(0.0, abs=1e-12)


def test_deproject_faceon_dec_offset() -> None:
    # Face-on disk (incl=0, pa=0): a point 0.1 deg north of center
    # should have radius = 0.1 deg.
    r, pa = deproject(
        center_coord=(180.0, 0.0),
        incl=0,
        pa=0,
        ra=np.array([180.0]),
        dec=np.array([0.1]),
    )
    assert r[0] == pytest.approx(0.1, rel=1e-6)


def test_deproject_return_offset_gives_six_arrays() -> None:
    result = deproject(
        center_coord=(180.0, 0.0),
        ra=np.array([180.0]),
        dec=np.array([0.0]),
        return_offset=True,
    )
    assert len(result) == 6


def test_deproject_inclined_stretches_minor_axis() -> None:
    # A 60-degree inclination stretches the minor axis by 1/cos(60)=2,
    # so the same angular offset in the minor-axis direction results in
    # a larger deprojected radius than the face-on case.
    common = dict(
        center_coord=(180.0, 0.0),
        pa=0,
        ra=np.array([181.0]),
        dec=np.array([0.0]),
    )
    r_faceon, _ = deproject(incl=0, **common)
    r_inclined, _ = deproject(incl=60, **common)
    assert r_inclined[0] > r_faceon[0]


# ======================================================================
# reduce_image_input
# ======================================================================


def test_reduce_image_input_ndarray(simple_header: fits.Header) -> None:
    data = np.ones((20, 20))
    arr, hdr_out, wcs_out = reduce_image_input(data, header=simple_header)
    assert arr.shape == (20, 20)
    assert isinstance(hdr_out, fits.Header)
    assert isinstance(wcs_out, WCS)


def test_reduce_image_input_image_hdu(simple_header: fits.Header) -> None:
    data = np.ones((20, 20))
    hdu = fits.ImageHDU(data=data, header=simple_header)
    arr, hdr_out, _ = reduce_image_input(hdu)
    assert arr.shape == (20, 20)
    assert hdr_out['NAXIS1'] == 20


def test_reduce_image_input_hdulist(simple_header: fits.Header) -> None:
    data = np.ones((20, 20))
    hdu = fits.ImageHDU(data=data, header=simple_header)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    arr, _, _ = reduce_image_input(hdul, ihdu=1)
    assert arr.shape == (20, 20)


def test_reduce_image_input_missing_file_suppress() -> None:
    arr, hdr, wcs = reduce_image_input(
        '/nonexistent/image.fits', suppress_error=True)
    assert arr is None
    assert hdr is None
    assert wcs is None


def test_reduce_image_input_missing_file_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        reduce_image_input('/nonexistent/image.fits', suppress_error=False)


def test_reduce_image_input_wrong_ndim(simple_header: fits.Header) -> None:
    # A 3-D array does not satisfy the ndim == 2 requirement.
    data_3d = np.ones((5, 20, 20))
    with pytest.raises(ValueError, match="2-dimensional"):
        reduce_image_input(data_3d, header=simple_header)
