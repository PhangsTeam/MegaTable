import numpy as np
import pytest
from astropy.io import fits


def make_simple_header(
        naxis1: int = 20,
        naxis2: int = 20,
        crval1: float = 180.0,
        crval2: float = 0.0,
        cdelt: float = 0.001) -> fits.Header:
    """
    Build a minimal two-axis TAN-projection FITS header.

    The reference point (crval1, crval2) is placed at the centre of
    the image.  With the default pixel scale of 0.001 deg (3.6 arcsec)
    the 20 x 20 map spans roughly ±34 arcsec in both axes.
    """
    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = naxis1
    hdr['NAXIS2'] = naxis2
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CRVAL1'] = crval1
    hdr['CRVAL2'] = crval2
    hdr['CRPIX1'] = naxis1 / 2 + 0.5   # half-integer → reference at centre
    hdr['CRPIX2'] = naxis2 / 2 + 0.5
    hdr['CDELT1'] = -cdelt              # RA decreases with pixel index
    hdr['CDELT2'] = cdelt
    hdr['BUNIT'] = 'K'
    return hdr


@pytest.fixture
def simple_header() -> fits.Header:
    """Return the default 20×20 TAN-projection FITS header."""
    return make_simple_header()


@pytest.fixture
def uniform_image_hdu(simple_header: fits.Header) -> fits.ImageHDU:
    """Return an HDU whose data array is all ones."""
    data = np.ones((simple_header['NAXIS2'], simple_header['NAXIS1']))
    return fits.ImageHDU(data=data, header=simple_header)
