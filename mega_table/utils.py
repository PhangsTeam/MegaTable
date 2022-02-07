import os
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits

HDU_types = (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)

# --------------------------------------------------------------------


def identical_units(u1, u2):
    if not u.Unit(u1).is_equivalent(u.Unit(u2)):
        return False
    elif (u.Unit(u1) / u.Unit(u2)).to('') != 1:
        return False
    else:
        return True


# --------------------------------------------------------------------


def calc_pixel_area(header):
    from astropy.wcs import WCS
    wcs = WCS(header)
    return wcs.proj_plane_pixel_area()


# --------------------------------------------------------------------


def calc_pixel_per_beam(header, suppress_no_beam_error=True):
    from astropy.wcs import WCS
    from radio_beam import Beam
    from radio_beam.beam import NoBeamException
    try:
        beam = Beam.from_fits_header(header)
        wcs = WCS(header)
        return (beam.sr / wcs.proj_plane_pixel_area()).to('').value
    except NoBeamException as e:
        if suppress_no_beam_error:
            return None
        else:
            raise NoBeamException(e)


# --------------------------------------------------------------------


def nanaverage(a, **kwargs):
    """
    Compute weighted average along a specified axis, ignoring NaNs.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged.
    **kwargs
        Keyword arguments to be passed to `~numpy.ma.average`

    Return
    ------
    avg : ndarray or scalar
        Return the average along the specified axis.
    """
    avg = np.ma.average(np.ma.array(a, mask=np.isnan(a)), **kwargs)
    avg = np.ma.filled(avg, np.nan)
    return avg if avg.size > 1 else avg.item()


# --------------------------------------------------------------------


def nanrms(a, **kwargs):
    """
    Compute the weighted rms along a specified axis, ignoring NaNs.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged.
    **kwargs
        Keyword arguments to be passed to `~numpy.ma.average`

    Return
    ------
    rms : ndarray or scalar
        Return the rms along the specified axis.
    """
    rms = np.sqrt(np.ma.average(
        np.ma.array(a, mask=np.isnan(a))**2, **kwargs))
    rms = np.ma.filled(rms, np.nan)
    return rms if rms.size > 1 else rms.item()


# --------------------------------------------------------------------


def reduce_image_input(
        image, ihdu, header, suppress_error=False):
    """
    Reduce any combination of input values to (data, header, wcs).

    Parameters
    ----------
    image : str, fits.HDUList, fits.HDU, or np.ndarray
        The input image.
    ihdu : int, optional
        If 'image' is a str or an HDUList, this keyword should
        specify which HDU (extension) to use (default=0)
    header : astropy.fits.Header, optional
        If 'image' is an ndarray, this keyword should be a FITS
        header providing the WCS information.
    suppress_error : bool, optional
        Whether to suppress the error message if 'image' looks
        like a file but is not found on disk (default=False)

    Returns
    -------
    data : np.ndarray
    hdr : fits.Header object
    wcs : astropy.wcs.WCS object
    """
    if isinstance(image, np.ndarray):
        data = image
        hdr = header
    elif isinstance(image, HDU_types):
        data = image.data
        hdr = image.header
    elif isinstance(image, fits.HDUList):
        data = np.copy(image[ihdu].data)
        hdr = image[ihdu].header.copy()
    else:
        if (isinstance(image, (str, bytes, os.PathLike)) and
                not Path(image).is_file()):
            if suppress_error:
                return None, None, None
            else:
                raise ValueError("Input image not found")
        with fits.open(image) as hdul:
            data = np.copy(hdul[ihdu].data)
            hdr = hdul[ihdu].header.copy()
    wcs = WCS(hdr)
    if not (data.ndim == hdr['NAXIS'] == 2):
        raise ValueError(
            "Input image and/or header is not 2-dimensional")
    if not data.shape == (hdr['NAXIS2'], hdr['NAXIS1']):
        raise ValueError(
            "Input image and header have inconsistent shape")
    if not wcs.axis_type_names == ['RA', 'DEC']:
        raise ValueError(
            "Input header have unexpected axis type")
    return data, hdr, wcs


# --------------------------------------------------------------------


def deproject(
        center_ra=0*u.deg, center_dec=0*u.deg,
        incl=0*u.deg, pa=0*u.deg,
        header=None, wcs=None, naxis=None, ra=None, dec=None,
        return_offset=False):

    """
    Calculate deprojected radii and projected angles.

    This function deals with projected images of astronomical objects
    with an intrinsic disk geometry. Given sky coordinates of the
    disk center, disk inclination and position angle, this function
    calculates deprojected radii and projected angles based on
    (1) a FITS header (`header`), or
    (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
    (3) RA and DEC coordinates (`ra` + `dec`).
    Both deprojected radii and projected angles are defined relative
    to the center in the inclined disk frame. For (1) and (2), the
    outputs are 2D images; for (3), the outputs are arrays with shapes
    matching the broadcasted shape of `ra` and `dec`.

    Parameters
    ----------
    center_ra : `~astropy.units.Quantity` object or number, optional
        RA coordinate of the disk center. Default is 0 degree.
    center_dec : `~astropy.units.Quantity` object or number, optional
        Dec coordinate of the disk center. Default is 0 degree.
    incl : `~astropy.units.Quantity` object or number, optional
        Inclination angle of the disk (0 degree means face-on)
        Default is 0 degree.
    pa : `~astropy.units.Quantity` object or number, optional
        Position angle of the disk (red/receding side, North->East)
        Default is 0 degree.
    header : `~astropy.io.fits.Header` object, optional
        FITS header specifying the WCS and size of the output 2D maps
    wcs : `~astropy.wcs.WCS` object, optional
        WCS of the output 2D maps
    naxis : array-like (with two elements), optional
        Size of the output 2D maps
    ra : array-like, optional
        RA coordinate of the sky locations of interest
    dec : array-like, optional
        DEC coordinate of the sky locations of interest
    return_offset : bool, optional
        Whether to return the angular offset coordinates together with
        deprojected radii and angles. Default is to not return.

    Returns
    -------
    deprojected coordinates : tuple of array-like
        If `return_offset` is set to True, the returned arrays include
        deprojected radii, projected angles, as well as angular offset
        coordinates along East-West and North-South direction;
        otherwise only the former two arrays are returned.

    Notes
    -----
    This is the Python version of an IDL function `deproject` included
    in the `cpropstoo` package. See URL below:
    https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro
    """

    if hasattr(center_ra, 'unit'):
        x0_deg = center_ra.to(u.deg).value
    else:
        x0_deg = center_ra
    if hasattr(center_dec, 'unit'):
        y0_deg = center_dec.to(u.deg).value
    else:
        y0_deg = center_dec
    if hasattr(incl, 'unit'):
        incl_deg = incl.to(u.deg).value
    else:
        incl_deg = incl
    if hasattr(pa, 'unit'):
        pa_deg = pa.to(u.deg).value
    else:
        pa_deg = pa

    if header is not None:
        wcs_cel = WCS(header).celestial
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    elif (wcs is not None) and (naxis is not None):
        wcs_cel = wcs.celestial
        naxis1, naxis2 = naxis
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    else:
        ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        if hasattr(ra_deg, 'unit'):
            ra_deg = ra_deg.to(u.deg).value
            dec_deg = dec_deg.to(u.deg).value

    # recast the ra and dec arrays in term of the center coordinates
    # arrays are now in degrees from the center
    dx_deg = (ra_deg - x0_deg) * np.cos(np.deg2rad(y0_deg))
    dy_deg = dec_deg - y0_deg

    # rotation angle (rotate x-axis up to the major axis)
    rotangle = np.pi/2 - np.deg2rad(pa_deg)

    # create deprojected coordinate grids
    deprojdx_deg = (dx_deg * np.cos(rotangle) +
                    dy_deg * np.sin(rotangle))
    deprojdy_deg = (dy_deg * np.cos(rotangle) -
                    dx_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(incl_deg))

    # make map of deprojected distance from the center
    radius_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)

    # make map of angle w.r.t. position angle
    projang_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

    if return_offset:
        return radius_deg, projang_deg, dx_deg, dy_deg
    else:
        return radius_deg, projang_deg
