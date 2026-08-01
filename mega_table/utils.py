import os
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
from astropy import units as u
from astropy.units import Quantity, UnitBase
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord

HDU_types = (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)
UnitLike = Union[str, UnitBase]
ImageInput = Union[
    str, bytes, os.PathLike, fits.HDUList,
    fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU, np.ndarray,
]
CenterCoord = Union[
    SkyCoord,
    Tuple[Union[float, Quantity], Union[float, Quantity]],
]

# --------------------------------------------------------------------


def identical_units(u1: UnitLike, u2: UnitLike) -> bool:
    """
    Check whether two units are exactly identical.

    Parameters
    ----------
    u1 : str or `~astropy.units.UnitBase`
        First unit specification.
    u2 : str or `~astropy.units.UnitBase`
        Second unit specification.

    Returns
    -------
    bool
        `True` when the two inputs represent the same unit exactly,
        not merely equivalent units.
    """
    if not u.Unit(u1).is_equivalent(u.Unit(u2)):
        return False
    elif (u.Unit(u1) / u.Unit(u2)).to('') != 1:
        return False
    else:
        return True


# --------------------------------------------------------------------


def calc_pixel_area(header: fits.Header) -> Quantity:
    """
    Calculate the projected area of one image pixel.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        FITS header describing a celestial image.

    Returns
    -------
    `~astropy.units.Quantity`
        Pixel area returned by the WCS projection.
    """
    wcs = WCS(header)
    return wcs.proj_plane_pixel_area()


# --------------------------------------------------------------------


def calc_pixel_per_beam(
        header: fits.Header,
        suppress_no_beam_error: bool = True) -> Optional[float]:
    """
    Calculate the number of pixels per synthesized beam.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        FITS header describing a radio image.
    suppress_no_beam_error : bool, optional
        If `True`, return `None` when the header does not contain beam
        information. If `False`, re-raise the missing-beam exception.

    Returns
    -------
    float or None
        Number of pixels per beam, or `None` when no beam is present and
        `suppress_no_beam_error` is `True`.

    Raises
    ------
    radio_beam.beam.NoBeamException
        Raised when beam information is missing and
        `suppress_no_beam_error` is `False`.
    """
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


def nanaverage(a: ArrayLike, **kwargs: Any) -> Any:
    """
    Compute a weighted average while ignoring NaN values.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged.
    **kwargs
        Additional keyword arguments passed to `~numpy.ma.average`.

    Returns
    -------
    ndarray or scalar
        Weighted average along the requested axis.
    """
    avg = np.ma.average(np.ma.array(a, mask=np.isnan(a)), **kwargs)
    avg = np.ma.filled(avg, np.nan)
    return avg if avg.size > 1 else avg.item()


# --------------------------------------------------------------------


def nanrms(a: ArrayLike, **kwargs: Any) -> Any:
    """
    Compute a weighted root-mean-square while ignoring NaN values.

    Parameters
    ----------
    a : array_like
        Array containing data to be summarized.
    **kwargs
        Additional keyword arguments passed to `~numpy.ma.average`.

    Returns
    -------
    ndarray or scalar
        Root-mean-square along the requested axis.
    """
    rms = np.sqrt(np.ma.average(
        np.ma.array(a, mask=np.isnan(a))**2, **kwargs))
    rms = np.ma.filled(rms, np.nan)
    return rms if rms.size > 1 else rms.item()


# --------------------------------------------------------------------


def reduce_image_input(
        image: ImageInput,
        ihdu: int = 0,
        header: Optional[fits.Header] = None,
        suppress_error: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[fits.Header], Optional[WCS]]:
    """
    Normalize image inputs to data, header, and WCS objects.

    Parameters
    ----------
    image : str, bytes, path-like, `~astropy.io.fits.HDUList`, FITS HDU,
        or ndarray
        Image source to normalize.
    ihdu : int, optional
        HDU index to use when `image` is a FITS filename or HDU list.
    header : `~astropy.io.fits.Header`, optional
        FITS header describing the array when `image` is an ndarray.
    suppress_error : bool, optional
        If `True`, return `(None, None, None)` instead of raising when
        `image` looks like a file path but does not exist.

    Returns
    -------
    data : ndarray or None
        Image data array.
    hdr : `~astropy.io.fits.Header` or None
        Header associated with the image data.
    wcs : `~astropy.wcs.WCS` or None
        Celestial WCS built from `hdr`.

    Raises
    ------
    ValueError
        Raised when the input image cannot be found, is not
        two-dimensional, has inconsistent dimensions, or does not use
        celestial RA/Dec axes.
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
        center_coord: Optional[CenterCoord] = None,
        incl: Union[float, Quantity] = 0*u.deg,
        pa: Union[float, Quantity] = 0*u.deg,
        header: Optional[fits.Header] = None,
        wcs: Optional[WCS] = None,
        naxis: Optional[Sequence[int]] = None,
        ra: Optional[Union[ArrayLike, Quantity]] = None,
        dec: Optional[Union[ArrayLike, Quantity]] = None,
        return_offset: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
    ],
]:
    """
    Calculate deprojected coordinates from sky coordinates.

    This function deals with sky images of astronomical objects with
    an intrinsic disk geometry. Given disk center coordinates,
    inclination, and position angle, it calculates the deprojected
    coordinates in the disk plane (radius and azimuthal angle) from
    (1) a FITS header (`header`), or
    (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
    (3) RA and DEC coodinates (`ra` + `dec`).
    Note that the deprojected azimuthal angle is defined w.r.t. the
    line of nodes (given by the position angle). For (1) and (2), the
    outputs are 2D images; for (3), the outputs are arrays with shapes
    matching the broadcasted shape of `ra` and `dec`.

    Parameters
    ----------
    center_coord : `~astropy.coordinates.SkyCoord` or 2-tuple, optional
        Sky coordinates of the disk center.
    incl : number or `~astropy.units.Quantity`, optional
        Disk inclination angle. Zero degrees corresponds to a face-on
        disk.
    pa : number or `~astropy.units.Quantity`, optional
        Disk position angle for the receding major axis, measured from
        north through east.
    header : `~astropy.io.fits.Header`, optional
        FITS header specifying the WCS and output map shape.
    wcs : `~astropy.wcs.WCS`, optional
        WCS of the output maps when `header` is not supplied.
    naxis : sequence of int, optional
        Two-element map shape associated with `wcs`.
    ra : array_like or `~astropy.units.Quantity`, optional
        Right ascension values to deproject when working from explicit
        coordinates instead of a WCS description.
    dec : array_like or `~astropy.units.Quantity`, optional
        Declination values to deproject when working from explicit
        coordinates instead of a WCS description.
    return_offset : bool, optional
        If `True`, also return intermediate offset coordinates in both
        sky and disk frames.

    Returns
    -------
    deprojected_coordinates : tuple of ndarray
        Returns `(radius_deg, projang_deg)` by default. When
        `return_offset` is `True`, the returned tuple becomes
        `(radius_deg, projang_deg, dx_deg, dy_deg, dmaj_deg, dmin_deg)`.

    Notes
    -----
    This is the Python version of an IDL function `deproject` included
    in the `cpropstoo` package. See URL below:
    https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro
    """

    if isinstance(center_coord, SkyCoord):
        x0_deg = center_coord.ra.degree
        y0_deg = center_coord.dec.degree
    else:
        x0_deg, y0_deg = center_coord
        if hasattr(x0_deg, 'unit'):
            x0_deg = x0_deg.to(u.deg).value
            y0_deg = y0_deg.to(u.deg).value
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
    dmaj_deg = (dx_deg * np.cos(rotangle) +
                dy_deg * np.sin(rotangle))
    dmin_deg = (dy_deg * np.cos(rotangle) -
                dx_deg * np.sin(rotangle))
    dmin_deg /= np.cos(np.deg2rad(incl_deg))

    # make map of deprojected distance from the center
    radius_deg = np.sqrt(dmaj_deg**2 + dmin_deg**2)

    # make map of angle w.r.t. position angle
    projang_deg = np.rad2deg(np.arctan2(dmin_deg, dmaj_deg))

    if return_offset:
        return radius_deg, projang_deg, \
            dx_deg, dy_deg, dmaj_deg, dmin_deg
    else:
        return radius_deg, projang_deg
