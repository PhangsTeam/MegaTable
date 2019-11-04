import os
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from .core import GeneralRegionTable, VoronoiTessTable
from .utils import nanaverage


######################################################################
######################################################################


class PhangsAlmaMixin(object):

    """
    Mixin class offering tools to deal with PHANGS-ALMA data products.
    """

    # ----------------------------------------------------------------

    def calc_CO_stats(
            self, bm0file, sm0file, sewfile, res, **kwargs):

        rstr = f"{res.to('pc').value:.0f}pc"
        cols = [
            # sums
            (f"Area_CO21_total_{rstr}", 'arcsec2'),
            (f"Area_CO21_broad_{rstr}", 'arcsec2'),
            (f"Area_CO21_strict_{rstr}", 'arcsec2'),
            (f"Flux_CO21_broad_{rstr}", 'K km s-1 arcsec2'),
            (f"Flux_CO21_strict_{rstr}", 'K km s-1 arcsec2'),
            # area-weighted averages
            (f"A<I_CO21_{rstr}>", 'K km s-1'),
            (f"A<I^2_CO21_{rstr}>", 'K2 km2 s-2'),
            (f"A<sigv_CO21_{rstr}>", 'km s-1'),
            (f"A<sigv^2_CO21_{rstr}>", 'km2 s-2'),
            (f"A<I*sigv^2_CO21_{rstr}>", 'K km3 s-3'),
            (f"A<sigv^2/I_CO21_{rstr}>", 'km s-1 K-1'),
            # CO flux-weighted averages
            (f"F<I_CO21_{rstr}>", 'K km s-1'),
            (f"F<I^2_CO21_{rstr}>", 'K2 km2 s-2'),
            (f"F<sigv_CO21_{rstr}>", 'km s-1'),
            (f"F<sigv^2_CO21_{rstr}>", 'km2 s-2'),
            (f"F<I*sigv^2_CO21_{rstr}>", 'K km3 s-3'),
            (f"F<sigv^2/I_CO21_{rstr}>", 'km s-1 K-1'),
        ]

        # if no data file found: add placeholder (empty) columns
        if not (bm0file.is_file() and sm0file.is_file() and
                sewfile.is_file()):
            for col, unit in cols:
                self[col] = np.full(len(self), np.nan) * u.Unit(unit)
            return

        # read files
        with fits.open(bm0file) as hdul:
            bm0_map = hdul[0].data.copy()
            hdr = hdul[0].header.copy()
        with fits.open(sm0file) as hdul:
            sm0_map = hdul[0].data.copy()
            if sm0_map.shape != bm0_map.shape:
                raise ValueError("Input maps have inconsistent shape")
        with fits.open(sewfile) as hdul:
            sew_map = hdul[0].data.copy()
            if sew_map.shape != bm0_map.shape:
                raise ValueError("Input maps have inconsistent shape")
        bm0_map[~np.isfinite(bm0_map)] = 0
        nan_map = np.ones_like(bm0_map).astype('float')
        nan_map[~np.isfinite(sm0_map) | (sm0_map <= 0) |
                ~np.isfinite(sew_map) | (sew_map <= 0)] = np.nan
        sm0_map[np.isnan(nan_map)] = 0
        sew_map[np.isnan(nan_map)] = 0

        # pixel size (in arcsec^2)
        pixsz = np.abs(
            hdr['CDELT1'] * hdr['CDELT2'] * u.deg**2
        ).to('arcsec2').value

        # maps corresponding to each column
        maps = [
            # sums
            np.ones_like(bm0_map).astype('float')*pixsz,
            (bm0_map != 0).astype('float')*pixsz,
            (sm0_map > 0).astype('float')*pixsz,
            bm0_map*pixsz,
            sm0_map*pixsz,
            # area-weighted averages (among regions w/ CO detection)
            sm0_map * nan_map,
            sm0_map**2 * nan_map,
            sew_map * nan_map,
            sew_map**2 * nan_map,
            sm0_map*sew_map**2 * nan_map,
            sew_map**2/sm0_map * nan_map,
            # CO flux-weighted averages
            sm0_map,
            sm0_map**2,
            sew_map,
            sew_map**2,
            sm0_map*sew_map**2,
            sew_map**2/sm0_map,
        ]

        # calculate statistics and add into table
        for (col, unit), map in zip(cols, maps):
            if col[:2] == 'A<':
                # area-weighted average
                self.calc_image_stats(
                    map, header=hdr, stat_func=nanaverage,
                    weight=None, colname=col, unit=unit, **kwargs)
            elif col[:2] == 'F<':
                # CO flux-weighted average
                self.calc_image_stats(
                    map, header=hdr, stat_func=nanaverage,
                    weight=sm0_map, colname=col, unit=unit, **kwargs)
            else:
                # sum
                self.calc_image_stats(
                    map, header=hdr, stat_func=np.nansum,
                    colname=col, unit=unit, **kwargs)

    # ----------------------------------------------------------------

    def calc_cprops_stats(
        self, cpropsfile, res, **kwargs):

        rstr = f"{res.to('pc').value:.0f}pc"
        cols = [
            # sums
            (f"Nobj_CPROPS_{rstr}", ''),
            (f"Flux_CPROPS_total_{rstr}", 'K km s-1 arcsec2'),
            # uniformly weighted averages
            (f"U<F_CPROPS_{rstr}>", 'K km s-1 arcsec2'),
            (f"U<R_CPROPS_{rstr}>", 'arcsec'),
            (f"U<F/R^2_CPROPS_{rstr}>", 'K km s-1'),
            (f"U<F^2/R^4_CPROPS_{rstr}>", 'K2 km2 s-2'),
            (f"U<F/R_CPROPS_{rstr}>", 'K km s-1 arcsec'),
            (f"U<sigv_CPROPS_{rstr}>", 'km s-1'),
            (f"U<sigv^2_CPROPS_{rstr}>", 'km2 s-2'),
            (f"U<F*sigv^2/R^2_CPROPS_{rstr}>", 'K km3 s-3'),
            (f"U<F*sigv^2/R^3_CPROPS_{rstr}>", 'K km3 s-3 arcsec-1'),
            (f"U<R*sigv^2/F_CPROPS_{rstr}>", 'km s-1 K-1 arcsec-1'),
            # CO flux-weighted averages
            (f"F<F_CPROPS_{rstr}>", 'K km s-1 arcsec2'),
            (f"F<R_CPROPS_{rstr}>", 'arcsec'),
            (f"F<F/R^2_CPROPS_{rstr}>", 'K km s-1'),
            (f"F<F^2/R^4_CPROPS_{rstr}>", 'K2 km2 s-2'),
            (f"F<F/R_CPROPS_{rstr}>", 'K km s-1 arcsec'),
            (f"F<sigv_CPROPS_{rstr}>", 'km s-1'),
            (f"F<sigv^2_CPROPS_{rstr}>", 'km2 s-2'),
            (f"F<F*sigv^2/R^2_CPROPS_{rstr}>", 'K km3 s-3'),
            (f"F<F*sigv^2/R^3_CPROPS_{rstr}>", 'K km3 s-3 arcsec-1'),
            (f"F<R*sigv^2/F_CPROPS_{rstr}>", 'km s-1 K-1 arcsec-1'),
        ]

        # if no CPROPS file found: add placeholder (empty) columns
        if not cpropsfile.is_file():
            for col, unit in cols:
                self[col] = np.full(len(self), np.nan) * u.Unit(unit)
            return

        # read CPROPS file
        try:
            t_cat = Table.read(cpropsfile)
        except ValueError as e:
            print(e)
            return
        ra_cat = t_cat['XCTR_DEG'].quantity.value
        dec_cat = t_cat['YCTR_DEG'].quantity.value
        flux_cat = (
            t_cat['FLUX_KKMS_PC2'] / t_cat['DISTANCE_PC']**2 *
            u.Unit('K km s-1 sr')).to('K km s-1 arcsec2').value
        sigv_cat = t_cat['SIGV_KMS'].quantity.value
        rad_cat = (  # expressed in Solomon+87 convention
            t_cat['RAD_PC'] / t_cat['DISTANCE_PC'] *
            u.rad).to('arcsec').value
        wt_arr = flux_cat.copy()
        wt_arr[~np.isfinite(rad_cat)] = 0
        flux_cat[~np.isfinite(rad_cat)] = np.nan
        sigv_cat[~np.isfinite(rad_cat)] = np.nan
        rad_cat[~np.isfinite(rad_cat)] = np.nan

        # entries corresponding to each column
        entries = [
            # sums
            np.isfinite(flux_cat).astype('int'),
            flux_cat,
            # uniformly weighted averages
            flux_cat,
            rad_cat,
            flux_cat/rad_cat**2,
            flux_cat**2/rad_cat**4,
            flux_cat/rad_cat,
            sigv_cat,
            sigv_cat**2,
            flux_cat*sigv_cat**2/rad_cat**2,
            flux_cat*sigv_cat**2/rad_cat**3,
            rad_cat*sigv_cat**2/flux_cat,
            # CO flux-weighted averages
            flux_cat,
            rad_cat,
            flux_cat/rad_cat**2,
            flux_cat**2/rad_cat**4,
            flux_cat/rad_cat,
            sigv_cat,
            sigv_cat**2,
            flux_cat*sigv_cat**2/rad_cat**2,
            flux_cat*sigv_cat**2/rad_cat**3,
            rad_cat*sigv_cat**2/flux_cat,
        ]

        # calculate statistics and add into table
        for (col, unit), entry in zip(cols, entries):
            if col[:2] == 'U<':
                # area-weighted average
                self.calc_catalog_stats(
                    entry, ra_cat, dec_cat,
                    stat_func=nanaverage, weight=None,
                    colname=col, unit=unit, **kwargs)
            elif col[:2] == 'F<':
                # CO flux-weighted average
                self.calc_catalog_stats(
                    entry, ra_cat, dec_cat,
                    stat_func=nanaverage, weight=wt_arr,
                    colname=col, unit=unit, **kwargs)
            else:
                # sum
                self.calc_catalog_stats(
                    entry, ra_cat, dec_cat, stat_func=np.nansum,
                    colname=col, unit=unit, **kwargs)


######################################################################
######################################################################


class EnvMaskMixin(object):

    """
    Mixin class offering tools to deal with (S4G) environmental masks.
    """

    # ----------------------------------------------------------------

    def calc_env_frac(
            self, envfile, wtfile, colname='new_col', **kwargs):

        from reproject import reproject_interp

        if not (envfile.is_file() and wtfile.is_file()):
            self[colname] = np.full(len(self), np.nan)
            return

        with fits.open(wtfile) as hdul:
            wtmap = hdul[0].data.copy()
            wtmap[~np.isfinite(wtmap) | (wtmap < 0)] = 0
            wthdr = hdul[0].header.copy()
        with fits.open(envfile) as hdul:
            envmap, footprint = reproject_interp(
                hdul[0], wthdr, order=0)
        envmap[~footprint.astype('?')] = 0
        envbimap = (envmap > 0).astype('float')
        self.calc_image_stats(
            envbimap, header=wthdr, stat_func=nanaverage,
            weight=wtmap, colname=colname, **kwargs)


######################################################################
#
# Radial Profiles
#
######################################################################


class RadialMegaTable(
        PhangsAlmaMixin, EnvMaskMixin, GeneralRegionTable):

    """
    Mega-table that quantifies a galaxy's radial profiles.

    Each ring (annular region) corresponds to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each ring.

    Parameters
    ----------
    gal_ra_deg : float, optional
        Right Ascension coordinate of the galaxy center (in degrees).
    gal_dec_deg : float, optional
        Declination coordinate of the galaxy center (in degrees).
    ring_width_deg : float
        The (deprojected) width of each ring (in degrees).
    gal_incl_deg : float, optional
        Inclination angle of the galaxy (in degrees).
        Default is 0 degree.
    gal_pa_deg : float, optional
        Position angle of the galaxy (in degrees; North through East).
        Default is 0 degree.
    max_rgal_deg : float, optional
        The (deprojected) maximum galactic radius (in degrees)
        covered by the radial profile.
        Default is to cover out to 10 times the ring width.
    """

    __name__ = "RadialMegaTable"

    # ----------------------------------------------------------------

    def __init__(
            self, gal_ra_deg, gal_dec_deg, ring_width_deg,
            gal_incl_deg=0, gal_pa_deg=0, max_rgal_deg=None):

        from functools import partial
        from .utils import deproject

        if max_rgal_deg is None:
            nring = 10
        else:
            nring = int(np.ceil((max_rgal_deg / ring_width_deg)))

        # function that determines whether a set of coordinates locate
        # inside the ring between rmin and rmax
        def coord2bool(
                ra, dec, rmin=None, rmax=None,
                gal_ra=None, gal_dec=None,
                gal_incl=None, gal_pa=None):
            projrad, projang = deproject(
                center_ra=gal_ra, center_dec=gal_dec,
                incl=gal_incl, pa=gal_pa, ra=ra, dec=dec)
            return ((projrad >= rmin) & (projrad < rmax))

        # ring names and definitions
        ring_names = [f"Ring#{iring+1}" for iring in np.arange(nring)]
        ring_defs = []
        rbounds = np.arange(nring+1) * ring_width_deg
        for rmin, rmax in zip(rbounds[:-1], rbounds[1:]):
            ring_defs += [
                partial(
                    coord2bool, rmin=rmin, rmax=rmax,
                    gal_ra=gal_ra_deg, gal_dec=gal_dec_deg,
                    gal_incl=gal_incl_deg, gal_pa=gal_pa_deg)]

        GeneralRegionTable.__init__(
            self, ring_defs, names=ring_names)
        self._table['rang_gal_min'] = rbounds[:-1] * u.deg
        self._table['rang_gal_max'] = rbounds[1:] * u.deg

    # ----------------------------------------------------------------

    def calc_image_stats(
            self, image, colname='new_col', unit=u.Unit(''),
            **kwargs):

        if (isinstance(image, (str, bytes, os.PathLike)) and
            not Path(image).is_file()):
            self[colname] = np.full(len(self), np.nan) * unit
            return

        GeneralRegionTable.calc_image_stats(
            self, image, colname=colname, unit=unit, **kwargs)

    # ----------------------------------------------------------------

    def clean(
            self, discard_NaN=None, keep_finite=None,
            discard_negative=None, keep_positive=None):
        """
        Cleaning regions that contain 'bad' column values.

        Parameters
        ----------
        discard_NaN : None or string or array of string, optional
            To remove regions showing NaN values in these column(s).
        keep_finite : None or string or array of string, optional
            To remove regions showing NaNs or Infs values in these
            column(s).
        discard_negative : None or string or array of string, optional
            To remove regions showing negative values in these
            column(s). (Note that NaNs and +Infs will not be removed)
        keep_finite : None or string or array of string, optional
            To only keep regions showing positive values in these
            column(s).
        """
        from itertools import compress
        
        if discard_NaN is not None:
            for col in np.atleast_1d(discard_NaN):
                flag = np.isnan(self[col])
                self._table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
        if keep_finite is not None:
            for col in np.atleast_1d(keep_finite):
                flag = ~np.isfinite(self[col])
                self._table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
        if discard_negative is not None:
            for col in np.atleast_1d(discard_negative):
                flag = self[col] < 0
                self._table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
        if keep_positive is not None:
            for col in np.atleast_1d(keep_positive):
                flag = ~(self[col] > 0)
                self._table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)


######################################################################
#
# Hexagonal/Rectangular/User-defined Tessellations
#
######################################################################


class TessellMegaTable(
        PhangsAlmaMixin, EnvMaskMixin, VoronoiTessTable):

    """
    Mega-table built from a tessellation of a galaxy's sky footprint.

    Each cell/aperture in the tessellation maps to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each cell/aperture, or
    by resampling images at the location of each seed.

    Parameters
    ----------
    header : astropy.fits.Header
        FITS header that defines the sky footprint of a galaxy.
    cell_shape : {'hexagon', 'square'}, optional
        The shape of the cell/aperture that comprise the tessellation.
        Default: 'hexagon', in which case a hexagonal tiling is used.
    cell_size_deg : float, optional
        The angular size of the cells/aperture (defined as the
        spacing between adjacent cell centers; in degrees).
        If None, the minimum absolute value of the 'CDELT' entry
        in the input header will be used.
    gal_ra_deg : float, optional
        Right Ascension coordinate of the galaxy center (in degrees).
        This is only used as a reference coordinate for calculating
        angular separation coordinates.
        If None, the 'CRVAL' entry in the input header will be used.
    gal_dec_deg : float, optional
        Declination coordinate of the galaxy center (in degrees).
        This is only used as a reference coordinate for calculating
        angular separation coordinates.
        If None, the 'CRVAL' entry in the input header will be used.
    """

    __name__ = "TessellMegaTable"

    # ----------------------------------------------------------------

    def __init__(
            self, header, cell_shape='hexagon', cell_size_deg=None,
            gal_ra_deg=None, gal_dec_deg=None):
        VoronoiTessTable.__init__(
            self, header,
            cell_shape=cell_shape, seed_spacing=cell_size_deg,
            ref_radec=(gal_ra_deg, gal_dec_deg))

    # ----------------------------------------------------------------

    def resample_image(
            self, infile, colname='new_col', unit=u.Unit(''),
            **kwargs):

        if not Path(infile).is_file():
            self[colname] = np.full(len(self), np.nan) * unit
            return

        VoronoiTessTable.resample_image(
            self, infile, colname=colname, unit=unit, **kwargs)


######################################################################
######################################################################
