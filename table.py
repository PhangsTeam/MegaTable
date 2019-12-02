from pathlib import Path
import warnings
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from .core import GeneralRegionTable, VoronoiTessTable
from .utils import nanaverage


######################################################################
######################################################################


class PhangsAlmaMixin(object):

    """
    Mixin class offering tools to deal with PHANGS-ALMA data products
    """

    # ----------------------------------------------------------------

    def calc_CO_stats(
            self, bm0file, sm0file, sewfile, res,
            suppress_error=False, **kwargs):
        """
        Calculate cloud-scale CO statistics using PHANGS-ALMA data

        Parameters
        ----------
        bm0file : str, path_like
            PHANGS-ALMA CO(2-1) broad moment-0 map file
        sm0file : str, path_like
            PHANGS-ALMA CO(2-1) strict moment-0 map file
        bm0file : str, path_like
            PHANGS-ALMA CO(2-1) strict effective width map file
        res : astropy.units.Quantity object
            CO data linear resolution (carrying a unit [=] pc)
        suppress_error : bool, optional
            Whether to suppress the error message if any of the input
            files are not found on disk (default=False)
        **kwargs
            Keyword arguments to be passed to 'calc_image_stats'
        """

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

        if not (Path(bm0file).is_file() and Path(sm0file).is_file() and
                Path(sewfile).is_file()):
            if suppress_error:
                # add placeholder (empty) columns
                for col, unit in cols:
                    self[col] = np.full(len(self), np.nan) * u.Unit(unit)
                return
            else:
                raise ValueError("Input file(s) not found")

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
        self, cpropsfile, res, suppress_error=False, **kwargs):
        """
        Calculate GMC statistics using (PHANGS-ALMA) CPROPS catalogs

        Parameters
        ----------
        cpropsfile : str, path_like
            PHANGS-ALMA CPROPS output file
        res : astropy.units.Quantity object
            Linear resolution (carrying a unit [=] pc) of the CO data
            on which CPROPS was run.
        suppress_error : bool, optional
            Whether to suppress the error message if any of the input
            files are not found on disk (default=False)
        **kwargs
            Keyword arguments to be passed to 'calc_image_stats'
        """

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

        if not Path(cpropsfile).is_file():
            if suppress_error:
                # add placeholder (empty) columns
                for col, unit in cols:
                    self[col] = np.full(len(self), np.nan) * u.Unit(unit)
                return
            else:
                raise ValueError("Input file not found")

        # read CPROPS file
        t_cat = Table.read(cpropsfile)
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
    Mixin class offering tools to deal with (S4G) environmental masks
    """

    # ----------------------------------------------------------------

    def calc_env_frac(
            self, envfile, wtfile=None, colname='frac_env',
            suppress_error=False, **kwargs):
        """
        Calculate environmental fractions using S4G environ. masks

        Parameters
        ----------
        envfile : str or path_like
            S4G environmental mask file (disk / bulge / bars / etc.)
        wtfile : str or path_like, optional
            A 2-D image file that assigns weight to each pixel.
            For example, one should use the CO moment-0 map to assign
            weight when estimating the fractional CO flux contribution
            from each environment.
            If None, the environmental fraction will be calculated
            using a uniform-weighting scheme (i.e., area-weighting).
        colname : str, optional
            Name of a column in the table to save the output values.
            Default: 'frac_env'
        suppress_error : bool, optional
            Whether to suppress the error message if any of the input
            files are not found on disk (default=False)
        **kwargs
            Keyword arguments to be passed to 'calc_image_stats'
        """

        from reproject import reproject_interp

        if (not Path(envfile).is_file() or
            wtfile is not None and not Path(wtfile).is_file()):
            if suppress_error:
                # add placeholder (empty) columns
                self[colname] = np.full(len(self), np.nan)
                return
            else:
                raise ValueError("Input file not found")

        if wtfile is None:
            with fits.open(envfile) as hdul:
                envmap = hdul[0].data.copy()
        else:
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
    rgal_bin_arcsec : float
        The (deprojected) width of each ring (in arcseconds).
    gal_incl_deg : float, optional
        Inclination angle of the galaxy (in degrees).
        Default is 0 degree.
    gal_posang_deg : float, optional
        Position angle of the galaxy (in degrees; North through East).
        Default is 0 degree.
    rgal_max_arcsec : float, optional
        The (deprojected) maximum galactic radius (in arcseconds)
        covered by the radial profile.
        Default is to cover out to 20 times the ring width.
    """

    __name__ = "RadialMegaTable"

    # ----------------------------------------------------------------

    def __init__(
            self, gal_ra_deg, gal_dec_deg, rgal_bin_arcsec,
            gal_incl_deg=0, gal_posang_deg=0, rgal_max_arcsec=None):

        from functools import partial
        from .utils import deproject

        if rgal_max_arcsec is None:
            nring = 20
        else:
            nring = int(np.ceil((rgal_max_arcsec / rgal_bin_arcsec)))

        # function that determines whether a set of coordinates locate
        # inside the ring between rmin and rmax
        def coord2bool(
                ra, dec, rmin=None, rmax=None,
                gal_ra=None, gal_dec=None,
                gal_incl=None, gal_posang=None):
            projrad, projang = deproject(
                center_ra=gal_ra, center_dec=gal_dec,
                incl=gal_incl, pa=gal_posang, ra=ra, dec=dec)
            return ((projrad >= rmin) & (projrad < rmax))

        # ring names and definitions
        ring_names = [f"Ring#{iring+1}" for iring in np.arange(nring)]
        ring_defs = []
        rbounds_arcsec = np.arange(nring+1) * rgal_bin_arcsec
        for rmin, rmax in zip(rbounds_arcsec[:-1], rbounds_arcsec[1:]):
            ring_defs += [
                partial(
                    coord2bool, rmin=rmin/3600, rmax=rmax/3600,
                    gal_ra=gal_ra_deg, gal_dec=gal_dec_deg,
                    gal_incl=gal_incl_deg, gal_posang=gal_posang_deg)]

        # initialize object
        GeneralRegionTable.__init__(
            self, ring_defs, names=ring_names)
        # save ring inner/outer radii in table
        self['r_gal_angl_min'] = rbounds_arcsec[:-1] * u.arcsec
        self['r_gal_angl_max'] = rbounds_arcsec[1:] * u.arcsec

        # record meta data in table
        self.meta['RA_DEG'] = gal_ra_deg
        self.meta['DEC_DEG'] = gal_dec_deg
        self.meta['INCL_DEG'] = gal_incl_deg
        self.meta['PA_DEG'] = gal_posang_deg
        self.meta['RBIN_AS'] = rgal_bin_arcsec
        if rgal_max_arcsec is None:
            self.meta['RMAX_AS'] = rgal_bin_arcsec * 19.99
        else:
            self.meta['RMAX_AS'] = rgal_max_arcsec

    #-----------------------------------------------------------------

    @classmethod
    def read(cls, filename, **kwargs):
        """
        Read (and reconstruct) a RadialMegaTable from file.

        Parameters
        ----------
        filename : str
            Name of the file to read from.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`

        Return
        ------
        table : RadialMegaTable
        """
        t = Table.read(filename, **kwargs)

        # input file should be a valid RadialMegaTable ouput
        if not 'TBLTYPE' in t.meta:
            raise ValueError("Input file not recognized")
        if t.meta['TBLTYPE'] != 'RadialMegaTable':
            raise ValueError(
                "Cannot reconstruct RadialMegaTable object from "
                "input file")

        # initiate RadialMegaTable w/ recorded meta data in file
        mt = cls(
            t.meta['RA_DEG'], t.meta['DEC_DEG'], t.meta['RBIN_AS'],
            gal_incl_deg=t.meta['INCL_DEG'],
            gal_posang_deg=t.meta['PA_DEG'],
            rgal_max_arcsec=t.meta['RMAX_AS'])

        # overwrite the underlying data table
        for key in t.colnames:
            mt[key] = t[key]
        for key in t.meta:
            mt.meta[key] = t.meta[key]

        return mt

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

        # remove rows together w/ corresponding region definitions
        # and keep track of changes in row numbers
        has_changed = False
        if discard_NaN is not None:
            for col in np.atleast_1d(discard_NaN):
                flag = np.isnan(self[col])
                self._table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
                if flag.sum() > 0:
                    has_changed = True
        if keep_finite is not None:
            for col in np.atleast_1d(keep_finite):
                flag = ~np.isfinite(self[col])
                self._table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
                if flag.sum() > 0:
                    has_changed = True
        if discard_negative is not None:
            for col in np.atleast_1d(discard_negative):
                flag = self[col] < 0
                self._table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
                if flag.sum() > 0:
                    has_changed = True
        if keep_positive is not None:
            for col in np.atleast_1d(keep_positive):
                flag = ~(self[col] > 0)
                self._table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
                if flag.sum() > 0:
                    has_changed = True

        if has_changed:
            warnings.warn(
                "Some rings has been removed during table cleaning."
                " Part of the RadialTessTable functionalities are "
                "thus disabled to avoid introducing bugs in further "
                "data reduction.")
            # remove core functionality
            self.find_coords_in_regions = None
            # present object reconstruction after writing to file
            self.meta['TBLTYPE'] = (
                self.meta['TBLTYPE'] + ' (CLEANED)')


######################################################################
#
# Hexagonal/Rectangular/User-defined Tessellations
#
######################################################################


class TessellMegaTable(
        PhangsAlmaMixin, EnvMaskMixin, VoronoiTessTable):

    """
    Mega-table built from a tessellation of a galaxy's sky footprint.

    Each aperture/tile in the tessellation maps to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each aperture, or
    by resampling images at the center of each aperture.

    Parameters
    ----------
    header : astropy.fits.Header
        FITS header that defines the sky footprint of a galaxy.
    aperture_shape : {'hexagon', 'square'}, optional
        The shape of the apertures/tiles that form the tessellation.
        Default: 'hexagon', in which case hexagonal tiles are used.
    aperture_size_arcsec : float, optional
        The angular size of the apertures/tiles (defined as the
        spacing between adjacent aperture centers; in arcseconds).
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
            self, header,
            aperture_shape='hexagon', aperture_size_arcsec=None,
            gal_ra_deg=None, gal_dec_deg=None):
        VoronoiTessTable.__init__(
            self, header,
            tile_shape=aperture_shape,
            seed_spacing=aperture_size_arcsec/3600,
            ref_radec=(gal_ra_deg, gal_dec_deg))

        # record metadata
        self.meta['APERTYPE'] = aperture_shape
        self.meta['APER_AS'] = aperture_size_arcsec
        self.meta['RA_DEG'] = self._ref_coord.ra.value
        self.meta['DEC_DEG'] = self._ref_coord.dec.value
        self.meta['NAXIS1'] = self._wcs._naxis[0]
        self.meta['NAXIS2'] = self._wcs._naxis[1]
        for key in self._wcs.to_header():
            self.meta[key] = self._wcs.to_header()[key]

    #-----------------------------------------------------------------

    @classmethod
    def read(cls, filename, ignore_inconsistency=False, **kwargs):
        """
        Read (and reconstruct) a TessellMegaTable from file.

        Parameters
        ----------
        filename : str
            Name of the file to read from.
        ignore_inconsistency : bool, optional
            Whether to suppress the error message if metadata seems
            inconsistent with table content (default: False)
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`

        Return
        ------
        table : TessellMegaTable
        """
        t = Table.read(filename, **kwargs)

        # input file should be a valid TessellMegaTable ouput
        if not 'TBLTYPE' in t.meta:
            raise ValueError("Input file not recognized")
        if t.meta['TBLTYPE'] != 'TessellMegaTable':
            raise ValueError(
                "Cannot reconstruct TessellMegaTable object from "
                "input file")

        # initiate TessellMegaTable w/ recorded meta data in file
        hdr = fits.Header()
        for key in t.meta:
            if key[0] == '_':
                hdr[key[1:]] = t.meta[key]
            else:
                hdr[key] = t.meta[key]
        mt = cls(
            hdr,
            aperture_shape=t.meta['APERTYPE'],
            aperture_size_arcsec=t.meta['APER_AS'],
            gal_ra_deg=t.meta['RA_DEG'],
            gal_dec_deg=t.meta['DEC_DEG'])

        # overwrite the underlying data table
        if len(mt) != len(t):
            if ignore_inconsistency:
                mt._table = Table()
            else:
                raise ValueError(
                    "Table content and metadata are inconsistent")
        for key in t.colnames:
            mt[key] = t[key]

        # read metadata
        for key in hdr:
            mt.meta[key] = hdr[key]

        return mt

    # ----------------------------------------------------------------

    def write(
            self, filename, keep_metadata=True, add_timestamp=True,
            **kwargs):
        """
        Write the TessellMegaTable object to a file.

        Parameters
        ----------
        filename : string
            Name of the file to write to.
        keep_metadata : bool, optional
            Whether to keep existing metadata (Default: True)
        add_timestamp : bool, optional
            Whether to add a time stamp in the metadata
            (Default: True)
        **kwargs
            Keyword arguments to be passed to `~astropy.table.write`
        """
        t = self._table.copy()
        if not keep_metadata:
            # remove all metadata
            for key in self._table.meta:
                t.meta.pop(key)
        else:
            # special treatment for WCS keywords
            for key in ('NAXIS1', 'NAXIS2'):
                t.meta['_'+key] = t.meta[key]
                t.meta.pop(key)
            for key in WCS(t.meta).to_header():
                t.meta['_'+key] = t.meta[key]
                t.meta.pop(key)
            # remove metadata not allowed in FITS headers
            hdr = fits.Header()
            for key in t.meta.copy():
                try:
                    hdr[key] = t.meta[key]
                except:
                    t.meta.pop(key)
        if 'TIMESTMP' in t.meta:
            # remove previous time stamp
            t.meta.pop('TIMESTMP')
        if add_timestamp:
            # add current time stamp
            import time
            t.meta['TIMESTMP'] = time.strftime('%c', time.gmtime())
        t.write(filename, **kwargs)

    # ----------------------------------------------------------------

    def clean(
            self, discard_NaN=None, keep_finite=None,
            discard_negative=None, keep_positive=None):
        """
        Cleaning table rows that contain 'bad' column values.

        Parameters
        ----------
        discard_NaN : None or string or array of string, optional
            To remove rows containing NaNs in these column(s).
        keep_finite : None or string or array of string, optional
            To remove NaNs and Infs in these column(s).
        discard_negative : None or string or array of string, optional
            To remove negative values in these column(s).
            (Note that NaNs and +Infs will not be removed)
        keep_finite : None or string or array of string, optional
            To only keep positive values in these column(s).
        """

        # keep track of changes in row numbers
        has_changed = False
        if discard_NaN is not None:
            for col in np.atleast_1d(discard_NaN):
                flag = np.isnan(self[col])
                self._table = self[~flag]
                if flag.sum() > 0:
                    has_changed = True
        if keep_finite is not None:
            for col in np.atleast_1d(keep_finite):
                flag = ~np.isfinite(self[col])
                self._table = self[~flag]
                if flag.sum() > 0:
                    has_changed = True
        if discard_negative is not None:
            for col in np.atleast_1d(discard_negative):
                flag = self[col] < 0
                self._table = self[~flag]
                if flag.sum() > 0:
                    has_changed = True
        if keep_positive is not None:
            for col in np.atleast_1d(keep_positive):
                flag = ~(self[col] > 0)
                self._table = self[~flag]
                if flag.sum() > 0:
                    has_changed = True

        if has_changed:
            warnings.warn(
                "Some tiles has been removed during table cleaning."
                " Part of the VonoroiTessTable functionalities are "
                "thus disabled to avoid introducing bugs in further "
                "data reduction.")
            # remove core functionality
            self.find_coords_in_regions = None
            # present object reconstruction after writing to file
            self.meta['TBLTYPE'] = (
                self.meta['TBLTYPE'] + ' (CLEANED)')

    #-----------------------------------------------------------------

    def show_apertures_on_sky(
            self, ax=None, image=None, ffigkw={}, **scatterkw):
        """
        Show RA-Dec locations of the tile centers on top of an image.

        ax : `~matplotlib.axes.Axes`, optional
            If 'image' is None, this is the Axes instance in which to
            make a scatter plot showing the tile centers.
        image : see below
            The image on which to overplot the tile centers.
            This will be passed to `aplpy.FITSFigure`.
        ffigkw : dict, optional
            Keyword arguments to be passed to `aplpy.FITSFigure`
        **scatterkw :
            Keyword arguments to be passed to `plt.scatter`
        """
        if image is not None:
            # show image using aplpy and overplot tile centers
            from aplpy import FITSFigure
            ffig = FITSFigure(image, **ffigkw)
            ffig.show_markers(
                self['RA'].quantity.value,
                self['DEC'].quantity.value,
                **scatterkw)
            return ffig
        else:
            # make a simple scatter plot
            if ax is None:
                import matplotlib.pyplot as plt
                plt.scatter(
                    self['RA'].quantity.value,
                    self['DEC'].quantity.value,
                    **scatterkw)
                return plt.gca()
            else:
                ax.scatter(
                    self['RA'].quantity.value,
                    self['DEC'].quantity.value,
                    **scatterkw)
                return ax


######################################################################
######################################################################
