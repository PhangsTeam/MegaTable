import warnings
import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.io import fits
from astropy.wcs import WCS
from .core import GeneralRegionTable, VoronoiTessTable


######################################################################
#
# Radial Profiles
#
######################################################################


class RadialMegaTable(GeneralRegionTable):

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
            return (projrad >= rmin) & (projrad < rmax)

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
        self.meta['RBIN_DEG'] = rgal_bin_arcsec / 3600
        if rgal_max_arcsec is None:
            self.meta['RMAX_DEG'] = rgal_bin_arcsec / 3600 * 19.99
        else:
            self.meta['RMAX_DEG'] = rgal_max_arcsec / 3600

    # ----------------------------------------------------------------

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
        t = QTable.read(filename, **kwargs)

        # input file should be a valid RadialMegaTable ouput
        if 'TBLTYPE' not in t.meta:
            raise ValueError("Input file not recognized")
        if t.meta['TBLTYPE'] != 'RadialMegaTable':
            raise ValueError(
                "Cannot reconstruct RadialMegaTable object from "
                "input file")

        # initiate RadialMegaTable w/ recorded meta data in file
        mt = cls(
            t.meta['RA_DEG'], t.meta['DEC_DEG'],
            t.meta['RBIN_DEG']*3600,
            gal_incl_deg=t.meta['INCL_DEG'],
            gal_posang_deg=t.meta['PA_DEG'],
            rgal_max_arcsec=t.meta['RMAX_DEG']*3600)

        # overwrite the underlying data table
        mt.table = QTable()
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
        keep_positive : None or string or array of string, optional
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
                self.table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
                if flag.sum() > 0:
                    has_changed = True
        if keep_finite is not None:
            for col in np.atleast_1d(keep_finite):
                flag = ~np.isfinite(self[col])
                self.table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
                if flag.sum() > 0:
                    has_changed = True
        if discard_negative is not None:
            for col in np.atleast_1d(discard_negative):
                flag = self[col] < 0
                self.table = self[~flag]
                self._region_defs = compress(self._region_defs, ~flag)
                if flag.sum() > 0:
                    has_changed = True
        if keep_positive is not None:
            for col in np.atleast_1d(keep_positive):
                flag = ~(self[col] > 0)
                self.table = self[~flag]
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


class TessellMegaTable(VoronoiTessTable):

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
        self.meta['APER_DEF'] = aperture_shape
        self.meta['APER_DEG'] = aperture_size_arcsec / 3600
        self.meta['RA_DEG'] = self._ref_coord.ra.value
        self.meta['DEC_DEG'] = self._ref_coord.dec.value
        self.meta['NAXIS1'] = self._wcs._naxis[0]
        self.meta['NAXIS2'] = self._wcs._naxis[1]
        for key in self._wcs.to_header():
            self.meta[key] = self._wcs.to_header()[key]

    # ----------------------------------------------------------------

    @classmethod
    def read(cls, filename, **kwargs):
        """
        Read (and reconstruct) a TessellMegaTable from file.

        Parameters
        ----------
        filename : str
            Name of the file to read from.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`

        Return
        ------
        table : TessellMegaTable
        """
        t = QTable.read(filename, **kwargs)

        # input file should be a valid TessellMegaTable ouput
        if 'TBLTYPE' not in t.meta:
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
            aperture_shape=t.meta['APER_DEF'],
            aperture_size_arcsec=t.meta['APER_DEG']*3600,
            gal_ra_deg=t.meta['RA_DEG'],
            gal_dec_deg=t.meta['DEC_DEG'])

        # overwrite the underlying data table
        mt.table = QTable()
        for key in t.colnames:
            mt[key] = t[key]
        for key in t.meta:
            mt.meta[key] = t.meta[key]

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
        t = self.table.copy()
        if not keep_metadata:
            # remove all metadata
            for key in self.meta:
                t.meta.pop(key)
        else:
            # remove metadata not allowed in FITS headers
            hdr = fits.Header()
            for key in t.meta.copy():
                try:
                    hdr[key] = t.meta[key]
                except ValueError:
                    t.meta.pop(key)
            # special treatment for WCS keywords
            for key in ('NAXIS1', 'NAXIS2'):
                t.meta['_'+key] = t.meta[key]
                t.meta.pop(key)
            for key in WCS(t.meta).to_header():
                if key not in t.meta:
                    continue
                t.meta['_'+key] = t.meta[key]
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
        keep_positive : None or string or array of string, optional
            To only keep positive values in these column(s).
        """

        # keep track of changes in row numbers
        has_changed = False
        if discard_NaN is not None:
            for col in np.atleast_1d(discard_NaN):
                flag = np.isnan(self[col])
                self.table = self[~flag]
                if flag.sum() > 0:
                    has_changed = True
        if keep_finite is not None:
            for col in np.atleast_1d(keep_finite):
                flag = ~np.isfinite(self[col])
                self.table = self[~flag]
                if flag.sum() > 0:
                    has_changed = True
        if discard_negative is not None:
            for col in np.atleast_1d(discard_negative):
                flag = self[col] < 0
                self.table = self[~flag]
                if flag.sum() > 0:
                    has_changed = True
        if keep_positive is not None:
            for col in np.atleast_1d(keep_positive):
                flag = ~(self[col] > 0)
                self.table = self[~flag]
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

    # ----------------------------------------------------------------

    def create_maps_from_columns(self, colnames, header=None):
        """
        Create 2D maps from data in columns based on a FITS header.

        Parameters
        ----------
        colnames : iterable
            Name of the columns to create 2D maps for.
        header : `~astropy.fits.Header`, optional
            FITS header defining the WCS of the output 2D maps.
            If None (default), use the header based on which the
            TessellMegaTable object was initialized.

        Return
        ------
        arrays : list of ~numpy.ndarray
        """
        if header is None:
            wcs = WCS(fits.Header(self.meta)).celestial
        else:
            wcs = WCS(header).celestial

        # find pixels in apertures/tiles
        iax0 = np.arange(wcs._naxis[0])
        iax1 = np.arange(wcs._naxis[1]).reshape(-1, 1)
        ramap, decmap = wcs.all_pix2world(
            iax0, iax1, 0, ra_dec_order=True)
        indmap = self.find_coords_in_regions(
            ramap, decmap).reshape(ramap.shape)

        # create 2D maps
        arrays = []
        for colname in colnames:
            array = self[colname][indmap]
            array[indmap == -1] = np.nan
            arrays += [array]

        return arrays

    # ----------------------------------------------------------------

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
                self['RA'].value, self['DEC'].value, **scatterkw)
            return ffig
        else:
            # make a simple scatter plot
            if ax is None:
                import matplotlib.pyplot as plt
                plt.scatter(
                    self['RA'].value, self['DEC'].value, **scatterkw)
                return plt.gca()
            else:
                ax.scatter(
                    self['RA'].value, self['DEC'].value, **scatterkw)
                return ax


######################################################################
######################################################################
