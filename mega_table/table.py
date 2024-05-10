import warnings
import numpy as np
from functools import partial
from astropy import units as u
from astropy.table import QTable
from astropy.coordinates import SkyCoord
from .core import GeneralRegionTable, VoronoiTessTable
from .utils import deproject, identical_units, reduce_image_input


######################################################################
#
# Radial Binning
#
######################################################################


class RadialMegaTable(GeneralRegionTable):

    """
    Mega-table built from radial bins for a (inclined) galaxy disk.

    Each ring (annular region) corresponds to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each ring.

    Parameters
    ----------
    gal_ra_deg : float
        Right Ascension of the galaxy center (in degrees).
    gal_dec_deg : float
        Declination of the galaxy center (in degrees).
    rgal_bin_arcsec : float
        Radial width of each ring (in arcseconds).
    gal_incl_deg : float, optional
        Inclination angle of the galaxy (in degrees).
        Default is 0 degree.
    gal_posang_deg : float, optional
        Position angle of the galaxy (in degrees; North through East).
        Default is 0 degree.
    rgal_max_arcsec : float, optional
        The maximum galactic radius (in arcseconds) that the outermost
        ring will cover.
        Default is to cover out to 20 times the ring width.
    """

    __name__ = "RadialMegaTable"

    # ----------------------------------------------------------------

    def __init__(
            self, gal_ra_deg, gal_dec_deg, rgal_bin_arcsec,
            gal_incl_deg=0, gal_posang_deg=0, rgal_max_arcsec=None):

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
                ra=ra, dec=dec, center_coord=(gal_ra, gal_dec),
                incl=gal_incl, pa=gal_posang)
            return (projrad >= rmin) & (projrad < rmax)

        # ring ids and definitions
        ring_ids = np.arange(nring, dtype='int')
        rbounds_arcsec = np.arange(nring+1) * rgal_bin_arcsec
        ring_defs = [
            partial(
                coord2bool, rmin=rmin/3600, rmax=rmax/3600,
                gal_ra=gal_ra_deg, gal_dec=gal_dec_deg,
                gal_incl=gal_incl_deg, gal_posang=gal_posang_deg)
            for rmin, rmax in zip(
                rbounds_arcsec[:-1], rbounds_arcsec[1:])
        ]

        # initialize object
        super().__init__(ring_defs, names=ring_ids)
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
        Read (and reconstruct) a RadialMegaTable object from file.

        Parameters
        ----------
        filename : str
            Name of the file to read.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`

        Return
        ------
        table : RadialMegaTable
        """
        t = QTable.read(filename, **kwargs)

        # input file should be a valid RadialMegaTable output
        if 'TBLTYPE' not in t.meta:
            raise ValueError("Input file not recognized")
        if t.meta['TBLTYPE'] != 'RadialMegaTable':
            raise ValueError(
                "Cannot reconstruct a RadialMegaTable object from "
                f"the input {t.meta['TBLTYPE']} file")

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


######################################################################
#
# Stripes along Minor Axis
#
######################################################################


class StripeMegaTable(GeneralRegionTable):

    """
    Mega-table built from stripes along a galaxy's minor axis.

    Each stripe corresponds to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each stripe.

    Parameters
    ----------
    gal_ra_deg : float
        Right Ascension of the galaxy center (in degrees).
    gal_dec_deg : float
        Declination of the galaxy center (in degrees).
    gal_posang_deg : float
        Position angle of the galaxy (in degrees; North through East).
        Default is 0 degree.
    xbin_arcsec : float
        Width of each stripe (in arcseconds) along the galaxy major axis.
    xmax_arcsec : float
        The maximum distance (in arcseconds) along the galaxy major axis
        that the outermost stripes will cover.
    ymax_arcsec : float, optional
        Width of each stripe (in arcseconds) along the galaxy minor axis.
        Default is the same as `xmax_arcsec`.
    """

    __name__ = "StripeMegaTable"

    def __init__(
            self, gal_ra_deg, gal_dec_deg, gal_posang_deg,
            xbin_arcsec, xmax_arcsec, ymax_arcsec=None):

        if ymax_arcsec is None:
            ymax_arcsec = xmax_arcsec

        nbin = int(np.ceil(xmax_arcsec / xbin_arcsec)) * 2

        def coord2bool(
                ra, dec,
                gal_ra_deg=None, gal_dec_deg=None, gal_posang_deg=None,
                xmin_arcsec=None, xmax_arcsec=None, ymax_arcsec=None):
            _, _, _, _, dmaj_deg, dmin_deg = deproject(
                ra=ra, dec=dec, center_coord=(gal_ra_deg, gal_dec_deg),
                incl=0, pa=gal_posang_deg, return_offset=True)
            dmaj_arcsec, dmin_arcsec = dmaj_deg * 3600, dmin_deg * 3600
            return (
                (dmaj_arcsec > xmin_arcsec) & (dmaj_arcsec < xmax_arcsec) &
                (np.abs(dmin_arcsec) < ymax_arcsec))

        # stripe ids and definitions
        stripe_ids = np.arange(nbin / 2, dtype='int') + 1
        stripe_ids = np.hstack([-stripe_ids[::-1], stripe_ids])
        xlims_arcsec = (np.arange(nbin+1) - nbin/2) * xbin_arcsec
        stripe_defs = [
            partial(
                coord2bool, gal_ra_deg=gal_ra_deg, gal_dec_deg=gal_dec_deg,
                gal_posang_deg=gal_posang_deg, ymax_arcsec=ymax_arcsec,
                xmin_arcsec=xmin, xmax_arcsec=xmax)
            for xmin, xmax in zip(
                xlims_arcsec[:-1], xlims_arcsec[1:])
        ]

        # initialize object
        super().__init__(stripe_defs, names=stripe_ids)
        # save stripe parameters in the table
        self['x_gal_angl_min'] = xlims_arcsec[:-1] * u.arcsec
        self['x_gal_angl_max'] = xlims_arcsec[1:] * u.arcsec

        # record meta data in table
        self.meta['TBLTYPE'] = 'StripeMegaTable'
        self.meta['RA_DEG'] = gal_ra_deg
        self.meta['DEC_DEG'] = gal_dec_deg
        self.meta['PA_DEG'] = gal_posang_deg
        self.meta['XBIN_AS'] = xbin_arcsec
        self.meta['XMAX_AS'] = xmax_arcsec
        self.meta['YMAX_AS'] = ymax_arcsec

    # ----------------------------------------------------------------

    @classmethod
    def read(cls, filename, **kwargs):
        """
        Read (and reconstruct) a StripeMegaTable object from file.

        Parameters
        ----------
        filename : str
            Name of the file to read.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`

        Return
        ------
        table : StripeMegaTable
        """
        t = QTable.read(filename, **kwargs)

        # input file should be a valid StripeMegaTable output
        if 'TLBTYPE' not in t.meta:
            raise ValueError("Input file not recognized")
        if t.meta['TBLTYPE'] != 'StripeMegaTable':
            raise ValueError(
                "Cannot reconstruct an StripeMegaTable object from "
                f"the input {t.meta['TBLTYPE']} file")

        # initiate StripeMegaTable w/ recorded data in file
        mt = cls(
            t.meta['RA_DEG'], t.meta['DEC_DEG'], t.meta['PA_DEG'],
            t.meta['XBIN_AS'], t.meta['XMAX_AS'], t.meta['YMAX_AS'])

        # overwrite the underlying data table
        mt.table = QTable()
        for key in t.colnames:
            mt[key] = t[key]
        for key in t.meta:
            mt.meta[key] = t.meta[key]

        return mt


######################################################################
#
# User-defined Apertures
#
######################################################################


class ApertureMegaTable(GeneralRegionTable):

    """
    Mega-table built from apertures centered on specified locations.

    Each aperture corresponds to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each aperture.

    Parameters
    ----------
    aperture_ra_deg : array_like
        Right Ascension of the aperture centers (in degrees).
    aperture_dec_deg : array_like
        Declination of the aperture centers (in degrees).
    aperture_size_arcsec : float
        The angular size of the apertures (in arcseconds).
    aperture_names : list of str, optional
        Names of the apertures.
    """

    __name__ = "ApertureMegaTable"

    # ----------------------------------------------------------------

    def __init__(
            self, aperture_ra_deg, aperture_dec_deg,
            aperture_size_arcsec, aperture_names=None):

        # function that determines whether a set of coordinates locate
        # inside an aperture
        def coord2bool(
                ra, dec, aper_ra=None, aper_dec=None, aper_size=None):
            coords = SkyCoord(ra*u.deg, dec*u.deg)
            center_coord = SkyCoord(aper_ra*u.deg, aper_dec*u.deg)
            angl_sep = center_coord.separation(coords)
            return angl_sep < aper_size * u.arcsec

        # aperture names and definitions
        aper_defs = [
            partial(
                coord2bool, aper_ra=aper_ra, aper_dec=aper_dec,
                aper_size=aperture_size_arcsec)
            for aper_ra, aper_dec in zip(
                *np.broadcast_arrays(
                    aperture_ra_deg, aperture_dec_deg))
        ]
        if aperture_names is None:
            aperture_names = [
                f"Aper#{iaper+1}"
                for iaper in np.arange(len(aper_defs))]

        # initialize object
        super().__init__(aper_defs, names=aperture_names)
        # save aperture parameters in the table
        self['RA'] = aperture_ra_deg * u.deg
        self['DEC'] = aperture_dec_deg * u.deg

        # record meta data in table
        self.meta['APER_DEG'] = aperture_size_arcsec / 3600

    # ----------------------------------------------------------------

    @classmethod
    def read(cls, filename, **kwargs):
        """
        Read (and reconstruct) an ApertureMegaTable object from file.

        Parameters
        ----------
        filename : str
            Name of the file to read.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`

        Return
        ------
        table : ApertureMegaTable
        """
        t = QTable.read(filename, **kwargs)

        # input file should be a valid ApertureMegaTable output
        if 'TLBTYPE' not in t.meta:
            raise ValueError("Input file not recognized")
        if t.meta['TBLTYPE'] != 'ApertureMegaTable':
            raise ValueError(
                "Cannot reconstruct an ApertureMegaTable object from "
                f"the input {t.meta['TBLTYPE']} file")

        # initiate ApertureMegaTable w/ recorded data in file
        mt = cls(
            t['RA'].to('deg').value, t['DEC'].to('deg').value,
            t.meta['APER_DEG']*3600, aperture_names=t['REGION'])

        # overwrite the underlying data table
        mt.table = QTable()
        for key in t.colnames:
            mt[key] = t[key]
        for key in t.meta:
            mt.meta[key] = t.meta[key]

        return mt

    # ----------------------------------------------------------------

    def resample_image(
            self, image, ihdu=0, header=None,
            colname='new_col', unit='', suppress_error=False,
            fill_outside='nearest'):
        """
        Resample an image at the location of the aperture centers.

        The resampling is done by finding the nearest image pixel to
        each aperture center.

        Parameters
        ----------
        image : str, fits.HDUList, fits.HDU, or np.ndarray
            The image to be resampled.
        ihdu : int, optional
            If 'image' is a str or an HDUList, this keyword should
            specify which HDU (extension) to use (default=0)
        header : astropy.fits.Header, optional
            If 'image' is an ndarray, this keyword should be a FITS
            header providing the WCS information.
        colname : str, optional
            Name of a column in the table to save the output values.
            Default: 'new_col'
        unit : str or astropy.unit.Unit, optional
            Physical unit of the output values (default='').
            If unit='header', the 'BUNIT' entry in the header is used.
        fill_outside : {'nearest', float}, optional
            The behavior outside the footprint of the input image.
            If fill_outside='nearest', all seeds outside the footprint
            are assigned the value of the nearest pixel in the image.
            Otherwise, all aperture centers outside the image footprint
            are assigned the value of this keyword.
        suppress_error : bool, optional
            Whether to suppress the error message if 'image' looks
            like a file but is not found on disk (default=False)
        """

        data, hdr, wcs = reduce_image_input(
            image, ihdu, header, suppress_error=suppress_error)
        if data is None:
            if unit != 'header':
                self[colname] = (
                    np.full(len(self), np.nan) * u.Unit(unit))
            else:
                self[colname] = np.full(len(self), np.nan)
            return

        # find the nearest pixel for each aperture center
        iax0, iax1 = wcs.all_world2pix(
            self['RA'].value, self['DEC'].value, 0, ra_dec_order=True)
        iax0 = np.round(iax0).astype('int')
        iax1 = np.round(iax1).astype('int')
        mask = ((iax0 < 0) | (iax0 > wcs._naxis[0]-1) |
                (iax1 < 0) | (iax1 > wcs._naxis[1]-1))
        iax0[iax0 < 0] = 0
        iax0[iax0 > wcs._naxis[0]-1] = wcs._naxis[0]-1
        iax1[iax1 < 0] = 0
        iax1[iax1 > wcs._naxis[1]-1] = wcs._naxis[1]-1

        # get data from the nearest pixels
        sub_data = data[iax1, iax0]

        # assign values for apertures outside the image footprint
        if fill_outside != 'nearest':
            sub_data[mask] = fill_outside

        # save the resampled values as a new column in the table
        self[colname] = sub_data
        if unit == 'header':
            self[colname].unit = u.Unit(hdr['BUNIT'])
        else:
            if 'BUNIT' in hdr:
                if not identical_units(
                        u.Unit(hdr['BUNIT']), u.Unit(unit)):
                    warnings.warn(
                        "Specified unit is not the same as the unit "
                        "recorded in the FITS header")
            self[colname].unit = u.Unit(unit)


######################################################################
#
# Hexagonal/Rectangular/User-defined Tessellations
#
######################################################################


class TessellMegaTable(VoronoiTessTable):

    """
    Mega-table built from a tessellation of a galaxy's sky footprint.

    Each tile in the tessellation maps to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each tile, or
    by resampling images at the center of each tile.

    Parameters
    ----------
    center_ra_deg : float
        Right Ascension of the FoV center (in degrees).
        This also determines the location of the central tile.
    center_dec_deg : float
        Declination of the FoV center (in degrees).
        This also determines the location of the central tile.
    fov_radius_arcsec : float
        Radius of the FoV covered by the tessellation (in arcsecs).
        This determines the overall extent of the tessellation.
    tile_size_arcsec : float
        The angular size of the tiles (defined as the spacing between
        adjacent tile centers; in arcseconds).
    tile_shape : {'hexagon', 'square'}, optional
        The shape of the tiles that form the tessellation.
        Default: 'hexagon', in which case hexagonal tiles are used.
    """

    __name__ = "TessellMegaTable"

    # ----------------------------------------------------------------

    def __init__(
            self, center_ra_deg, center_dec_deg, fov_radius_arcsec,
            tile_size_arcsec, tile_shape='hexagon'):
        super().__init__(
            center_ra=center_ra_deg, center_dec=center_dec_deg,
            fov_radius=fov_radius_arcsec/3600,
            seed_spacing=tile_size_arcsec/3600,
            tile_shape=tile_shape)

        # record metadata
        self.meta['RA_DEG'] = self._center.ra.to('deg').value
        self.meta['DEC_DEG'] = self._center.dec.to('deg').value
        self.meta['FOV_DEG'] = self._fov.to('deg').value
        self.meta['TILE_DEG'] = tile_size_arcsec / 3600
        self.meta['TILE_DEF'] = tile_shape

    # ----------------------------------------------------------------

    @classmethod
    def read(cls, filename, **kwargs):
        """
        Read (and reconstruct) a TessellMegaTable object from file.

        Parameters
        ----------
        filename : str
            Name of the file to read.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`

        Return
        ------
        table : TessellMegaTable
        """
        t = QTable.read(filename, **kwargs)

        # input file should be a valid TessellMegaTable output
        if 'TBLTYPE' not in t.meta:
            raise ValueError("Input file not recognized")
        if t.meta['TBLTYPE'] != 'TessellMegaTable':
            raise ValueError(
                "Cannot reconstruct a TessellMegaTable object from "
                f"the input {t.meta['TBLTYPE']} file")

        # initiate TessellMegaTable
        mt = cls(
            t.meta['RA_DEG'], t.meta['DEC_DEG'],
            t.meta['FOV_DEG']*3600, t.meta['TILE_DEG']*3600,
            tile_shape=t.meta['TILE_DEF'])

        # overwrite the underlying data table
        mt.table = QTable()
        for key in t.colnames:
            mt[key] = t[key]
        for key in t.meta:
            mt.meta[key] = t.meta[key]

        return mt

    # ----------------------------------------------------------------

    def show_tiles_on_sky(
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
