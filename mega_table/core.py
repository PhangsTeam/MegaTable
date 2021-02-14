import time
import warnings
import numpy as np
from scipy.spatial import cKDTree
from astropy import units as u
from astropy.table import QTable
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from .utils import identical_units, reduce_image_input, HDU_types


######################################################################
######################################################################


class BaseTable(object):

    """
    A simple wrapper around `~astropy.table.QTable`.

    Parameters
    ----------
    table : `~astropy.table.QTable`
        Core table
    """

    __name__ = "BaseTable"

    def __init__(self, table=None):
        if table is not None:
            self.table = QTable(table)
        else:
            self.table = QTable()

    def __str__(self):
        return self.table.__str__()

    def __repr__(self):
        return self.table.__repr__()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, value):
        self.table[key] = value

    @property
    def colnames(self):
        return self.table.colnames

    @property
    def info(self):
        return self.table.info

    @property
    def meta(self):
        return self.table.meta

    def write(
            self, filename, keep_metadata=True, add_timestamp=True,
            **kwargs):
        """
        Write table to file.

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
            for key in self.meta:
                try:
                    hdr[key] = t.meta[key]
                except ValueError:
                    t.meta.pop(key)
        if 'TIMESTMP' in t.meta:
            # remove previous time stamp
            t.meta.pop('TIMESTMP')
        if add_timestamp:
            # add current time stamp
            t.meta['TIMESTMP'] = time.strftime('%c', time.gmtime())
        t.write(filename, **kwargs)


######################################################################
######################################################################


class StatsTable(BaseTable):

    """
    Table class offering tools for calculating catalog/image stats.
    """

    # ----------------------------------------------------------------

    def find_coords_in_regions(self, ra, dec):
        """
        Placeholder function (to be overwritten by descendant classes)

        Parameters
        ----------
        ra : array_like
            R.A. of the coordinates in question
        dec : array_like
            Declination of the coordinates in question

        Return
        ------
        flagarr : 2-D boolean array
            A boolean array indicating whether each region contains
            each input coordinate. The shape of this array is:
            [# of coordinates, # of regions]
        """
        return np.full([len(ra), len(self)], False)

    # ----------------------------------------------------------------

    def calc_catalog_stats(
            self, entry, ra, dec, stat_func=None, weight=None,
            colname='new_col', unit='', **kwargs):
        """
        Calculate statistics of a catalog entry within each region.

        "Catalog" in this context means a table recording both the
        (ra, dec) coordinates and other properties of a list of
        identified objects on the sky. Typical examples: molecular
        cloud catalogs, HII region catalogs, stellar cluster catalogs.

        An example catalog entry: the "molecular cloud size" column
        in a cloud catalog.

        Parameters
        ----------
        entry : np.ndarray
            The catalog entry in question.
        ra : np.ndarray
            RA coordinates of the listed objects.
        dec : np.ndarray
            Dec coordinates of the listed objects.
        stat_func : callable
            A function that accepts an array of values, and return a
            scalar value (which is the calculated statistics). If
            'weight' is not None, this function should also accept a
            keyword named 'weights', which specifies the statistical
            weight of each value in the array.
        weight : np.ndarray, optional
            If not None, this keyword should be an ndarray specifying
            the statistical weight of each row in the catalog.
            Note that in this case, the broadcasted weight array will
            be passed to 'stat_func' in a keyword named 'weights'.
        colname : str, optional
            Name of a column in the table to save the output values.
            Default: 'new_col'
        unit : str or astropy.unit.Unit, optional
            Physical unit of the output values (default='').
        **kwargs
            Keyword arguments to be passed to 'stat_func'
        """
        if weight is not None:
            weights = np.broadcast_to(weight, entry.shape)
        else:
            weights = None

        # find object coordinates in regions
        findarr = self.find_coords_in_regions(ra, dec)

        # calculate (weighted) statistics within each region
        if not callable(stat_func):
            raise ValueError("Invalid input for 'stat_func'")
        arr = np.full(len(self), np.nan)
        for ind in range(len(self)):
            if findarr.ndim == 2:  # 2D boolean flag
                flagarr = findarr[:, ind]
            else:  # 1D index array
                flagarr = (findarr == ind)
            if weight is None:
                arr[ind] = stat_func(
                    entry.astype('float')[flagarr],
                    **kwargs)
            else:
                arr[ind] = stat_func(
                    entry.astype('float')[flagarr],
                    weights=weights.astype('float')[flagarr],
                    **kwargs)

        # save the output values as a new column in the table
        self[colname] = arr
        self[colname].unit = u.Unit(unit)

    # ----------------------------------------------------------------

    def calc_image_stats(
            self, image, ihdu=0, header=None,
            stat_func=None, weight=None,
            colname='new_col', unit='', suppress_error=False,
            **kwargs):
        """
        Calculate statistics of an image within each region.

        Parameters
        ----------
        image : str, fits.HDUList, fits.HDU, or np.ndarray
            The image to calculate statistics for.
        ihdu : int, optional
            If 'image' is a str or an HDUList, this keyword should
            specify which HDU (extension) to use (default=0)
        header : astropy.fits.Header, optional
            If 'image' is an ndarray, this keyword should be a FITS
            header providing the WCS information.
        stat_func : callable
            A function that accepts an array of values, and return a
            scalar value (which is the calculated statistics). If
            'weight' is not None, this function should also accept a
            keyword named 'weights', which specifies the statistical
            weight of each value in the array.
        weight : np.ndarray, optional
            If not None, this keyword should be an ndarray specifying
            the statistical weight of each pixel in the input image.
            Note that in this case, the broadcasted weight array will
            be passed to 'stat_func' in a keyword named 'weights'.
        colname : str, optional
            Name of a column in the table to save the output values.
            Default: 'new_col'
        unit : str or astropy.unit.Unit, optional
            Physical unit of the output values (default='').
            If unit='header', the 'BUNIT' entry in the header is used.
        suppress_error : bool, optional
            Whether to suppress the error message if 'image' looks
            like a file but is not found on disk (default=False)
        **kwargs
            Keyword arguments to be passed to 'stat_func'
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

        # generate RA & Dec arrays
        iax0 = np.arange(wcs._naxis[0])
        iax1 = np.arange(wcs._naxis[1]).reshape(-1, 1)
        ramap, decmap = wcs.all_pix2world(
            iax0, iax1, 0, ra_dec_order=True)

        if weight is not None:
            w = np.broadcast_to(weight, data.shape).flatten()
        else:
            w = None

        if unit == 'header':
            u_data = u.Unit(hdr['BUNIT'])
        else:
            if 'BUNIT' in hdr:
                if not identical_units(
                        u.Unit(hdr['BUNIT']), u.Unit(unit)):
                    warnings.warn(
                        "Specified unit is not the same as the unit "
                        "recorded in the FITS header")
            u_data = u.Unit(unit)

        self.calc_catalog_stats(
            data.flatten(), ramap.flatten(), decmap.flatten(),
            stat_func=stat_func, weight=w,
            colname=colname, unit=u_data, **kwargs)


######################################################################
######################################################################


class GeneralRegionTable(StatsTable):

    """
    Table build from a set of user-defined regions on the sky.

    Each region corresponds to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each region.

    Note that the regions can overlap with each other.

    Parameters
    ----------
    region_defs : a list of FITS HDU objects and/or functions
        This parameter defines the sky regions that comprise the
        table, with each element in the list corresponds to a region.
        - If that element is an HDU object, it should contain a bitmap
        that defines the corresponding region (1=in, 0=out).
        - If that element is a function object, it should take an
        RA array and a DEC array as the only input parameters, and
        return a boolean array specifying whether these coordinates
        are in the region.
    names : list of str, optional
        A list of names to be included as the 1st column in the table
        (should be concise and suffice to identify the regions).
    """

    __name__ = "GeneralRegionTable"

    # ----------------------------------------------------------------

    def __init__(self, region_defs, names=None):

        # verify the region definitions
        for ireg, reg_def in enumerate(region_defs):
            if (not isinstance(reg_def, HDU_types) and
                    not callable(reg_def)):
                raise ValueError(
                    f"Invalid region definition: #{ireg}")

        # initialize table
        super().__init__()
        self._region_defs = region_defs
        if names is None:
            names = [f"REGION#{i}" for i in range(len(region_defs))]
        self['REGION'] = names
        self.meta['TBLTYPE'] = self.__name__

    def find_coords_in_regions(self, ra, dec):
        """
        Find out which regions contain which input coordinates.

        Parameters
        ----------
        ra : array_like
            R.A. of the coordinates in question
        dec : array_like
            Declination of the coordinates in question

        Return
        ------
        flagarr : 2-D boolean array
            A boolean array indicating whether each region contains
            each input coordinate. The shape of this array is:
            [# of coordinates, # of regions]
        """
        # initialize flag array
        flagarr = np.full([len(ra), len(self)], False)

        # loop over region definitions
        for ireg, reg_def in enumerate(self._region_defs):
            if isinstance(reg_def, HDU_types):
                # find the nearest pixels for the input coordinates
                wcs = WCS(reg_def.header)
                iax0, iax1 = wcs.all_world2pix(
                    ra, dec, 0, ra_dec_order=True)
                iax0 = np.round(iax0).astype('int')
                iax1 = np.round(iax1).astype('int')
                # find coordinates outside the WCS footprint
                mask = ((iax0 < 0) | (iax0 > wcs._naxis[0]-1) |
                        (iax1 < 0) | (iax1 > wcs._naxis[1]-1))
                iax0[iax0 < 0] = 0
                iax0[iax0 > wcs._naxis[0]-1] = wcs._naxis[0]-1
                iax1[iax1 < 0] = 0
                iax1[iax1 > wcs._naxis[1]-1] = wcs._naxis[1]-1
                # find bitmap values in the nearest pixels
                flag = reg_def.data[iax1, iax0].astype('?')
                # mask coordinates outside WCS footprint
                flag[mask] = False
                flagarr[:, ireg] = flag
            else:
                # insert RA & DEC into function and get boolean array
                flagarr[:, ireg] = reg_def(ra, dec)

        return flagarr


######################################################################
######################################################################


class VoronoiTessTable(StatsTable):

    """
    Table built from a Voronoi tessellation of (a part of) the sky.

    Each seed/tile in the Voronoi diagram maps to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each Voronoi tile, or
    by resampling images at the location of each seed.

    Angular separations are calculated with small angle approximation.

    Parameters
    ----------
    center_ra : float
        Right Ascension of the FoV center (in degrees)
    center_dec : float
        Declination of the FoV center (in degrees)
    fov_radius : float
        Radius of the FoV to cover with the tessellation (in degrees)
    seeds_ra : array_like, optional
        Right Ascension of user-defined seed locations (in degrees)
    seeds_dec : array_like, optional
        Declination of user-defined seed locations (in degrees)
    seed_spacing : float, optional
        If the seed locations are to be automatically generated,
        this keyword specifies the spacing between adjacent seeds
        (in degrees).
    tile_shape : {'square', 'hexagon'}, optional
        If the seed locations are to be automatically generated,
        this keyword specifies the shape of the Voronoi tile.
        Default: 'square', in which case the Voronoi diagram comprises
        regularly spaced square tiles.
    """

    __name__ = "VoronoiTessTable"

    # ----------------------------------------------------------------

    def __init__(
            self, center_ra, center_dec, fov_radius,
            seeds_ra=None, seeds_dec=None,
            seed_spacing=None, tile_shape='square'):

        # if seed locations are specified, write them into the table
        if seeds_ra is not None and seeds_dec is not None:
            ras, decs = np.broadcast_arrays(seeds_ra, seeds_dec)
            t = QTable()
            t['RA'] = ras * u.deg
            t['DEC'] = decs * u.deg
            super().__init__(t)
            self._center = SkyCoord(
                center_ra * u.deg, center_dec * u.deg)
            self._fov = fov_radius * u.deg
            self.meta['TBLTYPE'] = self.__name__
            return

        # if seed locations are not specified, generate a list of
        # locations based on specified tile shape and seed spacing
        super().__init__()
        self._center = SkyCoord(center_ra * u.deg, center_dec * u.deg)
        self._fov = fov_radius * u.deg
        cos_dec = np.cos(self._center.dec.to('rad').value)
        if seed_spacing is None:
            raise ValueError("Missing seed spacing information")
        else:
            spacing = seed_spacing * u.deg
        if tile_shape == 'square':
            half_nx = half_ny = int(np.ceil(
                (self._fov / spacing).to('').value)) + 1
        elif tile_shape == 'hexagon':
            half_nx = int(np.ceil(
                (self._fov / spacing).to('').value)) + 1
            half_ny = int(np.ceil(
                (self._fov / spacing).to('').value / np.sqrt(3))) + 1
        else:
            raise ValueError(
                f"Unknown tile shape: {tile_shape}")
        # find RA & DEC coordinates for seeds
        iy, ix = np.mgrid[-half_ny:half_ny+1, -half_nx:half_nx+1]
        if tile_shape == 'square':
            ra_arr = (
                self._center.ra + ix.flatten() * spacing / cos_dec
            ).to('deg')
            dec_arr = (
                self._center.dec + iy.flatten() * spacing
            ).to('deg')
        else:
            ra_arr = (
                self._center.ra +
                np.concatenate([ix, ix+0.5], axis=None) *
                spacing / cos_dec
            ).to('deg')
            dec_arr = (
                self._center.dec +
                np.concatenate([iy, iy+0.5], axis=None) *
                spacing * np.sqrt(3)
            ).to('deg')
        self['RA'] = ra_arr
        self['DEC'] = dec_arr
        # discard tiles outside the FoV radius
        sep = np.sqrt(
            (self['RA'] - self._center.ra)**2 * cos_dec**2 +
            (self['DEC'] - self._center.dec)**2)
        self.table = self[sep <= self._fov + spacing/np.sqrt(3)]
        self.meta['TBLTYPE'] = self.__name__

    # ----------------------------------------------------------------

    def find_coords_in_regions(self, ra, dec, fill_value=-1):
        """
        Find out which regions(tiles) contain which input coordinates.

        Parameters
        ----------
        ra : array_like
            R.A. of the coordinates in question
        dec : array_like
            Declination of the coordinates in question
        fill_value : float, optional
            The index value to return for input coordinates that have
            no matched regions (default: -1).

        Return
        ------
        indices : 1-D index array
            An index array indicating which tile each input coordinate
            belongs to. The length of this array equals the number of
            input coordinates.
        """

        # identify reference coordinate
        radec_ctr = np.array(
            [self._center.ra.value, self._center.dec.value])
        scale_arr = np.array(
            [np.cos(self._center.dec.to('rad').value), 1])

        # find the angular offset coordinates for the seeds
        radec_seeds = np.stack(
            [self['RA'].value, self['DEC'].value], axis=-1)
        offset_seeds = (radec_seeds - radec_ctr) * scale_arr

        # find the angular offset coordinates for input locations
        radec_loc = np.stack([ra.flatten(), dec.flatten()], axis=-1)
        offset_loc = (radec_loc - radec_ctr) * scale_arr

        # find coordinates in tiles
        kdtree = cKDTree(offset_seeds)
        _, indices = kdtree.query(offset_loc)

        # for all coordinates outside the FoV, overwrite
        # their matched indices with "fill_value"
        mask = np.sqrt(np.sum(offset_loc**2, axis=-1)) > self._fov
        indices[mask] = fill_value

        return indices

    # ----------------------------------------------------------------

    def resample_image(
            self, image, ihdu=0, header=None,
            colname='new_col', unit='', suppress_error=False,
            fill_outside='nearest'):
        """
        Resample an image at the location of the Voronoi seeds.

        The resampling is done by finding the nearest pixel to the
        location of each seed in the image.

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
            Otherwise, all seeds outside the image footprint are
            assigned the value of this keyword.
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

        # find nearest the nearest pixel for each seed location
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

        # get nearest pixel value
        sub_data = data[iax1, iax0]

        # assign values for seed locations outside the image footprint
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
######################################################################
