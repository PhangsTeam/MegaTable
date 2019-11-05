import os
import warnings
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from .utils import identical_units

HDU_types = (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)


######################################################################
######################################################################


class HiddenTable(object):

    """
    Use protected attribute to hide an astropy table.

    Parameters
    ----------
    table : astropy.table.Table
        The table to hide
    """

    __name__ = "HiddenTable"

    def __str__(self):
        return self._table.__str__()

    def __repr__(self):
        return self._table.__repr__()

    def __len__(self):
        return len(self._table)

    def __getitem__(self, key):
        return self._table[key]

    def __setitem__(self, key, value):
        self._table[key] = value

    def __init__(self, table=None):
        if table is not None:
            self._table = table
        else:
            self._table = Table()

    def write(self, filename, keep_metadata=True, **kwargs):
        """
        Write the hidden table to a file.

        Parameters
        ----------
        filename : string
            Name of the file to write to.
        keep_metadata : bool, optional
            Whether to keep table meta data in the output file.
            Default is to keep.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.write`
        """
        if not keep_metadata:
            t = self._table.copy()
            for key in self._table.meta:
                t.meta.pop(key)
            t.write(filename, **kwargs)
        else:
            self._table.write(filename, **kwargs)


######################################################################
######################################################################


class StatsMixin(object):

    """
    Mixin class offering tools for calculating catalog/image stats.
    """

    #-----------------------------------------------------------------

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

        # find object coordinates in regions
        flagarr = self.find_coords_in_regions(ra, dec)

        # calculate (weighted) statistics within each region
        if not callable(stat_func):
            raise ValueError("Invalid input for 'stat_func'")
        arr = np.full(len(self), np.nan)
        for ind in range(len(self)):
            if weight is None:
                arr[ind] = stat_func(
                    entry.astype('float')[flagarr[:, ind]],
                    **kwargs)
            else:
                arr[ind] = stat_func(
                    entry.astype('float')[flagarr[:, ind]],
                    weights=weights.astype('float')[flagarr[:, ind]],
                    **kwargs)

        # save the output values as a new column in the table
        self._table[colname] = arr
        self._table[colname].unit = u.Unit(unit)

    #-----------------------------------------------------------------

    def _reduce_image_input(
            self, image, ihdu, header, suppress_error=False):
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

    #-----------------------------------------------------------------

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

        data, hdr, wcs = self._reduce_image_input(
            image, ihdu, header, suppress_error=suppress_error)
        if data is None:
            self._table[colname] = (
                np.full(len(self), np.nan) * u.Unit(unit))
            return
        
        # find pixels in cells
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


class GeneralRegionTable(StatsMixin, HiddenTable):

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
        - If that element is a function oject, it should take an
        RA array and a DEC array as the only input parameters, and
        return a boolean array specifying whether these coordinates
        are in the region.
    names : list of str, optional
        A list of names to be included as the 1st column in the table
        (should be concise and suffice to identify the regions).
    """

    __name__ = "GeneralRegionTable"

    #-----------------------------------------------------------------

    def __init__(
        self, region_defs, names=None):

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
        self._table['REGION'] = names
        self._table.meta['TBL_TYPE'] = self.__name__

    def find_coords_in_regions(self, ra, dec):
        """
        Find out which regions contain which input coordinates.

        Parameters
        ----------
        ra : array_like
            R.A. of the coordinates in question
        dec : array_like
            Declication of the coordinates in question

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


class VoronoiTessTable(StatsMixin, HiddenTable):

    """
    Table built from a Voronoi tessellation of (a part of) the sky.

    Each seed/cell in the Voronoi diagram maps to a row in the table.
    Once a table is constructed, additional columns can be added
    by calculating statistics of images within each Voronoi cell, or
    by resampling images at the location of each seed.

    Parameters
    ----------
    header : astropy.fits.Header
        FITS header that defines the footprint of the Voronoi diagram.
    seeds_ra : array_like, optional
        If the user would like to specify the seed locations,
        this would be their R.A. coordinates (in degrees)
    seeds_dec : array_like, optional
        If the user would like to specify the seed locations,
        this would be their Declination coordinates (in degrees)
    cell_shape : {'square', 'hexagon'}, optional
        If the seed locations are to be automatically generated,
        this keyword specifies the shape of the Voronoi cell.
        Default: 'square', in which case the Voronoi diagram comprises
        regularly spaced square cells.
    seed_spacing : float, optional
        If the seed locations are to be automatically generated,
        this keyword specifies the spacing between adjacent seeds.
        If None, the minimum absolute value of the 'CDELT' entry
        in the input header will be used.
    ref_radec : two-tuple, optional
        (RA, Dec) coordinate of the reference location (in degrees).
        As all calculations are done in the angular separation frame,
        this keyword defines the "origin" of this (dx, dy) coordinate.
        If None, the 'CRVAL' entry in the input header will be used.
    """

    __name__ = "VoronoiTessTable"

    #-----------------------------------------------------------------

    def __init__(
            self, header, seeds_ra=None, seeds_dec=None,
            cell_shape='square', seed_spacing=None, ref_radec=None):

        # if seed locations are specified, write them into the table
        if seeds_ra is not None and seeds_dec is not None:
            t = Table()
            t['RA'] = np.atleast_1d(seeds_ra)*u.deg
            t['DEC'] = np.atleast_1d(seeds_dec)*u.deg
            super().__init__(t)
            # extract celestial WCS information from header
            self._wcs = WCS(header).celestial
            if not self._wcs.axis_type_names == ['RA', 'DEC']:
                raise ValueError(
                    "Input header have unexpected axis type")
            # find the sky coordinate of the reference point
            if ref_radec is None:
                self._ref_coord = SkyCoord(*self._wcs.wcs.crval*u.deg)
            else:
                self._ref_coord = SkyCoord(*np.array(ref_radec)*u.deg)
            # record metadata
            self._table.meta['TBL_TYPE'] = self.__name__
            self._table.meta['CELL_TYP'] = 'user-defined'
            self._table.meta['SIZE_DEG'] = '-'
            self._table.meta['RA_DEG'] = self._ref_coord.ra.value
            self._table.meta['DEC_DEG'] = self._ref_coord.dec.value
            self._table.meta['NAXIS1'] = self._wcs._naxis[0]
            self._table.meta['NAXIS2'] = self._wcs._naxis[1]
            for key in self._wcs.to_header():
                self._table.meta[key] = self._wcs.to_header()[key]
            return

        # if seed locations are not specified, generate a list of
        # locations based on specified cell shape and seed spacing
        super().__init__()
        # extract celestial WCS information from header
        self._header = header
        self._wcs = WCS(self._header).celestial
        if not self._wcs.axis_type_names == ['RA', 'DEC']:
            raise ValueError(
                "Input header have unexpected axis type")
        # find the sky coordinate of the reference point
        if ref_radec is None:
            self._ref_coord = SkyCoord(*self._wcs.wcs.crval*u.deg)
        else:
            self._ref_coord = SkyCoord(*np.array(ref_radec)*u.deg)
        # use CDELT if no 'cell_spacing' value is provided
        if seed_spacing is None:
            spacing = np.min(np.abs(self._wcs.wcs.cdelt))*u.deg
        else:
            spacing = seed_spacing*u.deg
        # estimate the size of the entire Voronoi diagram
        footprint_world = self._wcs.calc_footprint()
        radius = []
        for coord in footprint_world:
            corner_coord = SkyCoord(*coord*u.deg)
            radius += [
                corner_coord.separation(
                    self._ref_coord).to('deg').value]
        if cell_shape == 'square':
            half_nx = half_ny = int(np.ceil(
                np.max(radius) / spacing.value))
        elif cell_shape == 'hexagon':
            half_nx = int(np.ceil(
                np.max(radius) / spacing.value))
            half_ny = int(np.ceil(
                np.max(radius) / spacing.value / np.sqrt(3)))
        else:
            raise ValueError(
                f"Unknown cell shape: {cell_shape}")
        # find RA & DEC coordinates for seeds
        iy, ix = np.mgrid[-half_ny:half_ny+1, -half_nx:half_nx+1]
        if cell_shape == 'square':
            ra_arr = (
                self._ref_coord.ra.value +
                ix.flatten() * spacing.value /
                np.cos(self._ref_coord.dec.to('rad').value))
            dec_arr = (
                self._ref_coord.dec.value +
                iy.flatten() * spacing.value)
        elif cell_shape == 'hexagon':
            ra_arr = (
                self._ref_coord.ra.value +
                np.concatenate([ix, ix+0.5], axis=None) *
                spacing.value /
                np.cos(self._ref_coord.dec.to('rad').value))
            dec_arr = (
                self._ref_coord.dec.value +
                np.concatenate([iy, iy+0.5], axis=None) *
                spacing.value * np.sqrt(3))
        self._table['RA'] = ra_arr * u.deg
        self._table['DEC'] = dec_arr * u.deg
        # discard cells outside the FITS header footprint
        iax0 = np.arange(self._wcs._naxis[0])
        iax1 = np.arange(self._wcs._naxis[1]).reshape(-1, 1)
        ramap, decmap = self._wcs.all_pix2world(
            iax0, iax1, 0, ra_dec_order=True)
        flagarr = self.find_coords_in_regions(ramap, decmap)
        self._table = self[flagarr.any(axis=0)]
        # record metadata
        self._table.meta['TBL_TYPE'] = self.__name__
        self._table.meta['CELL_TYP'] = cell_shape
        self._table.meta['SIZE_DEG'] = seed_spacing
        self._table.meta['RA_DEG'] = self._ref_coord.ra.value
        self._table.meta['DEC_DEG'] = self._ref_coord.dec.value
        self._table.meta['NAXIS1'] = self._wcs._naxis[0]
        self._table.meta['NAXIS2'] = self._wcs._naxis[1]
        for key in self._wcs.to_header():
            self._table.meta[key] = self._wcs.to_header()[key]

    #-----------------------------------------------------------------

    def find_coords_in_regions(self, ra, dec):
        """
        Find out which regions(cells) contain which input coordinates.

        Parameters
        ----------
        ra : array_like
            R.A. of the coordinates in question
        dec : array_like
            Declication of the coordinates in question

        Return
        ------
        flagarr : 2-D boolean array
            A boolean array indicating whether each cell contains
            each input coordinate. The shape of this array is:
            [# of coordinates, # of cells]
        """
        from scipy.spatial import cKDTree

        # identify reference coordinate
        radec_ctr = np.array(
            [self._ref_coord.ra.value, self._ref_coord.dec.value])
        scale_arr = np.array(
            [np.cos(self._ref_coord.dec.to('rad').value), 1])

        # find the angular offset coordinates for the seeds
        radec_seeds = np.stack(
            [self['RA'].quantity.value,
             self['DEC'].quantity.value], axis=-1)
        offset_seeds = (radec_seeds - radec_ctr) * scale_arr

        # find the angular offset coordinates for input locations
        radec_loc = np.stack([ra.flatten(), dec.flatten()], axis=-1)
        offset_loc = (radec_loc - radec_ctr) * scale_arr

        # find coordinates in cells
        kdtree = cKDTree(offset_seeds)
        _, indices = kdtree.query(offset_loc)
        flagarr = np.full([len(offset_loc), len(offset_seeds)], False)
        flagarr[np.arange(len(offset_loc)), indices] = True

        # "Unfind" all coordinates that are outside the footprint
        # of the FITS header used to generate the Voronoi diagram
        nax0, nax1 = self._wcs._naxis
        iax0, iax1 = self._wcs.all_world2pix(
            ra.flatten(), dec.flatten(), 0, ra_dec_order=False)
        mask = ((iax0 < 0) | (iax0 > nax0-1) |
                (iax1 < 0) | (iax1 > nax1-1))
        flagarr[mask, :] = False

        return flagarr

    #-----------------------------------------------------------------

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

        data, hdr, wcs = self._reduce_image_input(
            image, ihdu, header, suppress_error=suppress_error)
        if data is None:
            self._table[colname] = (
                np.full(len(self), np.nan) * u.Unit(unit))
            return

        # find nearest the nearest pixel for each seed location
        iax0, iax1 = wcs.all_world2pix(
            self['RA'].quantity.value,
            self['DEC'].quantity.value,
            0, ra_dec_order=True)
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
        self._table[colname] = sub_data
        if unit == 'header':
            self._table[colname].unit = u.Unit(hdr['BUNIT'])
        else:
            if 'BUNIT' in hdr:
                if not identical_units(
                        u.Unit(hdr['BUNIT']), u.Unit(unit)):
                    warnings.warn(
                        "Specified unit is not the same as the unit "
                        "recorded in the FITS header")
            self._table[colname].unit = u.Unit(unit)

    #-----------------------------------------------------------------

    def show_seeds_on_sky(
            self, ax=None, image=None, ffigkw={}, **scatterkw):
        """
        Show RA-Dec locations of the seeds on top of an image.

        ax : `~matplotlib.axes.Axes`, optional
            If 'image' is None, this is the Axes instance in which to
            make a scatter plot showing the seed locations.
        image : see below
            The image on which to overplot the seed locations.
            This will be passed to `aplpy.FITSFigure`.
        ffigkw : dict, optional
            Keyword arguments to be passed to `aplpy.FITSFigure`
        **scatterkw :
            Keyword arguments to be passed to `plt.scatter`
        """
        if image is not None:
            # show image using aplpy and overplot seed locations
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
