import os
import warnings
import numpy as np
from scipy.spatial import cKDTree
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


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
        if discard_NaN is not None:
            for col in np.atleast_1d(discard_NaN):
                self._table = self[~np.isnan(self[col])]
        if keep_finite is not None:
            for col in np.atleast_1d(keep_finite):
                self._table = self[np.isfinite(self[col])]
        if discard_negative is not None:
            for col in np.atleast_1d(discard_negative):
                self._table = self[~(self[col] < 0)]
        if keep_positive is not None:
            for col in np.atleast_1d(keep_positive):
                self._table = self[self[col] > 0]

    def write(self, filename, **kwargs):
        """
        Write the hidden table to a file.

        Parameters
        ----------
        filename : string
            Name of the file to write to.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.write`
        """
        self._table.write(filename, **kwargs)

    @classmethod
    def read(cls, filename, **kwargs):
        """
        Read table from a file and hide it in the protected attribute.

        Parameters
        ----------
        filename : string
            Name of the file to read.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`
        """
        return cls(table=Table.read(filename, **kwargs))


######################################################################
######################################################################


class VoronoiTessTable(HiddenTable):

    """
    Table built from a Voronoi tessellation of an image footprint.

    Each seed/cell in the Voronoi diagram maps to a row in the table.

    Once a table is constructed, additional columns can be added
    by resampling images at the location of each seed, or
    by calculating statistics of images within each Voronoi cell.

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
    ref_radec : two_tuple, optional
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
            self._table.meta['TBLTYPE'] = 'VoronoiTessTable'
            for key in self._wcs.to_header():
                self._table.meta[key] = self._wcs.to_header()[key]
            self._table.meta['NAXIS1'] = self._wcs._naxis[0]
            self._table.meta['NAXIS2'] = self._wcs._naxis[1]
            self._table.meta['REF-RA'] = self._ref_coord.ra.value
            self._table.meta['REF-DEC'] = self._ref_coord.dec.value
            self._table.meta['TESSTYPE'] = 'USER-DEFINED'
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
        ix = np.arange(self._wcs._naxis[0])
        iy = np.arange(self._wcs._naxis[1]).reshape(-1, 1)
        ramap, decmap = self._wcs.all_pix2world(
            ix, iy, 0, ra_dec_order=True)
        indices = self.find_locs_in_cells(ramap, decmap)
        mask = np.zeros(len(self)).astype('?')
        for ind in range(len(self)):
            if (indices == ind).sum() > 0:
                mask[ind] = True
        self._table = self[mask]
        # record metadata
        self._table.meta['TBLTYPE'] = 'VoronoiTessTable'
        for key in self._wcs.to_header():
            self._table.meta[key] = self._wcs.to_header()[key]
        self._table.meta['NAXIS1'] = self._wcs._naxis[0]
        self._table.meta['NAXIS2'] = self._wcs._naxis[1]
        self._table.meta['REF-RA'] = self._ref_coord.ra.value
        self._table.meta['REF-DEC'] = self._ref_coord.dec.value
        self._table.meta['TESSTYPE'] = (
            f"AUTO-GENERATED ({cell_shape.upper()} CELLS)")

    #-----------------------------------------------------------------

    def find_locs_in_cells(self, ra, dec):
        """
        Find out which cells do the input locations belong to.

        Parameters
        ----------
        ra : array_like
            R.A. of the locations in question
        dec : array_like
            Declication of the locations in question

        Return
        ------
        indices : array_like
            An array of the indices of cells each location belonds to.
            If any location is outside the footprint of the Voronoi
            diagram, the corresponding index is set to -1.
        """
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

        # find locations in cells
        kdtree = cKDTree(offset_seeds)
        _, indices = kdtree.query(offset_loc)

        # if any location is outside the footprint of the FITS header
        # used to generate the diagram, overwrite its cell index to -1
        nx, ny = self._wcs._naxis
        ix, iy = self._wcs.all_world2pix(
            ra.flatten(), dec.flatten(), 0, ra_dec_order=False)
        indices[(ix < 0) | (ix > nx-1) | (iy < 0) | (iy > ny-1)] = -1

        return indices.reshape(ra.shape)

    #-----------------------------------------------------------------

    def _reduce_input(self, image, ihdu, header):
        """
        Reduce any combination of input values to (data, header, wcs).

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

        Returns
        -------
        data : np.ndarray
        hdr : fits.Header object
        wcs : astropy.wcs.WCS object
        """
        if isinstance(image, np.ndarray):
            data = image
            hdr = header
        elif isinstance(image, tuple([fits.PrimaryHDU, fits.ImageHDU,
                                      fits.CompImageHDU])):
            data = image.data
            hdr = image.header
        elif isinstance(image, fits.HDUList):
            data = np.copy(image[ihdu].data)
            hdr = image[ihdu].header.copy()
        else:
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

    def resample_image(
            self, image, ihdu=0, header=None,
            colname='new_col', unit='',
            fill_outside='nearest'):
        """
        Resample an image at the location of the Voronoi seeds.

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
        """
        data, hdr, wcs = self._reduce_input(image, ihdu, header)

        # find nearest the nearest pixel for each seed location
        ix, iy = wcs.all_world2pix(
            self['RA'].quantity.value,
            self['DEC'].quantity.value,
            0, ra_dec_order=True)
        ix = np.round(ix).astype('int')
        iy = np.round(iy).astype('int')
        mask = ((ix < 0) | (ix > wcs._naxis[0]-1) |
                (iy < 0) | (iy > wcs._naxis[1]-1))
        ix[ix < 0] = 0
        ix[ix > wcs._naxis[0]-1] = wcs._naxis[0]-1
        iy[iy < 0] = 0
        iy[iy > wcs._naxis[1]-1] = wcs._naxis[1]-1

        # get nearest pixel value
        sub_data = data[iy, ix]

        # assign values for seed locations outside the image footprint
        if fill_outside != 'nearest':
            sub_data[mask] = fill_outside

        # save the resampled values as a new column in the table
        self._table[colname] = sub_data
        if unit == 'header':
            self._table[colname].unit = u.Unit(hdr['BUNIT'])
        else:
            self._table[colname].unit = u.Unit(unit)

    #-----------------------------------------------------------------

    def _nanaverage(self, a, weights=None, **kwargs):
        if weights is None:
            w = np.ones_like(a)
        else:
            w = weights.astype('float')
        flag = ~np.isnan(a) & ~np.isnan(w)
        if flag.sum() == 0:
            return np.nan
        else:
            return np.average(a[flag], weights=w[flag], **kwargs)

    #-----------------------------------------------------------------

    def calc_image_stats(
            self, image, ihdu=0, header=None, weight=None,
            colname='new_col', unit='',
            stat_func=None, **kwargs):
        """
        Calculate statistics of an image within each cell.

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
        stat_func : callable, optional
            A function that accepts an array of values, and return a
            scalar value (which is the calculated statistics). If
            'weight' is not None, this function should also accept a
            keyword named 'weights', which specifies the statistical
            weight of each value in the array.
        **kwargs
            Keyword arguments to be passed to 'stat_func'
        """
        data, hdr, wcs = self._reduce_input(image, ihdu, header)
        if weight is not None:
            weights = np.broadcast_to(weight, data.shape)

        # find pixels in cells
        ix = np.arange(wcs._naxis[0])
        iy = np.arange(wcs._naxis[1]).reshape(-1, 1)
        ramap, decmap = wcs.all_pix2world(
            ix, iy, 0, ra_dec_order=True)
        indices = self.find_locs_in_cells(ramap, decmap)

        # calculate weighted statistics within each cell
        if stat_func is None:
            func = self._nanaverage
        else:
            func = stat_func
        arr = np.full(len(self), np.nan)
        for ind in range(len(self)):
            if weight is None:
                arr[ind] = func(
                    data.astype('float')[indices == ind],
                    **kwargs)
            else:
                arr[ind] = func(
                    data.astype('float')[indices == ind],
                    weights=weights.astype('float')[indices == ind],
                    **kwargs)

        # save the summed values as a new column in the table
        self._table[colname] = arr
        if unit == 'header':
            self._table[colname].unit = u.Unit(hdr['BUNIT'])
        else:
            self._table[colname].unit = u.Unit(unit)

    #-----------------------------------------------------------------

    def calc_catalog_stats(
            self, value, ra, dec, weight=None,
            colname='new_col', unit='',
            stat_func=None, **kwargs):
        """
        Calculate statistics of a catalog entry within each cell.

        Parameters
        ----------
        value : np.ndarray
            Values in the catalog entry.
        ra : np.ndarray
            RA coordinates corresponding to each row of the catalog.
        dec : np.ndarray
            Dec coordinates corresponding to each row of the catalog.
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
        stat_func : callable, optional
            A function that accepts an array of values, and return a
            scalar value (which is the calculated statistics). If
            'weight' is not None, this function should also accept a
            keyword named 'weights', which specifies the statistical
            weight of each value in the array.
        **kwargs
            Keyword arguments to be passed to 'stat_func'
        """
        if weight is not None:
            weights = np.broadcast_to(weight, data.shape)

        # find rows in cells
        indices = self.find_locs_in_cells(ra, dec)

        # calculate weighted statistics within each cell
        if stat_func is None:
            func = self._nanaverage
        else:
            func = stat_func
        arr = np.full(len(self), np.nan)
        for ind in range(len(self)):
            if weight is None:
                arr[ind] = func(
                    value.astype('float')[indices == ind],
                    **kwargs)
            else:
                arr[ind] = func(
                    value.astype('float')[indices == ind],
                    weights=weights.astype('float')[indices == ind],
                    **kwargs)

        # save the summed values as a new column in the table
        self._table[colname] = arr
        self._table[colname].unit = u.Unit(unit)

    #-----------------------------------------------------------------

    def write(self, filename, keep_metadata=True, **kwargs):
        """
        Write a VoronoiTessTable to a file.

        Parameters
        ----------
        filename : string
            Name of the file to write to.
        keep_metadata : bool, optional
            Whether to keep table meta data in the output file.
            Default is to keep, which allows one to reconstruct
            the VoronoiTessTable by reading the file with
            `VoronoiTessTable.read`. Also when keep_metadata=True,
            output format must be 'ascii.ecsv'.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.write`
        """
        if not keep_metadata:
            t = self._table.copy()
            t.meta = OrderedDict()
            t.write(filename, **kwargs)
        else:
            if 'format' in kwargs:
                if kwargs['format'] != 'ascii.ecsv':
                    warnings.warn(
                        "Overwrite keyword 'format' to 'ascii.ecsv' "
                        "when keep_metadata=True")
            kwargs['format'] = 'ascii.ecsv'
            self._table.write(filename, **kwargs)

    #-----------------------------------------------------------------

    @classmethod
    def read(cls, filename, **kwargs):
        """
        Read a VoronoiTessTable from a file.

        Parameters
        ----------
        filename : str
            Name of the file to read from.
        **kwargs
            Keyword arguments to be passed to `~astropy.table.read`

        Return
        ------
        table : VonoroiTessTable
        """

        if 'format' in kwargs:
            if kwargs['format'] != 'ascii.ecsv':
                warnings.warn(
                    "Overwrite keyword 'format' to 'ascii.ecsv' "
                    "when reading VonoroiTessTable from file.")
        kwargs['format'] = 'ascii.ecsv'
        t = Table.read(filename, **kwargs)
        if not 'TBLTYPE' in t.meta:
            raise ValueError("Input file not recognized")
        if t.meta['TBLTYPE'] != 'VoronoiTessTable':
            raise ValueError("Input file not recognized")

        vtt = cls(
            fits.Header(t.meta), seeds_ra=[], seeds_dec=[],
            ref_radec=(t.meta['REF-RA'], t.meta['REF-DEC']))
        vtt._table = t

        return vtt

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


def deproject(
        center_coord=None, incl=0*u.deg, pa=0*u.deg,
        header=None, wcs=None, naxis=None, ra=None, dec=None,
        return_offset=False):

    """
    Calculate deprojected radii and projected angles in a disk.

    This function deals with projected images of astronomical objects
    with an intrinsic disk geometry. Given sky coordinates of the
    disk center, disk inclination and position angle, this function
    calculates deprojected radii and projected angles based on
    (1) a FITS header (`header`), or
    (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
    (3) RA and DEC coodinates (`ra` + `dec`).
    Both deprojected radii and projected angles are defined relative
    to the center in the inclined disk frame. For (1) and (2), the
    outputs are 2D images; for (3), the outputs are arrays with shapes
    matching the broadcasted shape of `ra` and `dec`.

    Parameters
    ----------
    center_coord : `~astropy.coordinates.SkyCoord` object or 2-tuple
        Sky coordinates of the disk center
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
