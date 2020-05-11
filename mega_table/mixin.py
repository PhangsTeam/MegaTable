from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
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

        if not (Path(bm0file).is_file() and
                Path(sm0file).is_file() and
                Path(sewfile).is_file()):
            if suppress_error:
                # add placeholder (empty) columns
                for col, unit in cols:
                    self[col] = (
                        np.full(len(self), np.nan) * u.Unit(unit))
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
                    self[col] = (
                        np.full(len(self), np.nan) * u.Unit(unit))
                return
            else:
                raise ValueError("Input file not found")

        # read CPROPS file
        t_cat = Table.read(cpropsfile)
        ra_cat = np.array(t_cat['XCTR_DEG'])
        dec_cat = np.array(t_cat['YCTR_DEG'])
        flux_cat = (
            t_cat['FLUX_KKMS_PC2'] / t_cat['DISTANCE_PC']**2 *
            u.Unit('K km s-1 sr')).to('K km s-1 arcsec2').value
        sigv_cat = np.array(t_cat['SIGV_KMS'])
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
