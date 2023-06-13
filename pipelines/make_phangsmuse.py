import sys
import json
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
# from astropy import uncertainty as unc
from astropy.io import fits
from astropy.table import Table

from mega_table.core import StatsTable
from mega_table.table import TessellMegaTable, RadialMegaTable
from mega_table.utils import nanaverage, nanrms

###############################################################################

# location of all relevant config files
config_dir = Path('/data/kant/0/sun.1608/PHANGS/mega-tables/code')

# location to save the output data tables
work_dir = Path('/data/kant/0/sun.1608/PHANGS/mega-tables')

# logging setting
logging = False

###############################################################################


class PhangsMuseMegaTable(StatsTable):

    """
    MegaTable for PHANGS-MUSE data.
    """

    # -------------------------------------------------------------------------
    # methods for nebulae statistics
    # -------------------------------------------------------------------------

    def add_hii_count(
            self, colname='N_HII', unit='',
            ra=None, dec=None, flux_Ha=None):
        if ra is None or dec is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        self.calc_catalog_stats(
            np.isfinite(flux_Ha).astype('int'), ra, dec,
            stat_func=np.nansum, colname=colname, unit=unit)

    def add_hii_stat_generic(
            self, colname=None, colname_e=None, unit=None,
            ra=None, dec=None, quantity=None, e_quantity=None,
            weight=None, e_weight=None):
        # check input
        if ra is None or dec is None:
            self[colname] = np.nan * u.Unit(unit)
            if colname_e is not None:
                self[colname_e] = np.nan * u.Unit(unit)
            return
        # clean up weight array
        if weight is not None:
            wt = weight.value.copy()
            wt[~np.isfinite(weight)] = 0
            e_wt = e_weight.value.copy()
            e_wt[~np.isfinite(weight)] = 0
        else:
            wt = e_wt = None
        # calculate (weighted) average for the quantity of interest
        self.calc_catalog_stats(
            quantity.to(unit).value, ra, dec, weight=wt,
            stat_func=nanaverage, colname=colname, unit=unit)
        if colname_e is None:
            return
        # calculate formal uncertainty on the (weighted) averaged value
        if weight is None:
            self.calc_catalog_stats(
                e_quantity.value**2, ra, dec, stat_func=np.nansum,
                colname='_sqsum(err)', unit=e_quantity.unit**2)
            self[colname_e] = (
                np.sqrt(self['_sqsum(err)']) / self['N_HII']).to(unit)
            self.table.remove_column('_sqsum(err)')
        else:
            findarr = self.find_coords_in_regions(ra, dec)
            if findarr.ndim == 2:
                avg = np.full(len(ra), np.nan) * self[colname].unit
                for ireg in range(len(self)):
                    avg[findarr[:, ireg]] = self[colname][ireg]
            else:
                avg = self[colname][findarr]
                avg[findarr == -1] = np.nan
            self.calc_catalog_stats(
                ((quantity - avg)**2 * e_wt**2 +
                 wt**2 * e_quantity**2).value,
                ra, dec, stat_func=np.nansum,
                colname='_sqsum(err)', unit=quantity.unit**2)
            self.calc_catalog_stats(
                wt, ra, dec, stat_func=np.nansum,
                colname='_sum(wt)', unit='')
            self[colname_e] = (
                np.sqrt(self['_sqsum(err)']) / self['_sum(wt)']).to(unit)
            self.table.remove_columns(['_sqsum(err)', '_sum(wt)'])

    def add_hii_ha_flux(
            self, colname='F_Halpha_HII', unit='erg s-1 cm-2',
            colname_e='e_F_Halpha_HII', unit_e='erg s-1 cm-2',
            ra=None, dec=None, flux_Ha=None, e_flux_Ha=None):
        self.add_hii_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            ra=ra, dec=dec, quantity=flux_Ha, e_quantity=e_flux_Ha)
        # convert uncertainty unit
        self[colname_e] = self[colname_e].to(unit_e)

    def add_hii_ha_flux_weighted(
            self, colname='<F_Halpha_HII>', unit='erg s-1 cm-2',
            colname_e='e_<F_Halpha_HII>', unit_e='erg s-1 cm-2',
            ra=None, dec=None, flux_Ha=None, e_flux_Ha=None):
        self.add_hii_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            ra=ra, dec=dec, quantity=flux_Ha, e_quantity=e_flux_Ha,
            weight=flux_Ha, e_weight=e_flux_Ha)
        # convert uncertainty unit
        self[colname_e] = self[colname_e].to(unit_e)

    def add_hii_metallicity(
            self, colname='Zprime_HII', unit='',
            colname_e='e_Zprime_HII', unit_e='',
            ra=None, dec=None,
            logOH=None, e_logOH=None, logOH_solar=None):
        if logOH_solar is None:
            logOH_solar = 8.69  # Asplund+09
        if logOH is not None:
            Zprime = 10 ** (logOH - logOH_solar) * u.Unit('')
        else:
            Zprime = None
        if e_logOH is not None:
            e_Zprime = Zprime * (10**e_logOH - 1)
        else:
            e_Zprime = None
        self.add_hii_stat_generic(
            colname, colname_e=colname_e, unit=unit,
            ra=ra, dec=dec, quantity=Zprime, e_quantity=e_Zprime)
        # convert uncertainty unit
        self[colname_e] = self[colname_e].to(unit_e)

    def add_hii_extinction(
            self, colname='EBV_HII', unit='mag',
            colname_e='e_EBV_HII', unit_e='mag',
            ra=None, dec=None, EBV=None, e_EBV=None):
        self.add_hii_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            ra=ra, dec=dec, quantity=EBV, e_quantity=e_EBV)
        # convert uncertainty unit
        self[colname_e] = self[colname_e].to(unit_e)

    # -------------------------------------------------------------------------
    # methods for DAP map statistics
    # -------------------------------------------------------------------------

    def add_area_average_for_image(
            self, colname=None, unit=None, colname_e=None, unit_e=None,
            img_file=None, err_file=None):

        if not Path(img_file).is_file():
            self[colname] = np.nan * u.Unit(unit)
            if colname_e is not None:
                self[colname_e] = np.nan * u.Unit(unit_e)
            return

        # sample image
        with fits.open(img_file) as hdul:
            data = hdul[0].data.copy()
            hdr = hdul[0].header.copy()
        self.calc_image_stats(
            data, header=hdr, stat_func=nanaverage, weight=None,
            colname=colname, unit='header')
        self[colname] = self[colname].to(unit)

        # sample error image
        if colname_e is None:
            return
        if not Path(err_file).is_file():
            self[colname_e] = np.nan * u.Unit(unit_e)
            return
        with fits.open(err_file) as hdul:
            data_e = hdul[0].data.copy()
        nanflag = np.isnan(data)
        # calculate the direct RMS of errors in each pixel
        self.calc_image_stats(
            data_e, header=hdr, stat_func=nanrms, weight=~nanflag,
            colname=colname_e, unit='header')
        self[colname_e] = self[colname_e].to(unit_e)

    # -------------------------------------------------------------------------
    # methods for SFR calculations
    # -------------------------------------------------------------------------

    def calc_surf_dens_sfr(
            self, colname='Sigma_SFR', unit='Msun yr-1 kpc-2',
            colname_e='e_Sigma_SFR', unit_e='Msun yr-1 kpc-2',
            method='Hacorr', I_Halpha=None, e_I_Halpha=None,
            cosi=1., e_sys=None, snr_thresh=None):
        if e_sys is None:
            e_sys = 0.0
        if snr_thresh is None:
            snr_thresh = 3
        if method == 'Hacorr':
            # Calzetti+07, Murphy+11
            C_Halpha = 5.37e-42 * u.Unit('Msun yr-1 erg-1 s')
            self[colname] = (
                C_Halpha * cosi * 4*np.pi*u.sr * I_Halpha).to(unit)
            e_stat = C_Halpha * cosi * 4*np.pi*u.sr * e_I_Halpha
            self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
            low_snr_flag = (
                np.isfinite(I_Halpha) &
                (I_Halpha / e_I_Halpha < snr_thresh))
            self[colname][low_snr_flag] = 0
            # self[colname_e][low_snr_flag] = np.nan
        else:
            raise ValueError(f"Unrecognized method: '{method}'")


class PhangsMuseTessellMegaTable(TessellMegaTable, PhangsMuseMegaTable):

    """
    TessellMegaTable for PHANGS-ALMA data.
    """


class PhangsMuseRadialMegaTable(RadialMegaTable, PhangsMuseMegaTable):

    """
    RadialMegaTable for PHANGS-ALMA data.
    """


# -----------------------------------------------------------------------------


def add_DAP_stats_to_table(
        t: PhangsMuseMegaTable, data_paths=None, verbose=True):

    gal_name = t.meta['GALAXY']

    res_str = 'copt'
    if verbose:
        print("  Add DAP map statistics")

    # MUSE DAP Halpha flux map (raw)
    if verbose:
        print("    Add DAP Halpha flux (raw)")
    in_file = data_paths['PHANGS_MUSE_DAP'].format(
        galaxy=gal_name, product='Ha6562_flux',
        resolution=res_str)
    err_file = data_paths['PHANGS_MUSE_DAP'].format(
        galaxy=gal_name, product='Ha6562_flux_err',
        resolution=res_str)
    t.add_area_average_for_image(
        # column to save the output
        colname='I_Halpha_DAP', unit='erg s-1 cm-2 arcsec-2',
        colname_e="e_I_Halpha_DAP", unit_e='erg s-1 cm-2 arcsec-2',
        # input parameters
        img_file=in_file, err_file=err_file)

    # MUSE DAP Halpha flux map (extinction-corrected)
    if verbose:
        print("    Add DAP Halpha flux (extinction-corrected)")
    in_file = data_paths['PHANGS_MUSE_DAP'].format(
        galaxy=gal_name, product='Ha6562_flux_corr',
        resolution=res_str)
    err_file = data_paths['PHANGS_MUSE_DAP'].format(
        galaxy=gal_name, product='Ha6562_flux_corr_err',
        resolution=res_str)
    t.add_area_average_for_image(
        # column to save the output
        colname='I_Halpha_DAP_corr', unit='erg s-1 cm-2 arcsec-2',
        colname_e="e_I_Halpha_DAP_corr", unit_e='erg s-1 cm-2 arcsec-2',
        # input parameters
        img_file=in_file, err_file=err_file)


def add_nebulae_stats_to_table(
        t: PhangsMuseMegaTable, data_paths=None, verbose=True):

    gal_name = t.meta['GALAXY']

    if verbose:
        print("  Add nebulae statistics")

    # read nebulae catalog file
    nebulae_file = Path(data_paths['PHANGS_MUSE_nebulae'].format(
        galaxy=gal_name))
    if not nebulae_file.is_file():
        if verbose:
            print("    Input file not found -- will do a dry run")
        ra = dec = flux_Ha = e_flux_Ha = logOH = e_logOH = EBV = e_EBV = None
    else:
        if verbose:
            print("    Input file found")
        nebulaecat = Table.read(nebulae_file)
        hiicat = nebulaecat[nebulaecat['HII_class_v3'] == 1]  # HII only
        ra = hiicat['cen_ra'].quantity.to('deg').value
        dec = hiicat['cen_dec'].quantity.to('deg').value
        flux_Ha = hiicat['HA6562_FLUX_CORR'].quantity.to('erg s-1 cm-2')
        e_flux_Ha = hiicat['HA6562_FLUX_CORR_ERR'].quantity.to('erg s-1 cm-2')
        logOH = np.array(hiicat['met_scal'])
        e_logOH = np.array(hiicat['met_scal_err'])
        EBV = hiicat['EBV'].quantity.to('mag')
        e_EBV = hiicat['EBV_ERR'].quantity.to('mag')
        mask = \
            np.isfinite(flux_Ha) & np.isfinite(logOH) & np.isfinite(EBV)
        flux_Ha[~mask] = np.nan
        e_flux_Ha[~mask] = np.nan
        logOH[~mask] = np.nan
        e_logOH[~mask] = np.nan
        EBV[~mask] = np.nan
        e_EBV[~mask] = np.nan

    # HII region number count
    if verbose:
        print("    Add HII region number count")
    t.add_hii_count(
        # column to save the output
        colname="N_HII",
        # input parameters
        ra=ra, dec=dec, flux_Ha=flux_Ha)

    # Halpha flux (direct mean)
    if verbose:
        print("    Add Halpha flux (direct mean)")
    t.add_hii_ha_flux(
        # column to save the output
        colname="F_Halpha_HII", colname_e="e_F_Halpha_HII",
        # input parameters
        ra=ra, dec=dec, flux_Ha=flux_Ha, e_flux_Ha=e_flux_Ha)

    # Halpha flux (weighted mean)
    if verbose:
        print("    Add Halpha flux (weighted mean)")
    t.add_hii_ha_flux_weighted(
        # column to save the output
        colname="<F_Halpha_HII>", colname_e="e_<F_Halpha_HII>",
        # input parameters
        ra=ra, dec=dec, flux_Ha=flux_Ha, e_flux_Ha=e_flux_Ha)

    # extinction
    if verbose:
        print("    Add extinction")
    t.add_hii_extinction(
        # column to save the output
        colname="EBV_HII", colname_e="e_EBV_HII",
        # input parameters
        ra=ra, dec=dec, EBV=EBV, e_EBV=e_EBV)

    # metallicity
    if verbose:
        print("    Add metallicity")
    t.add_hii_metallicity(
        # column to save the output
        colname="Zprime_HII", colname_e="e_Zprime_HII",
        # input parameters
        ra=ra, dec=dec, logOH=logOH, e_logOH=e_logOH)


def calc_high_level_params_in_table(
        t: PhangsMuseMegaTable, verbose=True):

    gal_cosi = np.cos(np.deg2rad(t.meta['INCL_DEG']))

    if verbose:
        print("  Calculate high-level parameters")

    # SFR surface density (Av-corrected Halpha)
    if verbose:
        print("    Calculate SFR surface density (Av-corrected Halpha)")
    method = 'Hacorr'
    t.calc_surf_dens_sfr(
        # columns to save the output
        colname=f"Sigma_SFR_{method}", colname_e=f"e_Sigma_SFR_{method}",
        # input parameters
        method=method, I_Halpha=t['I_Halpha_DAP_corr'],
        e_I_Halpha=t['e_I_Halpha_DAP_corr'],
        cosi=gal_cosi, snr_thresh=3)


def build_muse_table_from_base_table(
        t, data_paths=None, output_format=None,
        writefile=None, verbose=True):

    # if resolutions is None:
    #     resolutions = (None, )

    # ------------------------------------------------
    # add DAP map statistics
    # ------------------------------------------------

    add_DAP_stats_to_table(
        t, data_paths=data_paths, verbose=verbose)

    # ------------------------------------------------
    # add nebulae statistics
    # ------------------------------------------------

    add_nebulae_stats_to_table(
        t, data_paths=data_paths, verbose=verbose)

    # ------------------------------------------------
    # calculate high-level parameters
    # ------------------------------------------------

    calc_high_level_params_in_table(
        t, verbose=verbose)

    # ------------------------------------------------
    # clean and format output table
    # ------------------------------------------------

    if isinstance(t, TessellMegaTable):
        colnames = (
            ['ID', 'RA', 'DEC', 'r_gal', 'phi_gal'] +
            [str(entry) for entry in output_format['colname']])
        units = (
            ['', 'deg', 'deg', 'kpc', 'deg'] +
            [str(entry) for entry in output_format['unit']])
        formats = (
            ['.0f', '.5f', '.5f', '.2f', '.2f'] +
            [str(entry) for entry in output_format['format']])
        descriptions = (
            ["Aperture ID",
             "Right Ascension of the aperture center",
             "Declination of the aperture center",
             "Deprojected galactocentric radius",
             "Deprojected azimuthal angle (0 = receding major axis)"] +
            [str(entry) for entry in output_format['description']])
    elif isinstance(t, RadialMegaTable):
        colnames = (
            ['ID', 'r_gal'] +
            [str(entry) for entry in output_format['colname']])
        units = (
            ['', 'kpc'] +
            [str(entry) for entry in output_format['unit']])
        formats = (
            ['.0f', '.2f'] +
            [str(entry) for entry in output_format['format']])
        descriptions = (
            ["Annulus ID",
             "Deprojected galactocentric radius"] +
            [str(entry) for entry in output_format['description']])
    else:
        raise ValueError('Unsupported mega-table type for formatting')
    t.format(
        colnames=colnames, units=units, formats=formats,
        descriptions=descriptions, ignore_missing=True)

    # ------------------------------------------------
    # write table
    # ------------------------------------------------

    if writefile is not None:
        if verbose:
            print("  Write table")
        if Path(writefile).suffix == '.ecsv':
            t.write(
                writefile, add_timestamp=True,
                delimiter=',', overwrite=True)
        else:
            t.write(writefile, add_timestamp=True, overwrite=True)
        return writefile
    else:
        return t


# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # warning and logging settings
    warnings.simplefilter('ignore', RuntimeWarning)
    if logging:
        # output log to a file
        orig_stdout = sys.stdout
        log = open(work_dir / (str(Path(__file__).stem) + '.log'), 'w')
        sys.stdout = log
    else:
        orig_stdout = log = None

    with open(config_dir / "config_data_path.json") as f:
        data_paths = json.load(f)
    with open(config_dir / "config_tables.json") as f:
        table_configs = json.load(f)
    t_format = Table.read(config_dir / "format_phangsmuse.csv")

    # read sample table
    t_sample = Table.read(data_paths['PHANGS_sample_table'])

    # sub-select sample
    t_sample = t_sample[t_sample['data_has_megatable']]

    # loop through all galaxies
    for row in t_sample:

        # extract galaxy name
        gal_name = row['name'].upper()

        print("\n############################################################")
        print(f"# {gal_name}")
        print("############################################################\n")

        # TessellMegaTable
        tile_shape = table_configs['tessell_tile_shape']
        tile_size_str = (
            str(table_configs['tessell_tile_size']).replace('.', 'p') +
            table_configs['tessell_tile_size_unit'])

        tessell_base_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content='base',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        tessell_muse_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content='phangsmuse',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        if (tessell_base_table_file.is_file() and not
                tessell_muse_table_file.is_file()):
            print("Enhancing tessellation statistics table...")
            t = PhangsMuseTessellMegaTable.read(tessell_base_table_file)
            build_muse_table_from_base_table(
                t, data_paths=data_paths, output_format=t_format,
                writefile=tessell_muse_table_file)
            print("Done\n")

        # RadialMegaTable
        annulus_width_str = (
            str(table_configs['radial_annulus_width']).replace('.', 'p') +
            table_configs['radial_annulus_width_unit'])
        radial_base_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content='base',
                annulus_width_str=annulus_width_str))
        radial_muse_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content='phangsmuse',
                annulus_width_str=annulus_width_str))
        if (radial_base_table_file.is_file() and not
                radial_muse_table_file.is_file()):
            print("Enhancing radial statistics table...")
            t = PhangsMuseRadialMegaTable.read(radial_base_table_file)
            build_muse_table_from_base_table(
                t, data_paths=data_paths, output_format=t_format,
                writefile=radial_muse_table_file)
            print("Done\n")

    # logging settings
    if logging:
        sys.stdout = orig_stdout
        log.close()
