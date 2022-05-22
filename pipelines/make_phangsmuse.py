import sys
import json
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u, constants as const
from astropy import uncertainty as unc
from astropy.io import fits
from astropy.table import Table

from mega_table.core import StatsTable
from mega_table.table import TessellMegaTable, RadialMegaTable
from mega_table.utils import nanaverage

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

    def add_hii_ha_flux(
            self, colname='<F_Halpha_HII>', unit='erg s-1 cm-2',
            colname_e='e_<F_Halpha_HII>', unit_e='erg s-1 cm-2',
            ra=None, dec=None, flux_Ha=None, e_flux_Ha=None):
        if ra is None or dec is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        # contruct weight array
        wt = flux_Ha.value.copy()
        wt[~np.isfinite(wt)] = 0
        # calculate flux-weighted average
        self.calc_catalog_stats(
            flux_Ha.to(unit).value, ra, dec, weight=wt,
            stat_func=nanaverage, colname=colname, unit=unit)
        if colname_e is None:
            return
        # calculate formal uncertainty on the averaged value
        findarr = self.find_coords_in_regions(ra, dec)
        if findarr.ndim == 2:
            avg_flux_Ha = np.full(len(ra), np.nan) * self[colname].unit
            for ireg in range(len(self)):
                avg_flux_Ha[findarr[:, ireg]] = self[colname][ireg]
        else:
            avg_flux_Ha = self[colname][findarr]
            avg_flux_Ha[findarr == -1] = np.nan
        self.calc_catalog_stats(
            ((2 * flux_Ha - avg_flux_Ha)**2 * e_flux_Ha**2).value,
            ra, dec, stat_func=np.nansum, colname='_sqsum(err)',
            unit=flux_Ha.unit**2 * e_flux_Ha.unit**2)
        self.calc_catalog_stats(
            flux_Ha.value, ra, dec, stat_func=np.nansum,
            colname='_sum(F)', unit=flux_Ha.unit)
        self[colname_e] = (
            np.sqrt(self['_sqsum(err)']) / self['_sum(F)']).to(unit_e)
        self.table.remove_columns(['_sqsum(err)', '_sum(F)'])

    def add_hii_metallicity(
            self, colname='<Zprime_HII>', unit='',
            colname_e='e_<Zprime_HII>', unit_e='',
            ra=None, dec=None, flux_Ha=None, e_flux_Ha=None,
            logOH=None, e_logOH=None, logOH_solar=None):
        if ra is None or dec is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        # contruct weight array
        wt = flux_Ha.value.copy()
        wt[~np.isfinite(wt)] = 0
        # calculate flux-weighted average
        if logOH_solar is None:
            logOH_solar = 8.69  # Asplund+09
        Zprime = 10 ** (logOH - logOH_solar) * u.Unit('')
        self.calc_catalog_stats(
            Zprime.to(unit).value, ra, dec, weight=wt,
            stat_func=nanaverage, colname=colname, unit=unit)
        if colname_e is None:
            return
        # calculate formal uncertainty on the averaged value
        findarr = self.find_coords_in_regions(ra, dec)
        if findarr.ndim == 2:
            avg_Zprime = np.full(len(ra), np.nan) * self[colname].unit
            for ireg in range(len(self)):
                avg_Zprime[findarr[:, ireg]] = self[colname][ireg]
        else:
            avg_Zprime = self[colname][findarr]
            avg_Zprime[findarr == -1] = np.nan
        e_Zprime = Zprime * (10**e_logOH - 1)
        self.calc_catalog_stats(
            ((Zprime - avg_Zprime)**2 * e_flux_Ha**2 +
             flux_Ha**2 * e_Zprime**2).value,
            ra, dec, stat_func=np.nansum, colname='_sqsum(err)',
            unit=Zprime.unit**2 * flux_Ha.unit**2)
        self.calc_catalog_stats(
            flux_Ha.value, ra, dec, stat_func=np.nansum,
            colname='_sum(F)', unit=flux_Ha.unit)
        self[colname_e] = (
            np.sqrt(self['_sqsum(err)']) / self['_sum(F)']).to(unit_e)
        self.table.remove_columns(['_sqsum(err)', '_sum(F)'])

    def add_hii_extinction(
            self, colname='med[E(B-V)_HII]', unit='mag',
            colname_e='e_med[E(B-V)_HII]', unit_e='mag',
            ra=None, dec=None, EBV=None, e_EBV=None):
        if ra is None or dec is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        # calculate median
        self.calc_catalog_stats(
            EBV.to(unit).value, ra, dec,
            stat_func=np.median, colname=colname, unit=unit)
        if colname_e is None:
            return
        # calculate error on median
        self[colname_e] = np.nan * u.Unit(unit_e)
        # def err_on_median(x): return np.median(x).pdf_std()
        # EBV_dist = unc.normal(EBV, std=e_EBV, n_samples=1000)
        # self.calc_catalog_stats(
        #     EBV_dist.to(unit).value, ra, dec,
        #     stat_func=err_on_median, colname=colname, unit=unit_e)

    # -------------------------------------------------------------------------
    # methods for DAP map statistics
    # -------------------------------------------------------------------------


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
        t: PhangsMuseMegaTable, data_paths=None, res_pcs=None, verbose=True):

    gal_name = t.meta['GALAXY']
    gal_dist = t.meta['DIST_MPC'] * u.Mpc
    gal_cosi = np.cos(np.deg2rad(t.meta['INCL_DEG']))

    for res_pc in res_pcs:

        if res_pc is None:
            res_str = 'copt'
        else:
            res_str = f"{res_pc}pc"
        if verbose:
            print(f"  Add DAP map statistics @ {res_str} resolution")


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
        hiicat = nebulaecat[nebulaecat['HII_class'] == 1]  # HII regions only
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

    # Halpha line flux
    if verbose:
        print("    Add Halpha line flux")
    t.add_hii_ha_flux(
        # column to save the output
        colname="<F_Halpha_HII>", colname_e="e_<F_Halpha_HII>",
        # input parameters
        ra=ra, dec=dec, flux_Ha=flux_Ha, e_flux_Ha=e_flux_Ha)

    # metallicity
    if verbose:
        print("    Add metallicity")
    t.add_hii_metallicity(
        # column to save the output
        colname="<Zprime_HII>", colname_e="e_<Zprime_HII>",
        # input parameters
        ra=ra, dec=dec, flux_Ha=flux_Ha, e_flux_Ha=e_flux_Ha,
        logOH=logOH, e_logOH=e_logOH)

    # extinction
    if verbose:
        print("    Add extinction")
    t.add_hii_extinction(
        # column to save the output
        colname="med[E(B-V)_HII]", colname_e="e_med[E(B-V)_HII]",
        # input parameters
        ra=ra, dec=dec, EBV=EBV, e_EBV=e_EBV)


def build_muse_table_from_base_table(
        t, data_paths=None, output_format=None, resolutions=None,
        writefile=None, verbose=True):

    if resolutions is None:
        resolutions = (None, )

    # ------------------------------------------------
    # add DAP map statistics
    # ------------------------------------------------

    add_DAP_stats_to_table(
        t, data_paths=data_paths, res_pcs=resolutions, verbose=verbose)

    # ------------------------------------------------
    # add nebulae statistics
    # ------------------------------------------------

    add_nebulae_stats_to_table(
        t, data_paths=data_paths, verbose=verbose)

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
