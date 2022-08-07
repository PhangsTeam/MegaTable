import sys
import json
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.table import Table

from mega_table.table import TessellMegaTable

###############################################################################

# location of all relevant config files
config_dir = Path('/data/kant/0/sun.1608/PHANGS/mega-tables/code')

# location to save the output data tables
work_dir = Path('/data/kant/0/sun.1608/PHANGS/mega-tables')

# logging setting
logging = False

###############################################################################


class PhangsGaussKernelMegaTable(TessellMegaTable):

    """
    MegaTable for PHANGS data sampled with Gaussian kernel averaging.
    """

    def add_gauss_average_from_convolved_image(
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
        self.resample_image(
            data, header=hdr, colname=colname, unit='header',
            fill_outside=np.nan)
        self[colname] = self[colname].to(unit)

        # sample error image
        if colname_e is None:
            return
        if not Path(err_file).is_file():
            self[colname_e] = np.nan * u.Unit(unit_e)
            return
        with fits.open(err_file) as hdul:
            data_e = hdul[0].data.copy()
        self.resample_image(
            data_e, header=hdr, colname=colname_e, unit='header',
            fill_outside=np.nan)
        self[colname_e] = self[colname_e].to(unit_e)

    def calc_surf_dens_sfr(
            self, colname='Sigma_SFR', unit='Msun yr-1 kpc-2',
            colname_e='e_Sigma_SFR', unit_e='Msun yr-1 kpc-2',
            method='FUVW4', I_Halpha=None, e_I_Halpha=None,
            I_UV=None, e_I_UV=None, I_IR=None, e_I_IR=None,
            cosi=1., e_sys=None, snr_thresh=None):
        cal_UVIR = {  # Leroy+19, Table 7
            'FUVW4': (-43.42, 154*u.nm, -42.73, 22*u.um),
            'NUVW4': (-43.24, 154*u.nm, -42.79, 22*u.um),
            'FUVW3': (-43.42, 231*u.nm, -42.79, 12*u.um),
            'NUVW3': (-43.24, 231*u.nm, -42.86, 12*u.um)}
        cal_IR = {  # Leroy+19, Table 7
            'W4ONLY': (-42.63, 22*u.um),
            'W3ONLY': (-42.70, 12*u.um)}
        if e_sys is None:
            e_sys = 0.0
        if snr_thresh is None:
            snr_thresh = 3
        if method in cal_UVIR:
            C_UV = 10**cal_UVIR[method][0] * u.Unit('Msun yr-1 erg-1 s')
            nu_UV = const.c / cal_UVIR[method][1]
            C_IR = 10**cal_UVIR[method][2] * u.Unit('Msun yr-1 erg-1 s')
            nu_IR = const.c / cal_UVIR[method][3]
            self[colname] = (
                C_UV * nu_UV * cosi * 4*np.pi*u.sr * I_UV +
                C_IR * nu_IR * cosi * 4*np.pi*u.sr * I_IR).to(unit)
            e_stat = np.sqrt(
                (C_UV * nu_UV * cosi * 4*np.pi*u.sr * e_I_UV)**2 +
                (C_IR * nu_IR * cosi * 4*np.pi*u.sr * e_I_IR)**2)
            self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
            low_snr_flag = (
                np.isfinite(I_UV) & np.isfinite(I_IR) &
                ((I_UV / e_I_UV < snr_thresh) |
                 (I_IR / e_I_IR < snr_thresh)))
            self[colname][low_snr_flag] = 0
            # self[colname_e][low_snr_flag] = np.nan
        elif method in cal_IR:
            C_IR = 10**cal_IR[method][0] * u.Unit('Msun yr-1 erg-1 s')
            nu_IR = const.c / cal_IR[method][1]
            self[colname] = (
                C_IR * nu_IR * cosi * 4*np.pi*u.sr * I_IR).to(unit)
            e_stat = C_IR * nu_IR * cosi * 4*np.pi*u.sr * e_I_IR
            self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
            low_snr_flag = (I_IR / e_I_IR < snr_thresh)
            self[colname][low_snr_flag] = 0
            # self[colname_e][low_snr_flag] = np.nan
        elif method == 'HaW4':
            # Calzetti+07, Murphy+11
            C_Halpha = 5.37e-42 * u.Unit('Msun yr-1 erg-1 s')
            C_IR = 0.031 * C_Halpha
            nu_IR = const.c / (22 * u.um)
            self[colname] = (
                C_Halpha * cosi * 4*np.pi*u.sr * I_Halpha +
                C_IR * nu_IR * cosi * 4*np.pi*u.sr * I_IR).to(unit)
            e_stat = np.sqrt(
                (C_Halpha * cosi * 4*np.pi*u.sr * e_I_Halpha)**2 +
                (C_IR * nu_IR * cosi * 4*np.pi*u.sr * e_I_IR)**2)
            self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
            low_snr_flag = (
                np.isfinite(I_Halpha) & np.isfinite(I_IR) &
                ((I_Halpha / e_I_Halpha < snr_thresh) |
                 (I_IR / e_I_IR < snr_thresh)))
            self[colname][low_snr_flag] = 0
            # self[colname_e][low_snr_flag] = np.nan
        else:
            raise ValueError(f"Unrecognized method: '{method}'")

    def calc_surf_dens_sfr_new(
            self, colname='Sigma_SFR', unit='Msun yr-1 kpc-2',
            colname_e='e_Sigma_SFR', unit_e='Msun yr-1 kpc-2',
            method='Hacorr', I_Halpha=None, e_I_Halpha=None,
            I_FUV=None, e_I_FUV=None, I_W4=None, e_I_W4=None,
            I_W1=None, e_I_W1=None, cosi=1., e_sys=None, snr_thresh=None):
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
        elif method == 'HaW4recal':
            # Calzetti+07, Murphy+11
            C_Halpha = 5.37e-42 * u.Unit('Msun yr-1 erg-1 s')
            # Belfiore+22
            a1 = 0.44
            logQmax = -1.51
            logCmax = -42.87
            nu_W1 = const.c / (3.4 * u.um)
            nu_W4 = const.c / (22 * u.um)
            logQ = np.log10((I_Halpha / (nu_W1 * I_W1)).to('').value)
            logC_W4 = logCmax + a1 * (logQ - logQmax)
            logC_W4[logQ > logQmax] = logCmax
            C_W4 = 10**logC_W4 * u.Unit('Msun yr-1 erg-1 s')
            self[colname] = (
                C_Halpha * cosi * 4*np.pi*u.sr * I_Halpha +
                C_W4 * nu_W4 * cosi * 4*np.pi*u.sr * I_W4).to(unit)
            e_stat = np.sqrt(
                (C_Halpha * cosi * 4*np.pi*u.sr * e_I_Halpha)**2 +
                (C_W4 * nu_W4 * cosi * 4*np.pi*u.sr * e_I_W4)**2)
            self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
            low_snr_flag = (
                np.isfinite(I_Halpha) & np.isfinite(I_W4) &
                ((I_Halpha / e_I_Halpha < snr_thresh) |
                 (I_W4 / e_I_W4 < snr_thresh)))
            self[colname][low_snr_flag] = 0
            # self[colname_e][low_snr_flag] = np.nan
        elif method == 'FUVW4recal':
            # Leroy+19
            C_FUV = 10**-43.42 * u.Unit('Msun yr-1 erg-1 s')
            nu_FUV = const.c / (154 * u.nm)
            # Belfiore+22, FUV/W1
            a1 = 0.22
            logQmax = 0.60
            logCmax = -42.73
            nu_W1 = const.c / (3.4 * u.um)
            nu_W4 = const.c / (22 * u.um)
            logQ = np.log10((nu_FUV * I_FUV / (nu_W1 * I_W1)).to('').value)
            logC_W4 = logCmax + a1 * (logQ - logQmax)
            logC_W4[logQ > logQmax] = logCmax
            C_W4 = 10**logC_W4 * u.Unit('Msun yr-1 erg-1 s')
            self[colname] = (
                C_FUV * nu_FUV * cosi * 4*np.pi*u.sr * I_FUV +
                C_W4 * nu_W4 * cosi * 4*np.pi*u.sr * I_W4).to(unit)
            e_stat = np.sqrt(
                (C_FUV * nu_FUV * cosi * 4*np.pi*u.sr * e_I_FUV)**2 +
                (C_W4 * nu_W4 * cosi * 4*np.pi*u.sr * e_I_W4)**2)
            self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
            low_snr_flag = (
                np.isfinite(I_FUV) & np.isfinite(I_W4) &
                ((I_FUV / e_I_FUV < snr_thresh) |
                 (I_W4 / e_I_W4 < snr_thresh)))
            self[colname][low_snr_flag] = 0
            # self[colname_e][low_snr_flag] = np.nan
        else:
            raise ValueError(f"Unrecognized method: '{method}'")

    def calc_surf_dens_mol(
            self, colname='Sigma_mol', unit='Msun pc-2',
            colname_e='e_Sigma_mol', unit_e='Msun pc-2',
            I_CO=None, e_I_CO=None, alpha_CO=None,
            cosi=1., e_sys=None, snr_thresh=None):
        self[colname] = (alpha_CO * cosi * I_CO).to(unit)
        e_stat = alpha_CO * cosi * e_I_CO
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
        # mask entries below S/N threshold
        if snr_thresh is None:
            snr_thresh = 3
        low_snr_flag = (I_CO / e_I_CO < snr_thresh)
        self[colname][low_snr_flag] = 0
        # self[colname_e][low_snr_flag] = np.nan

    def calc_surf_dens_atom(
            self, colname='Sigma_atom', unit='Msun pc-2',
            colname_e='e_Sigma_atom', unit_e='Msun pc-2',
            I_HI=None, e_I_HI=None,
            cosi=1., e_sys=None, snr_thresh=None):
        alpha_HI = 0.0197 * u.Unit('Msun pc-2 K-1 km-1 s')
        self[colname] = (alpha_HI * cosi * I_HI).to(unit)
        e_stat = alpha_HI * cosi * e_I_HI
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
        # mask entries below S/N threshold
        if snr_thresh is None:
            snr_thresh = 3
        low_snr_flag = (I_HI / e_I_HI < snr_thresh)
        self[colname][low_snr_flag] = 0
        # self[colname_e][low_snr_flag] = np.nan

    def calc_surf_dens_star(
            self, colname='Sigma_star', unit='Msun pc-2',
            colname_e='e_Sigma_star', unit_e='Msun pc-2',
            method='3p4um', I_IR=None, e_I_IR=None, MtoL=None,
            cosi=1., e_sys=None, snr_thresh=None):
        # Leroy+21
        if method == '3p4um':
            nu_IR = const.c / (3.4 * u.um)
            MtoL_IR = MtoL / 0.042
        elif method == '3p6um':
            nu_IR = const.c / (3.6 * u.um)
            MtoL_IR = MtoL / 0.037
        else:
            raise ValueError(f"Unrecognized method: '{method}'")
        self[colname] = (
            MtoL_IR * 4*np.pi*u.sr * nu_IR * cosi * I_IR).to(unit)
        e_stat = MtoL_IR * 4*np.pi*u.sr * nu_IR * cosi * e_I_IR
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)
        # mask entries below S/N threshold
        if snr_thresh is None:
            snr_thresh = 3
        low_snr_flag = (I_IR / e_I_IR < snr_thresh)
        self[colname][low_snr_flag] = 0
        # self[colname_e][low_snr_flag] = np.nan


# -----------------------------------------------------------------------------


def add_gauss_average_to_table(
        t: PhangsGaussKernelMegaTable, res_kpc=None,
        data_paths=None, verbose=True):

    gal_name = t.meta['GALAXY']

    res_str = f"{res_kpc*1e3:.0f}pc"
    res_str_alt = f"{res_kpc:.1f}kpc".replace('.', 'p')

    if verbose:
        print("  Add Gaussian kernel average statistics")

    # GALEX and WISE data (z0MGS)
    if verbose:
        print("    Add GALEX and WISE data")
    products = (
        'FUV', 'NUV',
        'W1', 'W2', 'W3', 'W4')
    colnames = (
        "I_154nm", "I_231nm",
        "I_3p4um", "I_4p6um", "I_12um", "I_22um")
    for product, colname in zip(products, colnames):
        in_file = data_paths['PHANGS_z0MGS'].format(
            galaxy=gal_name, product=product,
            postfix_resolution=f"_{res_str}")
        err_file = data_paths['PHANGS_z0MGS'].format(
            galaxy=gal_name, product=f"{product}_err",
            postfix_resolution=f"_{res_str}")
        t.add_gauss_average_from_convolved_image(
            # column to save the output
            colname=f"{colname}_gauss", unit='MJy sr-1',
            colname_e=f"e_{colname}_gauss", unit_e='MJy sr-1',
            # input parameters
            img_file=in_file, err_file=err_file)

    # Spitzer IRAC data
    if verbose:
        print("    Add Spitzer IRAC data")
    in_file = data_paths['PHANGS_IRAC'].format(
        galaxy=gal_name, product='3p6um',
        postfix_resolution=f"_{res_str}")
    err_file = data_paths['PHANGS_IRAC'].format(
        galaxy=gal_name, product='3p6um_err',
        postfix_resolution=f"_{res_str}")
    t.add_gauss_average_from_convolved_image(
        # column to save the output
        colname='I_3p6um_gauss', unit='MJy sr-1',
        colname_e="e_I_3p6um_gauss", unit_e='MJy sr-1',
        # input parameters
        img_file=in_file, err_file=err_file)

    # PHANGS narrow-band Halpha data
    if verbose:
        print("    Add PHANGS narrow-band Halpha data")
    in_file = data_paths['PHANGS_Halpha'].format(
        galaxy=gal_name, product='cleaned',
        postfix_resolution=f"_{res_str}")
    err_file = data_paths['PHANGS_Halpha'].format(
        galaxy=gal_name, product='err_cleaned',
        postfix_resolution=f"_{res_str}")
    t.add_gauss_average_from_convolved_image(
        # column to save the output
        colname='I_Halpha_gauss', unit='erg s-1 cm-2 arcsec-2',
        colname_e="e_I_Halpha_gauss", unit_e='erg s-1 cm-2 arcsec-2',
        # input parameters
        img_file=in_file, err_file=err_file)

    # PHANGS-VLA & archival HI data
    # !!! no data at 1.5 kpc resolution is currently available !!!

    # PHANGS-ALMA CO(2-1) data
    if verbose:
        print("    Add PHANGS-ALMA CO(2-1) data")
    in_file = data_paths['PHANGS_ALMA_CO21'].format(
        galaxy=gal_name, product='mom0',
        postfix_masking='_broad',
        postfix_resolution=f"_{res_str_alt}")
    err_file = data_paths['PHANGS_ALMA_CO21'].format(
        galaxy=gal_name, product='emom0',
        postfix_masking='_broad',
        postfix_resolution=f"_{res_str_alt}")
    t.add_gauss_average_from_convolved_image(
        # column to save the output
        colname='I_CO21_gauss', unit='K km s-1',
        colname_e="e_I_CO21_gauss", unit_e='K km s-1',
        # input parameters
        img_file=in_file, err_file=err_file)

    # PHANGS-MUSE Halpha data (raw)
    if verbose:
        print("    Add PHANGS-MUSE Halpha data (raw)")
    in_file = data_paths['PHANGS_MUSE_DAP'].format(
        galaxy=gal_name, product='Ha6562_flux',
        resolution=res_str)
    err_file = data_paths['PHANGS_MUSE_DAP'].format(
        galaxy=gal_name, product='Ha6562_flux_err',
        resolution=res_str)
    t.add_gauss_average_from_convolved_image(
        # column to save the output
        colname='I_Halpha_DAP_gauss',
        unit='erg s-1 cm-2 arcsec-2',
        colname_e="e_I_Halpha_DAP_gauss",
        unit_e='erg s-1 cm-2 arcsec-2',
        # input parameters
        img_file=in_file, err_file=err_file)

    # PHANGS-MUSE Halpha data (extinction-corrected)
    if verbose:
        print("    Add PHANGS-MUSE Halpha data (Av-corrected)")
    in_file = data_paths['PHANGS_MUSE_DAP'].format(
        galaxy=gal_name, product='Ha6562_flux_corr',
        resolution=res_str)
    err_file = data_paths['PHANGS_MUSE_DAP'].format(
        galaxy=gal_name, product='Ha6562_flux_corr_err',
        resolution=res_str)
    t.add_gauss_average_from_convolved_image(
        # column to save the output
        colname='I_Halpha_DAP_corr_gauss',
        unit='erg s-1 cm-2 arcsec-2',
        colname_e="e_I_Halpha_DAP_corr_gauss",
        unit_e='erg s-1 cm-2 arcsec-2',
        # input parameters
        img_file=in_file, err_file=err_file)


def calc_high_level_params_in_table(
        t: PhangsGaussKernelMegaTable, verbose=True):

    gal_cosi = np.cos(np.deg2rad(t.meta['INCL_DEG']))

    if verbose:
        print("  Calculate high-level parameters")

    # SFR surface density
    if verbose:
        print("    Calculate SFR surface density")
    # UV+IR prescriptions
    bands_IR = ('W4', 'W3')
    data_IR = (t['I_22um_gauss'], t['I_12um_gauss'])
    errs_IR = (t['e_I_22um_gauss'], t['e_I_12um_gauss'])
    bands_UV = ('FUV', 'NUV', '')
    data_UV = (t['I_154nm_gauss'], t['I_231nm_gauss'], None)
    errs_UV = (t['e_I_154nm_gauss'], t['e_I_231nm_gauss'], None)
    methods = []
    for band_IR, I_IR, e_I_IR in zip(bands_IR, data_IR, errs_IR):
        for band_UV, I_UV, e_I_UV in zip(bands_UV, data_UV, errs_UV):
            if band_UV == '':
                method = f"{band_IR}ONLY"
            else:
                method = f"{band_UV}{band_IR}"
            methods += [method]
            t.calc_surf_dens_sfr(
                # columns to save the output
                colname=f"Sigma_SFR_{method}_gauss",
                colname_e=f"e_Sigma_SFR_{method}_gauss",
                # input parameters
                method=method,
                I_IR=I_IR, e_I_IR=e_I_IR, I_UV=I_UV, e_I_UV=e_I_UV,
                cosi=gal_cosi, snr_thresh=3)
    # find the best solution given a priority list
    t['Sigma_SFR_gauss'] = np.nan * u.Unit('Msun yr-1 kpc-2')
    t['e_Sigma_SFR_gauss'] = np.nan * u.Unit('Msun yr-1 kpc-2')
    for method in methods:
        if np.isfinite(t[f"Sigma_SFR_{method}_gauss"]).any():
            t['Sigma_SFR_gauss'] = t[f"Sigma_SFR_{method}_gauss"]
            t['e_Sigma_SFR_gauss'] = t[f"e_Sigma_SFR_{method}_gauss"]
            break
    # Halpha+WISE4 prescription
    method = 'HaW4'
    t.calc_surf_dens_sfr(
        # columns to save the output
        colname=f"Sigma_SFR_{method}_gauss",
        colname_e=f"e_Sigma_SFR_{method}_gauss",
        # input parameters
        method=method,
        I_IR=t['I_22um_gauss'], e_I_IR=t['e_I_22um_gauss'],
        I_Halpha=t['I_Halpha_gauss'], e_I_Halpha=t['e_I_Halpha_gauss'],
        cosi=gal_cosi, snr_thresh=3)
    # Av-corrected Halpha
    method = 'Hacorr'
    t.calc_surf_dens_sfr_new(
        # columns to save the output
        colname=f"Sigma_SFR_{method}_gauss",
        colname_e=f"e_Sigma_SFR_{method}_gauss",
        # input parameters
        method=method, I_Halpha=t['I_Halpha_DAP_corr_gauss'],
        e_I_Halpha=t['e_I_Halpha_DAP_corr_gauss'],
        cosi=gal_cosi, snr_thresh=3)
    # recalibrated Halpha+W4
    method = 'HaW4recal'
    t.calc_surf_dens_sfr_new(
        # columns to save the output
        colname=f"Sigma_SFR_{method}_gauss",
        colname_e=f"e_Sigma_SFR_{method}_gauss",
        # input parameters
        method=method, I_Halpha=t['I_Halpha_gauss'],
        e_I_Halpha=t['e_I_Halpha_gauss'],
        I_W1=t['I_3p4um_gauss'], e_I_W1=t['e_I_3p4um_gauss'],
        I_W4=t['I_22um_gauss'], e_I_W4=t['e_I_22um_gauss'],
        cosi=gal_cosi, snr_thresh=3)
    # recalibrated FUV+W4
    method = 'FUVW4recal'
    t.calc_surf_dens_sfr_new(
        # columns to save the output
        colname=f"Sigma_SFR_{method}_gauss",
        colname_e=f"e_Sigma_SFR_{method}_gauss",
        # input parameters
        method=method,
        I_FUV=t['I_154nm_gauss'], e_I_FUV=t['e_I_154nm_gauss'],
        I_W1=t['I_3p4um_gauss'], e_I_W1=t['e_I_3p4um_gauss'],
        I_W4=t['I_22um_gauss'], e_I_W4=t['e_I_22um_gauss'],
        cosi=gal_cosi, snr_thresh=3)

    # Stellar surface density
    if verbose:
        print("    Calculate stellar surface density")
    # IRAC1-based estimate
    t.calc_surf_dens_star(
        # columns to save the output
        colname="Sigma_star_3p6um_gauss",
        colname_e="e_Sigma_star_3p6um_gauss",
        # input parameters
        method='3p6um',
        I_IR=t['I_3p6um_gauss'], e_I_IR=t['e_I_3p6um_gauss'],
        MtoL=t['MtoL_3p4um'], cosi=gal_cosi, snr_thresh=3)
    # WISE1-based estimate
    t.calc_surf_dens_star(
        # columns to save the output
        colname="Sigma_star_3p4um_gauss",
        colname_e="e_Sigma_star_3p4um_gauss",
        # input parameters
        method='3p4um',
        I_IR=t['I_3p4um_gauss'], e_I_IR=t['e_I_3p4um_gauss'],
        MtoL=t['MtoL_3p4um'], cosi=gal_cosi, snr_thresh=3)
    # find the best solution given a priority list
    t['Sigma_star_gauss'] = np.nan * u.Unit('Msun pc-2')
    t['e_Sigma_star_gauss'] = np.nan * u.Unit('Msun pc-2')
    for method in ('3p6um', '3p4um'):
        if np.isfinite(t[f"Sigma_star_{method}_gauss"]).any():
            t['Sigma_star_gauss'] = t[f"Sigma_star_{method}_gauss"]
            t['e_Sigma_star_gauss'] = t[f"e_Sigma_star_{method}_gauss"]
            break

    # Atomic gas surface density
    # !!! temporarily using aperture-averaged measurements !!!
    if verbose:
        print("    Calculate atom gas surface density")
    t.calc_surf_dens_atom(
        # columns to save the output
        colname="Sigma_atom_gauss", colname_e="e_Sigma_atom_gauss",
        # input parameters
        I_HI=t['I_HI'], e_I_HI=t['e_I_HI'],  # <----------------- !!!
        cosi=gal_cosi, snr_thresh=3)

    # Molecular gas surface density
    if verbose:
        print("    Calculate molecular gas surface density")
    t.calc_surf_dens_mol(
        # columns to save the output
        colname="Sigma_mol_gauss", colname_e="e_Sigma_mol_gauss",
        # input parameters
        I_CO=t['I_CO21_gauss'], e_I_CO=t['e_I_CO21_gauss'],
        alpha_CO=t['alpha_CO21'], cosi=gal_cosi, snr_thresh=3)


def build_gauss_table_from_base_table(
        t, res_kpc=None, data_paths=None, output_format=None,
        writefile=None, verbose=True):

    # ------------------------------------------------
    # add gaussian kernel statistics
    # ------------------------------------------------

    add_gauss_average_to_table(
        t, res_kpc=res_kpc, data_paths=data_paths, verbose=verbose)

    # ------------------------------------------------
    # calculate high-level parameters
    # ------------------------------------------------

    calc_high_level_params_in_table(
        t, verbose=verbose)

    # ------------------------------------------------
    # clean and format output table
    # ------------------------------------------------

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
    t_format = Table.read(config_dir / "format_gauss.csv")

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
        tessell_gauss_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content='gauss',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        if (tessell_base_table_file.is_file() and not
                tessell_gauss_table_file.is_file()):
            print("Making gaussian kernel statistics table...")
            t = PhangsGaussKernelMegaTable.read(tessell_base_table_file)
            build_gauss_table_from_base_table(
                t, data_paths=data_paths, output_format=t_format,
                res_kpc=table_configs['tessell_tile_size'],
                writefile=tessell_gauss_table_file)
            print("Done\n")

    # logging settings
    if logging:
        sys.stdout = orig_stdout
        log.close()
