import sys
import json
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.table import Table, QTable

from mega_table.core import StatsTable
from mega_table.table import TessellMegaTable, RadialMegaTable
from mega_table.utils import nanaverage, nanrms, calc_pixel_per_beam

###############################################################################

# location of all relevant config files
config_dir = Path('/data/bell-kant/sun.1608/PHANGS/mega-tables/code')

# location to save the output data tables
work_dir = Path('/data/bell-kant/sun.1608/PHANGS/mega-tables')

# logging setting
logging = False

###############################################################################


class PhangsBaseMegaTable(StatsTable):

    """
    MegaTable for PHANGS base data.
    """

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
        # account for correlated errors among pixels within each beam
        pix_per_beam = calc_pixel_per_beam(hdr)
        hdr.remove('BUNIT', ignore_missing=True)
        self.calc_image_stats(
            (~nanflag).astype('float') / pix_per_beam,
            header=hdr, stat_func=np.sum, colname='_N_beam')
        self['_N_beam'][self['_N_beam'] < 1] = 1
        self[colname_e] /= np.sqrt(self['_N_beam'])
        self.table.remove_column('_N_beam')

    def add_environ_fraction(
            self, colname='env_frac', unit='',
            env_file=None, weight_file=None):

        from reproject import reproject_interp

        if not Path(env_file).is_file():
            self[colname] = np.nan * u.Unit(unit)
            return

        if weight_file is None:
            # no weight
            with fits.open(env_file) as hdul:
                env_map = hdul[0].data.copy()
                hdr = hdul[0].header.copy()
                weight_map = None
        else:
            # with weight
            if not Path(weight_file).is_file():
                self[colname] = np.nan * u.Unit(unit)
                return
            with fits.open(weight_file) as hdul:
                weight_map = hdul[0].data.copy()
                weight_map[~np.isfinite(weight_map)] = 0
                hdr = hdul[0].header.copy()
                hdr.remove('BUNIT', ignore_missing=True)
            with fits.open(env_file) as hdul:
                env_map, footprint = reproject_interp(
                    hdul[0], hdr, order=0)
                env_map[~footprint.astype('?')] = 0

        # calculate (weighted) average of the environ. mask with each tile
        self.calc_image_stats(
            (env_map > 0).astype('float'), header=hdr,
            stat_func=nanaverage, weight=weight_map,
            colname=colname, unit=unit)

    def add_rotation_curve(
            self, colname_V_circ='V_circ', unit_V_circ='km s-1',
            colname_e_V_circ='e_V_circ', unit_e_V_circ='km s-1',
            colname_beta='beta', unit_beta='',
            colname_e_beta='e_beta', unit_e_beta='',
            model_file=None, r_gal_angle=None):

        from scipy import interpolate

        if not Path(model_file).is_file():
            self[colname_V_circ] = np.nan * u.Unit(unit_V_circ)
            self[colname_e_V_circ] = np.nan * u.Unit(unit_e_V_circ)
            self[colname_beta] = np.nan * u.Unit(unit_beta)
            self[colname_e_beta] = np.nan * u.Unit(unit_e_beta)
            return

        # read model file
        t_model = QTable.read(model_file)

        # interpolate models
        models = []
        for model_col in (
                'V_circ', 'V_circ_up', 'V_circ_lo',
                'beta', 'beta_up', 'beta_lo'):
            models += [interpolate.interp1d(
                t_model['r_gal'].to('arcsec').value,
                t_model[model_col].value,
                bounds_error=False, fill_value=np.nan)]

        # sample models
        r = r_gal_angle.to('arcsec').value
        self[colname_V_circ] = (
            models[0](r) * t_model['V_circ'].unit
        ).to(unit_V_circ)
        self[colname_e_V_circ] = (
            (models[1](r) - models[2](r)) / 2 * t_model['V_circ'].unit
        ).to(unit_e_V_circ)
        self[colname_beta] = (
            models[3](r) * t_model['beta'].unit
        ).to(unit_beta)
        self[colname_e_beta] = (
            (models[4](r) - models[5](r)) / 2 * t_model['beta'].unit
        ).to(unit_e_beta)

    def calc_metallicity_pred(
            self, colname='Zprime_scaling', unit='',
            method='PHANGS', Mstar=None, Rstar=None, r_gal=None,
            logOH_solar=None):
        from CO_conversion_factor import metallicity
        if logOH_solar is None:
            logOH_solar = 8.69  # Asplund+09
        if method == 'PHANGS':
            logOH_Re = metallicity.predict_logOH_SAMI19(
                Mstar * 10**0.10)  # Appendix A in Sanchez+19
            logOH = metallicity.extrapolate_logOH_radially(
                logOH_Re, gradient='CALIFA14',
                Rgal=r_gal, Re=Rstar * 1.68)  # Eq. A.3 in Sanchez+14
            self[colname] = (
                10 ** (logOH - logOH_solar) * u.Unit('')).to(unit)
        else:
            raise ValueError(f"Unrecognized method: '{method}'")

    def calc_co_conversion(
            self, colname='alphaCO21', unit='Msun pc-2 K-1 km-1 s',
            method='SL24', Zprime=None, Sigma_star=None, Sigma_SFR=None,
            line_ratio=None):
        from CO_conversion_factor import alphaCO
        if line_ratio is None:
            line_ratio = 0.65  # Leroy+22 (only used for Galactic & S20)
        if method == 'Galactic':
            alphaCO10 = alphaCO.alphaCO10_Galactic
            alphaCO21 = alphaCO10 / line_ratio
        elif method == 'S20':
            alphaCO10 = alphaCO.predict_alphaCO10_S20(Zprime=Zprime)
            alphaCO21 = alphaCO10 / line_ratio
        elif method == 'SL24':
            alphaCO21 = alphaCO.predict_alphaCO_SL24(
                J='2-1', Zprime=Zprime,
                Sigma_star=Sigma_star, Sigma_sfr=Sigma_SFR)
        else:
            raise ValueError(f"Unrecognized method: '{method}'")
        self[colname] = alphaCO21.to(unit)

    def calc_stellar_m2l(
            self, colname='MtoL', unit='Msun Lsun-1',
            method='SFR-to-W1', Sigma_SFR=None,
            I_22um=None, e_I_22um=None, I_3p4um=None, e_I_3p4um=None,
            snr_thresh=None):
        if snr_thresh is None:
            snr_thresh = 3
        if method == 'SFR-to-W1':
            low_snr_flag = (
                (Sigma_SFR == 0) |
                (I_3p4um / e_I_3p4um < snr_thresh))
            a = -11.0  # Leroy+19, Table 6
            b = -0.375
            c = -10.2
            nu_3p4um = const.c / (3.4 * u.um)
            u_Lsun_3p4um = u.Unit('0.042 Lsun')  # solar lum. in W1 band
            Q = np.log10(
                (Sigma_SFR / (4*np.pi*u.sr * nu_3p4um * I_3p4um)
                 ).to(u.Msun / u.yr / u_Lsun_3p4um).value)
        elif method == 'W4-to-W1':
            low_snr_flag = (
                (I_22um / e_I_22um < snr_thresh) |
                (I_3p4um / e_I_3p4um < snr_thresh))
            a = 0.0  # Leroy+19, Table 6
            b = -0.4
            c = 0.75
            Q = np.log10((I_22um / I_3p4um).to('').value)
        else:
            raise ValueError(f"Unrecognized method: '{method}'")
        MtoL = 0.5 + b * (Q - a)
        MtoL[Q < a] = 0.5
        MtoL[Q > c] = 0.2
        MtoL[low_snr_flag | ~np.isfinite(MtoL)] = 0.35
        self[colname] = MtoL * u.Unit(unit)

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

    def calc_surf_dens_sfr_recal(
            self, colname='Sigma_SFR', unit='Msun yr-1 kpc-2',
            colname_e='e_Sigma_SFR', unit_e='Msun yr-1 kpc-2',
            method='Hacorr', I_Halpha=None, e_I_Halpha=None,
            I_FUV=None, e_I_FUV=None, I_W4=None, e_I_W4=None,
            I_W1=None, e_I_W1=None, cosi=1., e_sys=None, snr_thresh=None):
        if e_sys is None:
            e_sys = 0.0
        if snr_thresh is None:
            snr_thresh = 3
        if method == 'HaW4recal':
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
        elif method == '3p6umICA':
            nu_IR = const.c / (3.6 * u.um)
            if MtoL is None:
                MtoL_IR = 0.5 * u.Msun/u.Lsun / 0.037
            else:
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

    def calc_vol_dens_star(
            self, colname='rho_star_mp', unit='Msun pc-3',
            colname_e='e_rho_star_mp', unit_e='Msun pc-3',
            method='flat', Sigma_star=None, e_Sigma_star=None,
            Rstar=None, r_gal=None, flattening=None, e_sys=None):
        if flattening is None:
            flattening = 7.3  # Kregel+02; Sun+20a
        if method == 'flat':
            h_star = Rstar / flattening
        elif method == 'flared':
            h_star = Rstar / flattening * np.exp(
                (r_gal / Rstar).to('').value - 1)
        else:
            raise ValueError(f"Unrecognized method: {method}")
        self[colname] = (Sigma_star / 4 / h_star).to(unit)
        e_stat = e_Sigma_star / 4 / h_star
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)

    def calc_dyn_eq_pressure(
            self, colname='P_DE', unit='K cm-3',
            colname_e='e_P_DE', unit_e='K cm-3',
            Sigma_mol=None, e_Sigma_mol=None,
            Sigma_atom=None, e_Sigma_atom=None,
            rho_star_mp=None, e_rho_star_mp=None,
            vdisp_mol_z=None, vdisp_atom_z=None, e_sys=None):
        if vdisp_atom_z is None:
            vdisp_atom_z = 10 * u.Unit('km s-1')  # Leroy+08, Sun+20a
        if vdisp_mol_z is None:
            vdisp_mol_z = vdisp_atom_z  # Leroy+08
        Sigma_gas = Sigma_atom + Sigma_mol
        vdisp_gas_z = (  # Sun+20a Eq.14
            Sigma_mol * vdisp_mol_z +
            Sigma_atom * vdisp_atom_z) / Sigma_gas
        self[colname] = (  # Sun+20a Eq.12
            (np.pi / 2 * const.G * Sigma_gas**2 +
             Sigma_gas * vdisp_gas_z *
             np.sqrt(2 * const.G * rho_star_mp)) /
            const.k_B).to(unit)
        e_Sigma_gas = np.sqrt(e_Sigma_mol**2 + e_Sigma_atom**2)
        e_stat = np.sqrt(
            (np.pi * const.G * Sigma_gas +
             vdisp_gas_z * np.sqrt(2 * const.G * rho_star_mp))**2 *
            e_Sigma_gas**2 +
            (Sigma_gas * vdisp_gas_z *
             np.sqrt(const.G / 2 / rho_star_mp))**2 *
            e_rho_star_mp**2) / const.k_B
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(e_stat**2 + e_sys**2).to(unit_e)


class PhangsBaseTessellMegaTable(
        TessellMegaTable, PhangsBaseMegaTable):

    """
    TessellMegaTable for PHANGS base data.
    """

    def add_deprojected_coords(
            self, colname_r_gal='r_gal', unit_r_gal='kpc',
            colname_phi_gal='phi_gal', unit_phi_gal='deg',
            dist=None, **kwargs):
        from mega_table.utils import deproject
        r_gal, phi_gal = deproject(**kwargs)
        self[colname_r_gal] = (
            r_gal * u.deg / u.rad * dist).to(unit_r_gal)
        self[colname_phi_gal] = (phi_gal * u.deg).to(unit_phi_gal)


class PhangsBaseRadialMegaTable(
        RadialMegaTable, PhangsBaseMegaTable):

    """
    RadialMegaTable for PHANGS base data.
    """

    def add_linear_r_gal(
            self, colname_r_gal='r_gal', unit_r_gal='kpc',
            r_gal_angl_min=None, r_gal_angl_max=None, dist=None):
        self[colname_r_gal] = (
            (r_gal_angl_min + r_gal_angl_max).to('rad').value / 2 *
            dist).to(unit_r_gal)


# -----------------------------------------------------------------------------


def add_raw_measurements_to_table(
        t, data_paths=None, gal_params=None, verbose=True):

    gal_name = gal_params['name']
    gal_ang2lin = gal_params['dist_Mpc'] * u.Mpc / u.rad

    # GALEX and WISE data (z0MGS)
    if verbose:
        print("  Add GALEX and WISE data")
    products = (
        'FUV', 'NUV',
        'W1', 'W2', 'W3', 'W4')
    colnames = (
        "I_154nm", "I_231nm",
        "I_3p4um", "I_4p6um", "I_12um", "I_22um")
    postfixes = (
        '_gauss7p5', '_gauss7p5',
        '_gauss7p5', '_gauss7p5', '_gauss7p5', '_gauss15')
    for product, colname, postfix in zip(products, colnames, postfixes):
        in_file = data_paths['PHANGS_z0MGS'].format(
            galaxy=gal_name, product=product,
            postfix_resolution=postfix)
        err_file = data_paths['PHANGS_z0MGS'].format(
            galaxy=gal_name, product=f"{product}_err",
            postfix_resolution=postfix)
        t.add_area_average_for_image(
            # column to save the output
            colname=colname, unit='MJy sr-1',
            colname_e=f"e_{colname}", unit_e='MJy sr-1',
            # input parameters
            img_file=in_file, err_file=err_file)

    # Spitzer IRAC data
    if verbose:
        print("  Add Spitzer IRAC data")
    in_file = data_paths['PHANGS_IRAC'].format(
        galaxy=gal_name, product='3p6um',
        postfix_resolution='_gauss3p0')
    err_file = data_paths['PHANGS_IRAC'].format(
        galaxy=gal_name, product='3p6um_err',
        postfix_resolution='_gauss3p0')
    t.add_area_average_for_image(
        # column to save the output
        colname='I_3p6um', unit='MJy sr-1',
        colname_e="e_I_3p6um", unit_e='MJy sr-1',
        # input parameters
        img_file=in_file, err_file=err_file)
    in_file = data_paths['PHANGS_IRAC'].format(
        galaxy=gal_name, product='3p6umICA',
        postfix_resolution='')
    err_file = data_paths['PHANGS_IRAC'].format(
        galaxy=gal_name, product='3p6umICA_err',
        postfix_resolution='')
    t.add_area_average_for_image(
        # column to save the output
        colname='I_3p6umICA', unit='MJy sr-1',
        colname_e="e_I_3p6umICA", unit_e='MJy sr-1',
        # input parameters
        img_file=in_file, err_file=err_file)

    # PHANGS narrow-band Halpha data
    if verbose:
        print("  Add PHANGS narrow-band Halpha data")
    in_file = data_paths['PHANGS_Halpha'].format(
        galaxy=gal_name, product='cleaned',
        postfix_resolution='')
    err_file = data_paths['PHANGS_Halpha'].format(
        galaxy=gal_name, product='err_cleaned',
        postfix_resolution='')
    t.add_area_average_for_image(
        # column to save the output
        colname='I_Halpha', unit='erg s-1 cm-2 arcsec-2',
        colname_e="e_I_Halpha", unit_e='erg s-1 cm-2 arcsec-2',
        # input parameters
        img_file=in_file, err_file=err_file)

    # PHANGS-ALMA CO(2-1) data
    if verbose:
        print("  Add PHANGS-ALMA CO(2-1) data")
    in_file = data_paths['PHANGS_ALMA_CO21'].format(
        galaxy=gal_name, product='mom0',
        postfix_masking='_broad', postfix_resolution='')
    err_file = data_paths['PHANGS_ALMA_CO21'].format(
        galaxy=gal_name, product='emom0',
        postfix_masking='_broad', postfix_resolution='')
    t.add_area_average_for_image(
        # column to save the output
        colname='I_CO21', unit='K km s-1',
        colname_e="e_I_CO21", unit_e='K km s-1',
        # input parameters
        img_file=in_file, err_file=err_file)

    # ALMOND dense gas data
    if verbose:
        print("  Add ALMOND dense gas data")
    in_file = data_paths['PHANGS_ALMOND'].format(
        galaxy=gal_name, line='hcn10',
        product='mom0', resolution='1p5kpc')
    err_file = data_paths['PHANGS_ALMOND'].format(
        galaxy=gal_name, line='hcn10',
        product='emom0', resolution='1p5kpc')
    t.add_area_average_for_image(
        # column to save the output
        colname='I_HCN10', unit='K km s-1',
        colname_e="e_I_HCN10", unit_e='K km s-1',
        # input parameters
        img_file=in_file, err_file=err_file)
    in_file = data_paths['PHANGS_ALMOND'].format(
        galaxy=gal_name, line='hcop10',
        product='mom0', resolution='1p5kpc')
    err_file = data_paths['PHANGS_ALMOND'].format(
        galaxy=gal_name, line='hcop10',
        product='emom0', resolution='1p5kpc')
    t.add_area_average_for_image(
        # column to save the output
        colname='I_HCO+10', unit='K km s-1',
        colname_e="e_I_HCO+10", unit_e='K km s-1',
        # input parameters
        img_file=in_file, err_file=err_file)
    in_file = data_paths['PHANGS_ALMOND'].format(
        galaxy=gal_name, line='cs21',
        product='mom0', resolution='1p5kpc')
    err_file = data_paths['PHANGS_ALMOND'].format(
        galaxy=gal_name, line='cs21',
        product='emom0', resolution='1p5kpc')
    t.add_area_average_for_image(
        # column to save the output
        colname='I_CS21', unit='K km s-1',
        colname_e="e_I_CS21", unit_e='K km s-1',
        # input parameters
        img_file=in_file, err_file=err_file)

    # PHANGS-VLA & archival HI data
    if verbose:
        print("  Add PHANGS-VLA & archival HI data")
    in_file = data_paths['PHANGS_HI'].format(
        galaxy=gal_name, product='broadmask_mom0',
        postfix_resolution='')
    err_file = data_paths['PHANGS_HI'].format(
        galaxy=gal_name, product='broadmask_emom0',
        postfix_resolution='')
    t.add_area_average_for_image(
        # column to save the output
        colname='I_HI', unit='K km s-1',
        colname_e="e_I_HI", unit_e='K km s-1',
        # input parameters
        img_file=in_file, err_file=err_file)

    # PHANGS environmental mask-related measurements
    if verbose:
        print("  Add PHANGS environmental mask-related measurements")
    for env in ('center', 'bars', 'rings', 'lenses', 'sp_arms'):
        in_file = data_paths['PHANGS_env_mask'].format(
            galaxy=gal_name, product=env)
        # fraction of area
        w_file = None
        t.add_environ_fraction(
            # column to save the output
            colname=f"frac_area_{env}",
            # input parameters
            env_file=in_file, weight_file=w_file)
        # fraction of CO(2-1) flux
        w_file = data_paths['PHANGS_ALMA_CO21'].format(
            galaxy=gal_name, product='mom0',
            postfix_masking='_broad', postfix_resolution='')
        t.add_environ_fraction(
            # column to save the output
            colname=f"frac_CO21_{env}",
            # input parameters
            env_file=in_file, weight_file=w_file)
        # fraction of Halpha flux
        w_file = data_paths['PHANGS_Halpha'].format(
            galaxy=gal_name, product='cleaned',
            postfix_resolution='')
        t.add_environ_fraction(
            # column to save the output
            colname=f"frac_Halpha_{env}",
            # input parameters
            env_file=in_file, weight_file=w_file)

    # PHANGS-ALMA rotation curve-derived measurements
    if verbose:
        print("  Add PHANGS-ALMA rotation curve-derived measurements")
    # Universal Rotation Curve (URC) model
    in_file = data_paths['PHANGS_ALMA_rotcurve'].format(
        galaxy=gal_name, product='model_universal')
    t.add_rotation_curve(
        # columns to save the output
        colname_V_circ='V_circ_CO21_URC',
        colname_e_V_circ='e_V_circ_CO21_URC',
        colname_beta='beta_CO21_URC',
        colname_e_beta='e_beta_CO21_URC',
        # input parameters
        model_file=in_file, r_gal_angle=t['r_gal'] / gal_ang2lin)
    # Legendre polynomial model
    in_file = data_paths['PHANGS_ALMA_rotcurve'].format(
        galaxy=gal_name, product='model_legendre')
    t.add_rotation_curve(
        # columns to save the output
        colname_V_circ='V_circ_CO21_lgd',
        colname_e_V_circ='e_V_circ_CO21_lgd',
        colname_beta='beta_CO21_lgd',
        colname_e_beta='e_beta_CO21_lgd',
        # input parameters
        model_file=in_file, r_gal_angle=t['r_gal'] / gal_ang2lin)


def calc_high_level_params_in_table(
        t, gal_params=None, verbose=True):

    gal_name = gal_params['name']
    gal_cosi = np.cos((gal_params['incl_deg'] * u.deg).to('rad').value)
    gal_ang2lin = gal_params['dist_Mpc'] * u.Mpc / u.rad
    gal_Rstar = (
        gal_params['Rstar_arcsec'] * u.arcsec * gal_ang2lin).to('kpc')
    gal_Mstar = gal_params['Mstar_Msun'] * u.Msun

    # Scaling relation-based metallicity
    if verbose:
        print("  Calculate scaling relation-based metallicity")
    t.calc_metallicity_pred(
        # column to save the output
        colname='Zprime_scaling',
        # input parameters
        Mstar=gal_Mstar, Rstar=gal_Rstar, r_gal=t['r_gal'])

    # SFR surface density
    if verbose:
        print("  Calculate SFR surface density")
    # Belfiore+22 Halpha+WISE4 prescription
    method = 'HaW4recal'
    t.calc_surf_dens_sfr_recal(
        # columns to save the output
        colname=f"Sigma_SFR_{method}", colname_e=f"e_Sigma_SFR_{method}",
        # input parameters
        method=method, I_Halpha=t['I_Halpha'], e_I_Halpha=t['e_I_Halpha'],
        I_W1=t['I_3p4um'], e_I_W1=t['e_I_3p4um'],
        I_W4=t['I_22um'], e_I_W4=t['e_I_22um'],
        cosi=gal_cosi, snr_thresh=3)
    # Belfiore+22 FUV+WISE4 prescription
    method = 'FUVW4recal'
    t.calc_surf_dens_sfr_recal(
        # columns to save the output
        colname=f"Sigma_SFR_{method}", colname_e=f"e_Sigma_SFR_{method}",
        # input parameters
        method=method, I_FUV=t['I_154nm'], e_I_FUV=t['e_I_154nm'],
        I_W1=t['I_3p4um'], e_I_W1=t['e_I_3p4um'],
        I_W4=t['I_22um'], e_I_W4=t['e_I_22um'],
        cosi=gal_cosi, snr_thresh=3)
    # Murphy+11 Halpha+WISE4 prescription
    method = 'HaW4'
    t.calc_surf_dens_sfr(
        # columns to save the output
        colname=f"Sigma_SFR_{method}", colname_e=f"e_Sigma_SFR_{method}",
        # input parameters
        method=method, I_Halpha=t['I_Halpha'], e_I_Halpha=t['e_I_Halpha'],
        I_IR=t['I_22um'], e_I_IR=t['e_I_22um'],
        cosi=gal_cosi, snr_thresh=3)
    # Leroy+19 UV+IR prescriptions
    bands_IR = ('W4', 'W3')
    data_IR = (t['I_22um'], t['I_12um'])
    errs_IR = (t['e_I_22um'], t['e_I_12um'])
    bands_UV = ('FUV', 'NUV', '')
    data_UV = (t['I_154nm'], t['I_231nm'], None)
    errs_UV = (t['e_I_154nm'], t['e_I_231nm'], None)
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
                colname=f"Sigma_SFR_{method}",
                colname_e=f"e_Sigma_SFR_{method}",
                # input parameters
                method=method,
                I_IR=I_IR, e_I_IR=e_I_IR, I_UV=I_UV, e_I_UV=e_I_UV,
                cosi=gal_cosi, snr_thresh=3)
    # find the best solution given a priority list
    t['Sigma_SFR'] = np.nan * u.Unit('Msun yr-1 kpc-2')
    t['e_Sigma_SFR'] = np.nan * u.Unit('Msun yr-1 kpc-2')
    for method in ('HaW4recal', 'FUVW4recal', 'W4ONLY'):
        if np.isfinite(t[f"Sigma_SFR_{method}"]).any():
            t['Sigma_SFR'] = t[f"Sigma_SFR_{method}"]
            t['e_Sigma_SFR'] = t[f"e_Sigma_SFR_{method}"]
            break

    # Stellar mass-to-light ratio
    if verbose:
        print("  Calculate stellar mass-to-light ratio")
    if np.isfinite(t['Sigma_SFR_FUVW4']).any():
        # SFR-to-WISE1 color prescription (FUVW4)
        t.calc_stellar_m2l(
            # column to save the output
            colname='MtoL_3p4um',
            # input parameters
            method='SFR-to-W1', Sigma_SFR=t['Sigma_SFR_FUVW4'],
            I_3p4um=t['I_3p4um'], e_I_3p4um=t['e_I_3p4um'])
    elif np.isfinite(t['Sigma_SFR_NUVW4']).any():
        # SFR-to-WISE1 color prescription (NUVW4)
        t.calc_stellar_m2l(
            # column to save the output
            colname='MtoL_3p4um',
            # input parameters
            method='SFR-to-W1', Sigma_SFR=t['Sigma_SFR_NUVW4'],
            I_3p4um=t['I_3p4um'], e_I_3p4um=t['e_I_3p4um'])
    else:
        # WISE4-to-WISE1 color prescription
        t.calc_stellar_m2l(
            # column to save the output
            colname='MtoL_3p4um',
            # input parameters
            method='W4-to-W1', I_22um=t['I_22um'], e_I_22um=t['e_I_22um'],
            I_3p4um=t['I_3p4um'], e_I_3p4um=t['e_I_3p4um'])

    # Stellar surface density
    if verbose:
        print("  Calculate stellar surface density")
    # IRAC1-based estimate
    t.calc_surf_dens_star(
        # columns to save the output
        colname="Sigma_star_3p6um", colname_e="e_Sigma_star_3p6um",
        # input parameters
        method='3p6um', I_IR=t['I_3p6um'], e_I_IR=t['e_I_3p6um'],
        MtoL=t['MtoL_3p4um'], cosi=gal_cosi, snr_thresh=3)
    # WISE1-based estimate
    t.calc_surf_dens_star(
        # columns to save the output
        colname="Sigma_star_3p4um", colname_e="e_Sigma_star_3p4um",
        # input parameters
        method='3p4um', I_IR=t['I_3p4um'], e_I_IR=t['e_I_3p4um'],
        MtoL=t['MtoL_3p4um'], cosi=gal_cosi, snr_thresh=3)
    # ICA-based estimate
    t.calc_surf_dens_star(
        # columns to save the output
        colname="Sigma_star_3p6umICA", colname_e="e_Sigma_star_3p6umICA",
        # input parameters
        method='3p6umICA', I_IR=t['I_3p6umICA'], e_I_IR=t['e_I_3p6umICA'],
        MtoL=0.5*u.Msun/u.Lsun, cosi=gal_cosi, snr_thresh=3)
    # find the best solution given a priority list
    t['Sigma_star'] = np.nan * u.Unit('Msun pc-2')
    t['e_Sigma_star'] = np.nan * u.Unit('Msun pc-2')
    for method in ('3p6um', '3p4um'):
        if np.isfinite(t[f"Sigma_star_{method}"]).any():
            t['Sigma_star'] = t[f"Sigma_star_{method}"]
            t['e_Sigma_star'] = t[f"e_Sigma_star_{method}"]
            break

    # Atomic gas surface density
    if verbose:
        print("  Calculate atom gas surface density")
    t.calc_surf_dens_atom(
        # columns to save the output
        colname="Sigma_atom", colname_e="e_Sigma_atom",
        # input parameters
        I_HI=t['I_HI'], e_I_HI=t['e_I_HI'],
        cosi=gal_cosi, snr_thresh=3)

    # CO-to-H2 conversion factor
    if verbose:
        print("  Calculate CO-to-H2 conversion factor")
    # SL24 prescription (requires filled Mstar & SFR maps...)
    # --- \begin{ugly_patch} ---
    Mstar_file = data_paths['PHANGS_z0MGS'].format(
        galaxy='SFR_Mstar_filled/'+gal_name,
        product='Mstar', postfix_resolution='_gauss7p5')
    t.add_area_average_for_image(
        colname='_Sigma_star_filled', unit='Msun pc-2',
        img_file=Mstar_file)
    for product in ('SFR_FUVW4', 'SFR_NUVW4', 'SFR_W4ONLY'):
        SFR_file = data_paths['PHANGS_z0MGS'].format(
            galaxy='SFR_Mstar_filled/'+gal_name,
            product=product, postfix_resolution='_gauss15')
        t.add_area_average_for_image(
            colname='_Sigma_SFR_filled', unit='Msun yr-1 kpc-2',
            img_file=SFR_file)
        if np.isfinite(t['_Sigma_SFR_filled']).any():
            break
    t['_Sigma_star_filled'] *= gal_cosi
    t['_Sigma_SFR_filled'] *= gal_cosi
    t.calc_co_conversion(
        # columns to save the output
        colname="alpha_CO21_SL24",
        # input parameters
        method='SL24', Zprime=t['Zprime_scaling'],
        Sigma_star=t['_Sigma_star_filled'],
        Sigma_SFR=t['_Sigma_SFR_filled'])
    t.table.remove_columns(['_Sigma_star_filled', '_Sigma_SFR_filled'])
    # --- /end{ugly_patch} ---
    # S20 prescription
    t.calc_co_conversion(
        # columns to save the output
        colname="alpha_CO21_S20",
        # input parameters
        method='S20', Zprime=t['Zprime_scaling'])
    # Galactic value
    t.calc_co_conversion(
        # columns to save the output
        colname="alpha_CO21_Galactic",
        # input parameters
        method='Galactic')
    # find the best solution given a priority list
    t['alpha_CO21'] = np.nan * u.Unit('Msun pc-2 K-1 km-1 s')
    for colname in (
            'alpha_CO21_SL24', 'alpha_CO21_S20', 'alpha_CO21_Galactic'):
        if np.isfinite(t[colname]).any():
            t['alpha_CO21'] = t[colname]
            break

    # Molecular gas surface density
    if verbose:
        print("  Calculate molecular gas surface density")
    t.calc_surf_dens_mol(
        # columns to save the output
        colname="Sigma_mol", colname_e="e_Sigma_mol",
        # input parameters
        I_CO=t['I_CO21'], e_I_CO=t['e_I_CO21'], alpha_CO=t['alpha_CO21'],
        cosi=gal_cosi, snr_thresh=3)

    # Stellar volume density near mid-plane
    if verbose:
        print("  Calculate stellar volume density near mid-plane")
    t.calc_vol_dens_star(
        # columns to save the output
        colname="rho_star_mp", colname_e="e_rho_star_mp",
        # input parameters
        Sigma_star=t['Sigma_star'], e_Sigma_star=t['e_Sigma_star'],
        method='flat', Rstar=gal_Rstar)

    # Dynamical Equilibrium Pressure
    if verbose:
        print("  Calculate dynamical equilibrium pressure")
    t.calc_dyn_eq_pressure(
        # columns to save the output
        colname="P_DE", colname_e="e_P_DE",
        # input parameters
        Sigma_mol=t['Sigma_mol'], e_Sigma_mol=t['e_Sigma_mol'],
        Sigma_atom=t['Sigma_atom'], e_Sigma_atom=t['e_Sigma_atom'],
        rho_star_mp=t['rho_star_mp'], e_rho_star_mp=t['e_rho_star_mp'])


def build_tessell_base_table(
        tile_shape=None, tile_size_kpc=None, fov_radius_R25=None,
        data_paths=None, gal_params=None,
        notes='', version=0.0, output_format=None, writefile=None,
        verbose=True):

    # paraphrase galaxy parameters
    gal_name = gal_params['name']
    gal_ra = np.round(gal_params['ra_deg'], 6) * u.deg
    gal_dec = np.round(gal_params['dec_deg'], 6) * u.deg
    gal_pa = np.round(gal_params['pa_deg'], 1) * u.deg
    gal_incl = np.round(gal_params['incl_deg'], 1) * u.deg
    gal_dist = np.round(gal_params['dist_Mpc'], 2) * u.Mpc
    gal_ang2lin = gal_dist / u.rad
    gal_R25 = np.round((
        gal_params['R25_arcsec'] * u.arcsec * gal_ang2lin).to('kpc'), 2)
    gal_Rstar = np.round((
        gal_params['Rstar_arcsec'] * u.arcsec * gal_ang2lin).to('kpc'), 2)
    gal_Mstar = 10**np.round(
        np.log10(gal_params['Mstar_Msun']), 2) * u.Msun

    # ------------------------------------------------
    # initialize table
    # ------------------------------------------------

    if verbose:
        print("  Initialize table")
    fov_radius_arcsec = (
        fov_radius_R25 * (gal_R25 / gal_ang2lin).to('arcsec').value)
    tile_size_arcsec = (
        (tile_size_kpc * u.kpc / gal_ang2lin).to('arcsec').value)
    t = PhangsBaseTessellMegaTable(
        # galaxy center coordinates
        gal_ra.to('deg').value, gal_dec.to('deg').value,
        # full field-of-view radius in arcsec
        fov_radius_arcsec,
        # individual tile size in arcsec
        tile_size_arcsec,
        # individual tile shape
        tile_shape=tile_shape)

    # ------------------------------------------------
    # add deprojected galactocentric coordinates
    # ------------------------------------------------

    if verbose:
        print("  Add deprojected galactocentric coordinates")
    t.add_deprojected_coords(
        # columns to save the output
        colname_r_gal='r_gal', colname_phi_gal='phi_gal',
        # input parameters
        ra=t['RA'], dec=t['DEC'],
        center_coord=(gal_ra, gal_dec),
        incl=gal_incl, pa=gal_pa, dist=gal_dist)

    # ------------------------------------------------
    # add raw measurements & uncertainties
    # ------------------------------------------------

    add_raw_measurements_to_table(
        t, data_paths=data_paths, gal_params=gal_params,
        verbose=verbose)

    # ------------------------------------------------
    # calculate high-level parameters
    # ------------------------------------------------

    calc_high_level_params_in_table(
        t, gal_params=gal_params, verbose=verbose)

    # ------------------------------------------------
    # clean and format output table
    # ------------------------------------------------

    t.table.sort(['r_gal', 'phi_gal'])
    t['ID'] = np.arange(len(t))
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
    # add metadata
    # ------------------------------------------------

    t.meta['GALAXY'] = str(gal_name)
    t.meta['RA_DEG'] = float(np.round(gal_ra.to('deg').value, 6))
    t.meta['DEC_DEG'] = float(np.round(gal_dec.to('deg').value, 6))
    t.meta['INCL_DEG'] = float(np.round(gal_incl.to('deg').value, 1))
    t.meta['PA_DEG'] = float(np.round(gal_pa.to('deg').value, 1))
    t.meta['DIST_MPC'] = float(np.round(gal_dist.to('Mpc').value, 2))
    t.meta['R25_KPC'] = float(np.round(gal_R25.to('kpc').value, 2))
    t.meta['RSCL_KPC'] = float(np.round(gal_Rstar.to('kpc').value, 2))
    t.meta['LOGMSTAR'] = float(
        np.round(np.log10(gal_Mstar.to('Msun').value), 2))
    t.meta['TBLNOTE'] = str(notes)
    t.meta['VERSION'] = float(version)

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


def build_radial_base_table(
        annulus_width_kpc=None, fov_radius_R25=None,
        data_paths=None, gal_params=None,
        notes='', version=0.0, output_format=None, writefile=None,
        verbose=True):

    # paraphrase galaxy parameters
    gal_name = gal_params['name']
    gal_ra = np.round(gal_params['ra_deg'], 6) * u.deg
    gal_dec = np.round(gal_params['dec_deg'], 6) * u.deg
    gal_pa = np.round(gal_params['pa_deg'], 1) * u.deg
    gal_incl = np.round(gal_params['incl_deg'], 1) * u.deg
    gal_dist = np.round(gal_params['dist_Mpc'], 2) * u.Mpc
    gal_ang2lin = gal_dist / u.rad
    gal_R25 = np.round((
        gal_params['R25_arcsec'] * u.arcsec * gal_ang2lin).to('kpc'), 2)
    gal_Rstar = np.round((
        gal_params['Rstar_arcsec'] * u.arcsec * gal_ang2lin).to('kpc'), 2)
    gal_Mstar = 10**np.round(
        np.log10(gal_params['Mstar_Msun']), 2) * u.Msun

    # ------------------------------------------------
    # initialize table
    # ------------------------------------------------

    if verbose:
        print("  Initialize table")
    rgal_bin_arcsec = (
        (annulus_width_kpc * u.kpc / gal_ang2lin).to('arcsec').value)
    rgal_max_arcsec = (
        fov_radius_R25 * (gal_R25 / gal_ang2lin).to('arcsec').value)
    t = PhangsBaseRadialMegaTable(
        # galaxy center coordinates
        gal_ra.to('deg').value, gal_dec.to('deg').value,
        # radial bin width in arcsec
        rgal_bin_arcsec,
        # full field-of-view radius in arcsec
        rgal_max_arcsec=rgal_max_arcsec,
        # galaxy orientation parameters
        gal_incl_deg=gal_incl.to('deg').value,
        gal_posang_deg=gal_pa.to('deg').value)

    # ------------------------------------------------
    # add galactocentric radius in linear size
    # ------------------------------------------------

    if verbose:
        print("  Add galactocentric radius in linear size")
    t.add_linear_r_gal(
        # columns to save the output
        colname_r_gal='r_gal',
        # input parameters
        r_gal_angl_min=t['r_gal_angl_min'],
        r_gal_angl_max=t['r_gal_angl_max'],
        dist=gal_dist)

    # ------------------------------------------------
    # add raw measurements & uncertainties
    # ------------------------------------------------

    add_raw_measurements_to_table(
        t, data_paths=data_paths, gal_params=gal_params,
        verbose=verbose)

    # ------------------------------------------------
    # calculate high-level parameters
    # ------------------------------------------------

    calc_high_level_params_in_table(
        t, gal_params=gal_params, verbose=verbose)

    # ------------------------------------------------
    # clean and format output table
    # ------------------------------------------------

    t['ID'] = np.arange(len(t))
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
    t.format(
        colnames=colnames, units=units, formats=formats,
        descriptions=descriptions, ignore_missing=True)

    # ------------------------------------------------
    # add metadata
    # ------------------------------------------------

    t.meta['GALAXY'] = str(gal_name)
    t.meta['RA_DEG'] = float(np.round(gal_ra.to('deg').value, 6))
    t.meta['DEC_DEG'] = float(np.round(gal_dec.to('deg').value, 6))
    t.meta['INCL_DEG'] = float(np.round(gal_incl.to('deg').value, 1))
    t.meta['PA_DEG'] = float(np.round(gal_pa.to('deg').value, 1))
    t.meta['DIST_MPC'] = float(np.round(gal_dist.to('Mpc').value, 2))
    t.meta['R25_KPC'] = float(np.round(gal_R25.to('kpc').value, 2))
    t.meta['RSCL_KPC'] = float(np.round(gal_Rstar.to('kpc').value, 2))
    t.meta['LOGMSTAR'] = float(
        np.round(np.log10(gal_Mstar.to('Msun').value), 2))
    t.meta['TBLNOTE'] = str(notes)
    t.meta['VERSION'] = float(version)

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

    # load config files
    with open(config_dir / "config_data_path.json") as f:
        data_paths = json.load(f)
    with open(config_dir / "config_tables.json") as f:
        table_configs = json.load(f)
    t_format = Table.read(config_dir / "format_base.csv")

    # read sample table
    t_sample = Table.read(data_paths['PHANGS_sample_table'])

    # sub-select sample
    t_sample = t_sample[t_sample['data_has_megatable']]

    # loop through all galaxies
    for row in t_sample:

        # extract galaxy parameters
        gal_params = {
            'name': row['name'].upper(),
            'dist_Mpc': row['dist'],
            'ra_deg': row['orient_ra'],
            'dec_deg': row['orient_dec'],
            'incl_deg': row['orient_incl'],
            'pa_deg': row['orient_posang'],
            'Rstar_arcsec': row['size_scalelength'],
            'R25_arcsec': row['size_r25'],
            'Mstar_Msun': row['props_mstar'],
        }

        print("\n############################################################")
        print(f"# {gal_params['name']}")
        print("############################################################\n")

        # TessellMegaTable
        tile_shape = table_configs['tessell_tile_shape']
        tile_size = (
            table_configs['tessell_tile_size'] *
            u.Unit(table_configs['tessell_tile_size_unit']))
        tile_size_str = (
            str(table_configs['tessell_tile_size']).replace('.', 'p') +
            table_configs['tessell_tile_size_unit'])
        fov_radius_R25 = table_configs['tessell_FoV_radius']

        tessell_base_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_params['name'], content='base',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        if not tessell_base_table_file.is_file():
            print("Building tessellation statistics table...")
            build_tessell_base_table(
                tile_shape=tile_shape,
                tile_size_kpc=tile_size.to('kpc').value,
                fov_radius_R25=fov_radius_R25,
                data_paths=data_paths,
                gal_params=gal_params,
                version=table_configs['table_version'],
                notes=table_configs['table_notes'],
                output_format=t_format,
                writefile=tessell_base_table_file)
            print("Done\n")

        # RadialMegaTable
        annulus_width = (
            table_configs['radial_annulus_width'] *
            u.Unit(table_configs['radial_annulus_width_unit']))
        annulus_width_str = (
            str(table_configs['radial_annulus_width']).replace('.', 'p') +
            table_configs['radial_annulus_width_unit'])
        fov_radius_R25 = table_configs['radial_FoV_radius']

        radial_base_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_params['name'], content='base',
                annulus_width_str=annulus_width_str))
        if not radial_base_table_file.is_file():
            print("Building radial statistics table...")
            build_radial_base_table(
                annulus_width_kpc=annulus_width.to('kpc').value,
                fov_radius_R25=fov_radius_R25,
                data_paths=data_paths,
                gal_params=gal_params,
                version=table_configs['table_version'],
                notes=table_configs['table_notes'],
                output_format=t_format,
                writefile=radial_base_table_file)
            print("Done\n")

    # logging settings
    if logging:
        sys.stdout = orig_stdout
        log.close()
