import sys
import json
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.table import Table

from mega_table.core import StatsTable
from mega_table.table import TessellMegaTable, RadialMegaTable
from mega_table.utils import nanaverage, calc_pixel_area, calc_pixel_per_beam

###############################################################################

# location of all relevant config files
config_dir = Path('/data/kant/0/sun.1608/PHANGS/mega-tables/code')

# location to save the output data tables
work_dir = Path('/data/kant/0/sun.1608/PHANGS/mega-tables')

# logging setting
logging = False

###############################################################################


class PhangsAlmaMegaTable(StatsTable):

    """
    MegaTable for PHANGS-ALMA data.
    """

    # -------------------------------------------------------------------------
    # methods for object-based statistics
    # -------------------------------------------------------------------------

    def add_obj_count(
            self, colname='N_obj', unit='',
            ra=None, dec=None, flux=None):
        if ra is None or dec is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        self.calc_catalog_stats(
            np.isfinite(flux).astype('int'), ra, dec,
            stat_func=np.nansum, colname=colname, unit=unit)

    def add_obj_flux_frac(
            self, colname='fracF_CO21_obj', unit='',
            ra=None, dec=None, flux=None,
            header=None, broad_mom0=None, dist=None):
        if ra is None or dec is None or header is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        self.calc_catalog_stats(
            flux.value, ra, dec,
            stat_func=np.nansum, colname='_sum(F_obj)', unit=flux.unit)
        self.calc_image_stats(
            broad_mom0.value, header=header,
            stat_func=np.nansum, colname='_sum(I_bm0)', unit=broad_mom0.unit)
        pix_area = (
            calc_pixel_area(header).to('sr').value * dist**2).to('pc2')
        self[colname] = (
            self['_sum(F_obj)'] / (self['_sum(I_bm0)'] * pix_area)).to(unit)
        self.table.remove_columns(['_sum(F_obj)', '_sum(I_bm0)'])

    def add_obj_stat_generic(
            self, colname=None, unit=None,
            ra=None, dec=None, prefactor=None, flux=None, flux_power=0.,
            radius=None, radius_power=0., vdisp=None, vdisp_power=0.):
        # check input
        if ra is None or dec is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        if radius_power == 0 and radius is None:
            radius = np.ones(flux.shape)
        if vdisp_power == 0 and vdisp is None:
            vdisp = np.ones(flux.shape)
        # construct weight image
        wt = flux.value.copy()
        wt[~np.isfinite(wt)] = 0
        # calculate intensity-weighted average for the quantity
        y = (
            prefactor * flux**flux_power *
            radius**radius_power * vdisp**vdisp_power).to(unit)
        self.calc_catalog_stats(
            y.value, ra, dec, weight=wt,
            stat_func=nanaverage, colname=colname, unit=y.unit)

    def add_obj_co_flux(
            self, colname='<F_CO21_obj>', unit='K km s-1 pc2',
            ra=None, dec=None, flux=None, complete_corr=None):
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=1.,
            flux=flux, flux_power=1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_mass(
            self, colname='<M_mol_obj>', unit='Msun',
            ra=None, dec=None, flux=None, alpha_CO=None,
            complete_corr=None):
        prefactor = alpha_CO.unit
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=prefactor,
            flux=flux, flux_power=1)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_co_linewidth(
            self, colname='<sigmav_CO21_obj>', unit='km s-1',
            ra=None, dec=None, flux=None, vdisp=None,
            complete_corr=None):
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=1.,
            flux=flux, vdisp=vdisp, vdisp_power=1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_vel_disp(
            self, colname='<vdisp_mol_obj>', unit='km s-1',
            ra=None, dec=None, flux=None, vdisp=None,
            cosi=None, complete_corr=None):
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=cosi**0.5,
            flux=flux, vdisp=vdisp, vdisp_power=1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_radius(
            self, colname='<R_obj>', unit='pc',
            ra=None, dec=None, flux=None, radius=None,
            complete_corr=None):
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=1.,
            flux=flux, radius=radius, radius_power=1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_area(
            self, colname='<Area_obj>', unit='pc2',
            ra=None, dec=None, flux=None, radius=None,
            cosi=None, complete_corr=None):
        prefactor = np.pi / (2 * np.log(2)) / cosi
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=prefactor,
            flux=flux, radius=radius, radius_power=2)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_surf_density(
            self, colname='<Sigma_mol_obj>', unit='Msun pc-2',
            ra=None, dec=None, flux=None, radius=None,
            alpha_CO=None, cosi=None, complete_corr=None):
        # Rosolowsky+21, text following Eq.12
        prefactor = alpha_CO.unit * cosi / (2 * np.pi)
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=prefactor,
            flux=flux, flux_power=1, radius=radius, radius_power=-2)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_freefall_time(
            self, colname='<t_ff^-1_obj>', unit='Myr-1',
            ra=None, dec=None, flux=None, radius=None,
            alpha_CO=None, complete_corr=None):
        # Rosolowsky+21, Eq.17
        prefactor = np.sqrt(4 * const.G * alpha_CO.unit / np.pi**2)
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=prefactor,
            flux=flux, flux_power=0.5, radius=radius, radius_power=-1.5)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value**0.5
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_crossing_time(
            self, colname='<t_cross^-1_obj>', unit='Myr-1',
            ra=None, dec=None, flux=None, vdisp=None, radius=None,
            cosi=None, complete_corr=None):
        prefactor = cosi**0.5
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=prefactor,
            flux=flux, vdisp=vdisp, vdisp_power=1,
            radius=radius, radius_power=-1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_virial_param(
            self, colname='<alpha_vir_obj>', unit='',
            ra=None, dec=None, flux=None, vdisp=None, radius=None,
            alpha_CO=None, cosi=None, complete_corr=None):
        # Rosolowsky+21, text following Eq.10
        prefactor = 10 * cosi / (const.G * alpha_CO.unit)
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=prefactor,
            flux=flux, flux_power=-1, vdisp=vdisp, vdisp_power=2,
            radius=radius, radius_power=1)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value**-1
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    def add_obj_turb_pressure(
            self, colname='<P_turb_obj>', unit='K cm-3',
            ra=None, dec=None, flux=None, vdisp=None, radius=None,
            alpha_CO=None, cosi=None, complete_corr=None):
        # Rosolowsky+21, Eq.16
        prefactor = 3/8 / np.pi * alpha_CO.unit * cosi / const.k_B
        self.add_obj_stat_generic(
            colname=colname, unit=unit,
            ra=ra, dec=dec, prefactor=prefactor,
            flux=flux, flux_power=1, vdisp=vdisp, vdisp_power=2,
            radius=radius, radius_power=-3)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr

    # -------------------------------------------------------------------------
    # methods for pixel-based statistics
    # -------------------------------------------------------------------------

    def add_pix_area_frac(
            self, colname='fracA_CO21_pix', unit='',
            header=None, orig_mom0=None, masked_mom0=None):
        if header is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        self.calc_image_stats(
            np.isfinite(masked_mom0).astype('float'),
            header=header, stat_func=np.nansum,
            colname='_A_detected', unit='')
        self.calc_image_stats(
            np.isfinite(orig_mom0).astype('float'),
            header=header, stat_func=np.nansum,
            colname='_A_total', unit='')
        self[colname] = (
            self['_A_detected'] / self['_A_total']).to(unit)
        self.table.remove_columns(['_A_detected', '_A_total'])

    def add_pix_flux_frac(
            self, colname='fracF_CO21_pix', unit='',
            header=None, strict_mom0=None, broad_mom0=None):
        if header is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        self.calc_image_stats(
            strict_mom0.value, header=header, stat_func=np.nansum,
            colname='_F_strict', unit=strict_mom0.unit)
        self.calc_image_stats(
            broad_mom0.value, header=header, stat_func=np.nansum,
            colname='_F_broad', unit=broad_mom0.unit)
        self[colname] = (
            self['_F_strict'] / self['_F_broad']).to(unit)
        self.table.remove_columns(['_F_strict', '_F_broad'])

    def calc_completeness_corr(
            self, colname_corr_I='corr_I_CO21_pix', unit_corr_I='',
            colname_corr_c='corr_c_CO21_pix', unit_corr_c='',
            colname_corr_t='corr_t_ff^-1_pix', unit_corr_t='',
            colname_corr_P='corr_P_selfg_pix', unit_corr_P='',
            fracA=None, fracF=None):
        from scipy.special import erf, erfinv
        fA = fracA.to('').value.copy()
        fF = fracF.to('').value.copy()
        fF[fF < 0] = 0
        fF[fF > 1] = 1
        fF[fF < fA] = fA[fF < fA]
        fF_2 = 1/2 * (
            1 - erf(2 * erfinv(1-2*fF) - erfinv(1-2*fA)))
        fF_2[(fA == 1) & (fF == 1)] = 1
        fF_1p5 = 1/2 * (
            1 - erf(1.5 * erfinv(1-2*fF) - 0.5 * erfinv(1-2*fA)))
        fF_1p5[(fA == 1) & (fF == 1)] = 1
        fF_3 = 1/2 * (
            1 - erf(3 * erfinv(1-2*fF) - 2 * erfinv(1-2*fA)))
        fF_3[(fA == 1) & (fF == 1)] = 1
        self[colname_corr_I] = (
            fF / fF_2 * u.Unit('')).to(unit_corr_I)
        self[colname_corr_c] = (
            fF**2 / fF_2 / fA * u.Unit('')).to(unit_corr_c)
        self[colname_corr_t] = (
            fF / fF_1p5 * u.Unit('')).to(unit_corr_t)
        self[colname_corr_P] = (
            fF / fF_3 * u.Unit('')).to(unit_corr_P)

    def add_clumping(
            self, colname='c_CO21_pix', unit='',
            colname_e='e_c_CO21_pix', unit_e='',
            header=None, masked_mom0=None, masked_emom0=None,
            complete_corr=None):
        if header is None:
            self[colname] = np.nan * u.Unit(unit)
            self[colname_e] = np.nan * u.Unit(unit_e)
            return
        wt = masked_mom0.value.copy()
        wt[~np.isfinite(wt)] = 0
        # calculate clumping factor
        self.calc_image_stats(
            masked_mom0.value, weight=wt,
            header=header, stat_func=nanaverage,
            colname='_F<I>', unit=masked_mom0.unit)
        self.calc_image_stats(
            masked_mom0.value, weight=None,
            header=header, stat_func=nanaverage,
            colname='_A<I>', unit=masked_mom0.unit)
        self[colname] = (self['_F<I>'] / self['_A<I>']).to(unit)
        # calculate uncertainty on the clumping factor
        avg_mom0 = t.create_maps_from_columns(['_F<I>'], header)[0]
        self.calc_image_stats(
            (masked_emom0**2 * (masked_mom0 - avg_mom0)**2).value,
            header=header, stat_func=np.nansum, colname='_sqsum(err)',
            unit=masked_emom0.unit**2 * masked_mom0.unit**2)
        self.calc_image_stats(
            masked_emom0.value, header=header, stat_func=np.nansum,
            colname='_sum(I)', unit=masked_emom0.unit)
        self[colname_e] = (
            2 * np.sqrt(self['_sqsum(err)']) / self['_sum(I)'] / self['_A<I>'])
        self.table.remove_columns(
            ['_F<I>', '_A<I>', '_sqsum(err)', '_sum(I)'])
        # account for correlated errors among pixels within each beam
        pix_per_beam = calc_pixel_per_beam(header)
        self[colname_e] *= np.sqrt(pix_per_beam)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # convert uncertainty unit
        self[colname_e] = self[colname_e].to(unit_e)

    def add_pix_stat_generic(
            self, colname=None, colname_e=None, unit=None,
            header=None, prefactor=None,
            m0=None, em0=None, m0_power=0.,
            ew=None, eew=None, ew_power=0.):
        # check input
        if header is None:
            self[colname] = np.nan * u.Unit(unit)
            if colname_e is not None:
                self[colname_e] = np.nan * u.Unit(unit)
            return
        if ew_power == 0:
            if ew is None:
                ew = np.ones(m0.shape)
            if eew is None:
                eew = np.zeros(m0.shape)
        # construct weight image
        wt = m0.value.copy()
        wt[~np.isfinite(wt)] = 0
        # calculate intensity-weighted average for the quantity
        y = (prefactor * m0**m0_power * ew**ew_power).to(unit)
        self.calc_image_stats(
            y.value, weight=wt, header=header, stat_func=nanaverage,
            colname=colname, unit=y.unit)
        if colname_e is None:
            return
        # calculate formal uncertainty on the averaged value
        avg_y = self.create_maps_from_columns([colname], header)[0]
        self.calc_image_stats(
            (((m0_power+1) * y - avg_y)**2 * em0**2 +
             (ew_power * y * m0 / ew)**2 * eew**2).value,
            header=header, stat_func=np.nansum, colname='_sqsum(err)',
            unit=y.unit**2 * m0.unit**2)
        self.calc_image_stats(
            m0.value, header=header, stat_func=np.nansum,
            colname='_sum(I)', unit=m0.unit)
        self[colname_e] = (
            np.sqrt(self['_sqsum(err)']) / self['_sum(I)']).to(unit)
        self.table.remove_columns(['_sqsum(err)', '_sum(I)'])
        # account for correlated errors among pixels within each beam
        pix_per_beam = calc_pixel_per_beam(header)
        self[colname_e] *= np.sqrt(pix_per_beam)

    def add_pix_co_intensity(
            self, colname='<I_CO21_pix>', unit='K km s-1',
            colname_e='e_<I_CO21_pix>', unit_e='K km s-1',
            header=None, masked_mom0=None, masked_emom0=None,
            complete_corr=None):
        self.add_pix_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            header=header, prefactor=1.,
            m0=masked_mom0, em0=masked_emom0, m0_power=1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # convert uncertainty unit
        self[colname_e] = self[colname_e].to(unit_e)

    def add_pix_surf_density(
            self, colname='<Sigma_mol_pix>', unit='Msun pc-2',
            colname_e='e_<Sigma_mol_pix>', unit_e='Msun pc-2',
            header=None, masked_mom0=None, masked_emom0=None,
            alpha_CO=None, cosi=None, complete_corr=None, e_sys=None):
        self.add_pix_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            header=header, prefactor=alpha_CO.unit * cosi,
            m0=masked_mom0, em0=masked_emom0, m0_power=1)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value
        self[colname_e] *= alpha_CO.value
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # combine statistical and systematic uncertainties
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(self[colname_e]**2 + e_sys**2).to(unit_e)

    def add_pix_co_linewidth(
            self, colname='<sigmav_CO21_pix>', unit='km s-1',
            colname_e='e_<sigmav_CO21_pix>', unit_e='km s-1',
            header=None, masked_ew=None, masked_eew=None,
            masked_mom0=None, masked_emom0=None, complete_corr=None):
        self.add_pix_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            header=header, prefactor=1.,
            m0=masked_mom0, em0=masked_emom0, m0_power=0,
            ew=masked_ew, eew=masked_eew, ew_power=1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # convert uncertainty unit
        self[colname_e] = self[colname_e].to(unit_e)

    def add_pix_vel_disp(
            self, colname='<vdisp_mol_pix>', unit='km s-1',
            colname_e='e_<vdisp_mol_pix>', unit_e='km s-1',
            header=None, masked_ew=None, masked_eew=None,
            masked_mom0=None, masked_emom0=None, cosi=None,
            complete_corr=None, e_sys=None):
        self.add_pix_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            header=header, prefactor=cosi**0.5,
            m0=masked_mom0, em0=masked_emom0, m0_power=0,
            ew=masked_ew, eew=masked_eew, ew_power=1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # combine statistical and systematic uncertainties
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(self[colname_e]**2 + e_sys**2).to(unit_e)

    def add_pix_freefall_time(
            self, colname='<t_ff^-1_pix>', unit='Myr-1',
            colname_e='e_<t_ff^-1_pix>', unit_e='Myr-1',
            header=None, masked_mom0=None, masked_emom0=None,
            FWHM_beam=None, radius=None, alpha_CO=None,
            complete_corr=None, e_sys=None):
        area_beam = np.pi / (4 * np.log(2)) * FWHM_beam**2
        prefactor = np.sqrt(
            8 * const.G * alpha_CO.unit * area_beam / np.pi**2 / radius**3)
        self.add_pix_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            header=header, prefactor=prefactor,
            m0=masked_mom0, em0=masked_emom0, m0_power=0.5)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value**0.5
        self[colname_e] *= alpha_CO.value**0.5
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # combine statistical and systematic uncertainties
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(self[colname_e]**2 + e_sys**2).to(unit_e)

    def add_pix_crossing_time(
            self, colname='<t_cross^-1_pix>', unit='Myr-1',
            colname_e='e_<t_cross^-1_pix>', unit_e='Myr-1',
            header=None, masked_ew=None, masked_eew=None,
            masked_mom0=None, masked_emom0=None,
            radius=None, cosi=None, complete_corr=None, e_sys=None):
        prefactor = cosi**0.5 / radius
        self.add_pix_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            header=header, prefactor=prefactor,
            m0=masked_mom0, em0=masked_emom0, m0_power=0,
            ew=masked_ew, eew=masked_eew, ew_power=1)
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # combine statistical and systematic uncertainties
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(self[colname_e]**2 + e_sys**2).to(unit_e)

    def add_pix_virial_param(
            self, colname='<alpha_vir_pix>', unit='',
            colname_e='e_<alpha_vir_pix>', unit_e='',
            header=None, masked_ew=None, masked_eew=None,
            masked_mom0=None, masked_emom0=None,
            FWHM_beam=None, radius=None, alpha_CO=None, cosi=None,
            complete_corr=None, e_sys=None):
        area_beam = np.pi / (4 * np.log(2)) * FWHM_beam**2
        prefactor = 5 * cosi * radius / (const.G * alpha_CO.unit * area_beam)
        self.add_pix_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            header=header, prefactor=prefactor,
            m0=masked_mom0, em0=masked_emom0, m0_power=-1,
            ew=masked_ew, eew=masked_eew, ew_power=2)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value**-1
        self[colname_e] *= alpha_CO.value**-1
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # combine statistical and systematic uncertainties
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(self[colname_e]**2 + e_sys**2).to(unit_e)

    def add_pix_turb_pressure(
            self, colname='<P_turb_pix>', unit='K cm-3',
            colname_e='e_<P_turb_pix>', unit_e='K cm-3',
            header=None, masked_ew=None, masked_eew=None,
            masked_mom0=None, masked_emom0=None,
            FWHM_beam=None, radius=None, alpha_CO=None, cosi=None,
            complete_corr=None, e_sys=None):
        area_beam = np.pi / (4 * np.log(2)) * FWHM_beam**2
        volume = 4/3 * np.pi * radius**3
        prefactor = alpha_CO.unit * cosi * area_beam / volume / const.k_B
        self.add_pix_stat_generic(
            colname=colname, colname_e=colname_e, unit=unit,
            header=header, prefactor=prefactor,
            m0=masked_mom0, em0=masked_emom0, m0_power=1,
            ew=masked_ew, eew=masked_eew, ew_power=2)
        # scale results according to alpha_CO
        self[colname] *= alpha_CO.value
        self[colname_e] *= alpha_CO.value
        # apply completeness corrections
        if complete_corr is not None:
            self[colname] *= complete_corr
            self[colname_e] *= complete_corr
        # combine statistical and systematic uncertainties
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(self[colname_e]**2 + e_sys**2).to(unit_e)

    def add_pix_dyn_eq_pressure(
            self, colname='<P_DE_pix>', unit='K cm-3',
            colname_e='e_<P_DE_pix>', unit_e='K cm-3',
            header=None, masked_mom0=None, masked_emom0=None,
            Sigma_mol_cloud=None, e_Sigma_mol_cloud=None,
            Sigma_mol_kpc=None, e_Sigma_mol_kpc=None,
            Sigma_atom_kpc=None, e_Sigma_atom_kpc=None,
            vdisp_atom_z=None, rho_star_mp=None, e_rho_star_mp=None,
            FWHM_beam=None, radius=None, alpha_CO=None, cosi=None,
            complete_corr=None, e_sys=None):
        if vdisp_atom_z is None:
            vdisp_atom_z = 10 * u.Unit('km s-1')  # Leroy+08, Sun+20a
        # calculate cloud hydrostatic pressure due to self-gravity
        area_beam = np.pi / (4 * np.log(2)) * FWHM_beam**2
        prefactor = (
            3/8 * np.pi * const.G / const.k_B *
            (alpha_CO.unit * cosi * area_beam / (np.pi * radius**2))**2)
        self.add_pix_stat_generic(
            colname='_<P_selfg>', colname_e='_e_<P_selfg>', unit=unit,
            header=header, prefactor=prefactor,
            m0=masked_mom0, em0=masked_emom0, m0_power=2)
        P_cl_selfg = self['_<P_selfg>'] * alpha_CO.value**2
        e_P_cl_selfg = self['_e_<P_selfg>'] * alpha_CO.value**2
        if complete_corr is not None:
            P_cl_selfg *= complete_corr
            e_P_cl_selfg *= complete_corr
        self.table.remove_columns(['_<P_selfg>', '_e_<P_selfg>'])
        # calculate cloud hydrostatic pressure due to external gravity
        P_cl_molg = (
            1/2 * np.pi * const.G / const.k_B *
            Sigma_mol_cloud * Sigma_mol_kpc)
        P_cl_starg = (
            3/4 * np.pi * const.G / const.k_B *
            Sigma_mol_cloud * rho_star_mp * radius*2)
        # calculate ambient pressure in atomic gas
        P_amb_selfg = (
            1/2 * np.pi * const.G / const.k_B * Sigma_atom_kpc**2)
        P_amb_molg = (
            np.pi * const.G / const.k_B *
            Sigma_atom_kpc * Sigma_mol_kpc)
        P_amb_starg = (
            Sigma_atom_kpc * vdisp_atom_z / const.k_B *
            np.sqrt(2 * const.G * rho_star_mp))
        # combine all terms to get dynamical equilibrium pressure
        P_DE = P_cl_selfg + P_cl_molg + P_cl_starg + \
            P_amb_selfg + P_amb_molg + P_amb_starg
        self[colname] = P_DE.to(unit)
        # calculate uncertainty on dynamical equilibrium pressure
        if e_sys is None:
            e_sys = 0.0
        self[colname_e] = np.sqrt(
            e_P_cl_selfg**2 +
            ((P_cl_molg + P_cl_starg) *
             e_Sigma_mol_cloud / Sigma_mol_cloud)**2 +
            ((P_amb_selfg*2 + P_amb_molg + P_amb_starg) *
             e_Sigma_atom_kpc / Sigma_atom_kpc)**2 +
            ((P_cl_molg + P_amb_molg) *
             e_Sigma_mol_kpc / Sigma_mol_kpc)**2 +
            ((P_cl_starg + P_amb_starg/2) *
             e_rho_star_mp / rho_star_mp)**2 +
            e_sys**2).to(unit_e)

    def add_pix_co_conversion_g20(
            self, colname='<alpha_CO21_G20ICO>',
            unit='Msun pc-2 K-1 km-1 s',
            header=None, masked_mom0=None, Zprime=None,
            FWHM_beam=None, cosi=None, force_res_dependence=True):
        if header is None:
            self[colname] = np.nan * u.Unit(unit)
            return
        # Gong et al. (2020), Table 3
        if not force_res_dependence and (FWHM_beam >= 100*u.pc):
            # Without W_CO or scale dependence (expression 1b)
            self['_X_CO21'] = 2.0e20 * u.Unit('cm-2 K-1 km-1 s')
        else:
            # With W_CO and scale dependence (expression 4b)
            power = -0.97 + 0.34 * np.log10(FWHM_beam.to('pc').value)
            prefactor = 21.1e20 * u.Unit('cm-2 K-1 km-1 s')
            prefactor *= FWHM_beam.to('pc').value**-0.41
            unitless_mom0 = \
                masked_mom0.to('K km s-1').value * cosi * u.Unit('')
            self.add_pix_stat_generic(
                colname='_X_CO21', unit=prefactor.unit,
                header=header, prefactor=prefactor,
                m0=unitless_mom0, m0_power=power)
        # Add metallicity dependence
        self['_X_CO21'] *= Zprime**-0.5
        # Convert to surface density units
        self[colname] = (self['_X_CO21'] * const.u * 2.016).to(unit)
        # Add Helium correction
        self[colname] *= 1.35

    def calc_co_conversion(
            self, colname='alpha_CO21', unit='Msun pc-2 K-1 km-1 s',
            method='N12', line_ratio=None, Zprime=None,
            I_CO_cloud=None, vdisp_mol_150pc=None,
            I_CO_kpc=None, Sigma_else_kpc=None):
        from CO_conversion_factor import alphaCO
        if line_ratio is None:
            line_ratio = 0.65  # Leroy+22 (only used for N12 & B13)
        if method == 'N12':
            alphaCO10 = alphaCO.predict_alphaCO10_N12(
                Zprime=Zprime, WCO10GMC=I_CO_cloud/line_ratio)
            alphaCO21 = alphaCO10 / line_ratio
        elif method == 'B13':
            alphaCO10 = alphaCO.predict_alphaCO10_B13(
                Zprime=Zprime, WCO10kpc=I_CO_kpc/line_ratio,
                Sigmaelsekpc=Sigma_else_kpc, suppress_error=True)
            alphaCO21 = alphaCO10 / line_ratio
        elif method == 'T24':
            alphaCO21 = alphaCO.predict_alphaCO21_T24(
                vdisp_150pc=vdisp_mol_150pc)
        elif method == 'T24alt':
            alphaCO21 = alphaCO.predict_alphaCO21_T24(
                vdisp_150pc=vdisp_mol_150pc,
                add_metal=True, Zprime=Zprime)
        else:
            raise ValueError(f"Unrecognized method: '{method}'")
        self[colname] = alphaCO21.to(unit)


class PhangsAlmaTessellMegaTable(TessellMegaTable, PhangsAlmaMegaTable):

    """
    TessellMegaTable for PHANGS-ALMA data.
    """


class PhangsAlmaRadialMegaTable(RadialMegaTable, PhangsAlmaMegaTable):

    """
    RadialMegaTable for PHANGS-ALMA data.
    """


# -----------------------------------------------------------------------------


def add_pixel_stats_to_table(
        t: PhangsAlmaMegaTable, data_paths=None, res_pcs=None,
        colname_alphaCO='alpha_CO21', H_los=None, verbose=True):

    gal_name = t.meta['GALAXY']
    gal_dist = t.meta['DIST_MPC'] * u.Mpc
    gal_cosi = np.cos(np.deg2rad(t.meta['INCL_DEG']))

    if colname_alphaCO not in t.colnames:
        raise ValueError("No alphaCO column in table")

    if H_los is None:
        H_los = 100 * u.pc / gal_cosi  # Heyer & Dame (2015)

    for res_pc in res_pcs:

        if res_pc is None:
            res_str = 'native'
            postfix_res = ''
        else:
            res_str = f"{res_pc}pc"
            postfix_res = f"_{res_pc}pc"
        if verbose:
            print(f"  Add pixel-based GMC statistics @ {res_str} resolution")

        # read data files
        bm0_file = Path(data_paths['PHANGS_ALMA_CO21'].format(
            galaxy=gal_name, product='mom0',
            postfix_masking='_broad', postfix_resolution=postfix_res))
        sm0_file = Path(data_paths['PHANGS_ALMA_CO21'].format(
            galaxy=gal_name, product='mom0',
            postfix_masking='_strict', postfix_resolution=postfix_res))
        sem0_file = Path(data_paths['PHANGS_ALMA_CO21'].format(
            galaxy=gal_name, product='emom0',
            postfix_masking='_strict', postfix_resolution=postfix_res))
        sew_file = Path(data_paths['PHANGS_ALMA_CO21'].format(
            galaxy=gal_name, product='ew',
            postfix_masking='_strict', postfix_resolution=postfix_res))
        seew_file = Path(data_paths['PHANGS_ALMA_CO21'].format(
            galaxy=gal_name, product='eew',
            postfix_masking='_strict', postfix_resolution=postfix_res))
        if not (bm0_file.is_file() and
                sm0_file.is_file() and sem0_file.is_file() and
                sew_file.is_file() and seew_file.is_file()):
            if verbose:
                print("    Some input file not found -- will do a dry run")
            hdr = bm0 = orig_sm0 = sm0 = sem0 = sew = seew = None
            res = radius3d = np.nan * u.pc
        else:
            if verbose:
                print("    All input file found")
            with fits.open(bm0_file) as hdul:
                hdr = hdul[0].header.copy()
                hdr.remove('BUNIT')
                bm0 = hdul[0].data * u.Unit(hdul[0].header['BUNIT'])
            with fits.open(sm0_file) as hdul:
                sm0 = hdul[0].data * u.Unit(hdul[0].header['BUNIT'])
            with fits.open(sem0_file) as hdul:
                sem0 = hdul[0].data * u.Unit(hdul[0].header['BUNIT'])
            with fits.open(sew_file) as hdul:
                sew = hdul[0].data * u.Unit(hdul[0].header['BUNIT'])
            with fits.open(seew_file) as hdul:
                seew = hdul[0].data * u.Unit(hdul[0].header['BUNIT'])
            if not (bm0.shape == sm0.shape == sem0.shape ==
                    sew.shape == seew.shape):
                raise ValueError("Input maps have inconsistent shape")
            mask = (sm0 > 0) & (sew > 0)
            orig_sm0 = sm0.copy()
            sm0[~mask] = np.nan
            sem0[~mask] = np.nan
            sew[~mask] = np.nan
            seew[~mask] = np.nan
            res = (np.deg2rad(hdr['BMAJ']) * gal_dist).to('pc')
            radius3d = min(res/2, np.cbrt(res**2*H_los/8))

        # area coverage fraction
        if verbose:
            print("    Add area coverage fraction")
        t.add_pix_area_frac(
            # column to save the output
            colname=f"fracA_CO21_pix_{res_str}",
            # input parameters
            header=hdr, orig_mom0=orig_sm0, masked_mom0=sm0)

        # flux recovery fraction
        if verbose:
            print("    Add flux recovery fraction")
        t.add_pix_flux_frac(
            # column to save the output
            colname=f"fracF_CO21_pix_{res_str}",
            # input parameters
            header=hdr, strict_mom0=sm0, broad_mom0=bm0)

        # completeness corrections
        if verbose:
            print("    Calculate completeness corrections")
        t.calc_completeness_corr(
            # column to save the output
            colname_corr_I=f"corr_I_CO21_pix_{res_str}",
            colname_corr_c=f"corr_c_CO21_pix_{res_str}",
            colname_corr_t=f"corr_t_ff^-1_pix_{res_str}",
            colname_corr_P=f"corr_P_selfg_pix_{res_str}",
            # input parameters
            fracA=t[f"fracA_CO21_pix_{res_str}"],
            fracF=t[f"fracF_CO21_pix_{res_str}"])

        # clumping factor
        if verbose:
            print("    Add clumping factor")
        t.add_clumping(
            # column to save the output
            colname=f"c_CO21_pix_{res_str}",
            colname_e=f"e_c_CO21_pix_{res_str}",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            complete_corr=t[f"corr_c_CO21_pix_{res_str}"])

        # CO line integrated intensity
        if verbose:
            print("    Add CO line integrated intensity")
        t.add_pix_co_intensity(
            # column to save the output
            colname=f"<I_CO21_pix_{res_str}>",
            colname_e=f"e_<I_CO21_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            complete_corr=t[f"corr_I_CO21_pix_{res_str}"])

        # molecular gas surface density
        if verbose:
            print("    Add molecular gas surface density")
        t.add_pix_surf_density(
            # column to save the output
            colname=f"<Sigma_mol_pix_{res_str}>",
            colname_e=f"e_<Sigma_mol_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            alpha_CO=t[colname_alphaCO], cosi=gal_cosi,
            complete_corr=t[f"corr_I_CO21_pix_{res_str}"])

        # CO line width
        if verbose:
            print("    Add CO line width")
        t.add_pix_co_linewidth(
            # column to save the output
            colname=f"<sigmav_CO21_pix_{res_str}>",
            colname_e=f"e_<sigmav_CO21_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            masked_ew=sew, masked_eew=seew,
            complete_corr=None)

        # molecular gas velocity dispersion
        if verbose:
            print("    Add molecular gas velocity dispersion")
        t.add_pix_vel_disp(
            # column to save the output
            colname=f"<vdisp_mol_pix_{res_str}>",
            colname_e=f"e_<vdisp_mol_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            masked_ew=sew, masked_eew=seew,
            cosi=gal_cosi, complete_corr=None)

        # free-fall time
        if verbose:
            print("    Add free-fall time")
        t.add_pix_freefall_time(
            # column to save the output
            colname=f"<t_ff^-1_pix_{res_str}>",
            colname_e=f"e_<t_ff^-1_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            FWHM_beam=res, radius=radius3d, alpha_CO=t[colname_alphaCO],
            complete_corr=t[f"corr_t_ff^-1_pix_{res_str}"])

        # crossing time
        if verbose:
            print("    Add crossing time")
        t.add_pix_crossing_time(
            # column to save the output
            colname=f"<t_cross^-1_pix_{res_str}>",
            colname_e=f"e_<t_cross^-1_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            masked_ew=sew, masked_eew=seew,
            radius=radius3d, cosi=gal_cosi, complete_corr=None)

        # virial parameter
        if verbose:
            print("    Add virial parameter")
        t.add_pix_virial_param(
            # column to save the output
            colname=f"<alpha_vir_pix_{res_str}>",
            colname_e=f"e_<alpha_vir_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            masked_ew=sew, masked_eew=seew,
            FWHM_beam=res, radius=radius3d,
            alpha_CO=t[colname_alphaCO], cosi=gal_cosi,
            complete_corr=None)

        # internal turbulent pressure
        if verbose:
            print("    Add internal turbulent pressure")
        t.add_pix_turb_pressure(
            # column to save the output
            colname=f"<P_turb_pix_{res_str}>",
            colname_e=f"e_<P_turb_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            masked_ew=sew, masked_eew=seew,
            FWHM_beam=res, radius=radius3d,
            alpha_CO=t[colname_alphaCO], cosi=gal_cosi,
            complete_corr=t[f"corr_I_CO21_pix_{res_str}"])

        # dynamical equilibrium pressures
        if verbose:
            print("    Add dynamical equilibrium pressure")
        t.add_pix_dyn_eq_pressure(
            # column to save the output
            colname=f"<P_DE_pix_{res_str}>",
            colname_e=f"e_<P_DE_pix_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, masked_emom0=sem0,
            Sigma_mol_cloud=t[f"<Sigma_mol_pix_{res_str}>"],
            e_Sigma_mol_cloud=t[f"e_<Sigma_mol_pix_{res_str}>"],
            Sigma_mol_kpc=t['Sigma_mol'], e_Sigma_mol_kpc=t['e_Sigma_mol'],
            Sigma_atom_kpc=t['Sigma_atom'], e_Sigma_atom_kpc=t['e_Sigma_atom'],
            rho_star_mp=t['rho_star_mp'], e_rho_star_mp=t['e_rho_star_mp'],
            FWHM_beam=res, radius=radius3d,
            alpha_CO=t[colname_alphaCO], cosi=gal_cosi,
            complete_corr=t[f"corr_P_selfg_pix_{res_str}"])

        # Gong+20 ICO-based conversion factor
        if verbose:
            print("    Add Gong+20 ICO-based conversion factor")
        t.add_pix_co_conversion_g20(
            # column to save the output
            colname=f"<alpha_CO21_G20ICO_{res_str}>",
            # input parameters
            header=hdr, masked_mom0=sm0, Zprime=t['Zprime_scaling'],
            FWHM_beam=res, cosi=gal_cosi, force_res_dependence=True)


def add_object_stats_to_table(
        t: PhangsAlmaMegaTable, data_paths=None, res_pcs=None,
        colname_alphaCO='alpha_CO21', H_los=None, verbose=True):

    gal_name = t.meta['GALAXY']
    gal_dist = t.meta['DIST_MPC'] * u.Mpc
    gal_cosi = np.cos(np.deg2rad(t.meta['INCL_DEG']))

    if colname_alphaCO not in t.colnames:
        raise ValueError("No alphaCO column in table")

    if H_los is None:
        H_los = 100 * u.pc / gal_cosi  # Heyer & Dame (2015)

    for res_pc in res_pcs:

        if res_pc is None:
            res_str = 'native'
            postfix_res = ''
        else:
            res_str = f"{res_pc}pc"
            postfix_res = f"_{res_pc}pc"
        if verbose:
            print(f"  Add object-based GMC statistics @ {res_str} resolution")

        # read CPROPS catalog file
        cprops_file = Path(data_paths['PHANGS_ALMA_CPROPS'].format(
            galaxy=gal_name, postfix_resolution=postfix_res,
            postfix_sensitivity=''))
        bm0_file = Path(data_paths['PHANGS_ALMA_CO21'].format(
            galaxy=gal_name, product='mom0',
            postfix_masking='_broad', postfix_resolution=postfix_res))
        if not (cprops_file.is_file() and bm0_file.is_file()):
            if verbose:
                print("    Some input file not found -- will do a dry run")
            ra = dec = vdisp = flux = flux_noex = radius2d = radius3d = None
            hdr = bm0 = None
        else:
            if verbose:
                print("    All input file found")
            cpropscat = Table.read(cprops_file)
            ra = np.array(cpropscat['XCTR_DEG'])
            dec = np.array(cpropscat['YCTR_DEG'])
            vdisp = np.array(cpropscat['SIGV_KMS']) * u.Unit('km s-1')
            flux = (
                np.array(cpropscat['FLUX_KKMS_PC2']) /
                np.array(cpropscat['DISTANCE_PC'])**2 *
                u.Unit('K km s-1') * gal_dist**2).to('K km s-1 pc2')
            flux_noex = (
                np.array(cpropscat['FLUX_NOEX']) /
                np.array(cpropscat['DISTANCE_PC'])**2 *
                u.Unit('K km s-1') * gal_dist**2).to('K km s-1 pc2')
            radius2d = (
                np.array(cpropscat['RAD_PC']) /
                np.array(cpropscat['DISTANCE_PC']) *
                gal_dist).to('pc')
            radius3d = radius2d.copy()
            radius3d[radius2d > H_los/2] = \
                np.cbrt(radius2d**2 * H_los/2)[radius2d > H_los/2]
            mask = \
                np.isfinite(vdisp) & np.isfinite(radius2d) & np.isfinite(flux)
            vdisp[~mask] = np.nan
            flux[~mask] = np.nan
            flux_noex[~mask] = np.nan
            radius2d[~mask] = np.nan
            radius3d[~mask] = np.nan
            with fits.open(bm0_file) as hdul:
                hdr = hdul[0].header.copy()
                hdr.remove('BUNIT')
                bm0 = hdul[0].data * u.Unit(hdul[0].header['BUNIT'])

        # object number count
        if verbose:
            print("    Add object number count")
        t.add_obj_count(
            # column to save the output
            colname=f"N_obj_{res_str}",
            # input parameters
            ra=ra, dec=dec, flux=flux)

        # flux recovery fraction
        if verbose:
            print("    Add flux recovery fraction")
        t.add_obj_flux_frac(
            # column to save the output
            colname=f"fracF_CO21_obj_{res_str}",
            # input parameters
            ra=ra, dec=dec, flux=flux_noex,
            header=hdr, broad_mom0=bm0, dist=gal_dist)

        # CO line flux
        if verbose:
            print("    Add CO line flux")
        t.add_obj_co_flux(
            # column to save the output
            colname=f"<F_CO21_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux,
            complete_corr=t[f"corr_I_CO21_pix_{res_str}"])

        # molecular gas mass
        if verbose:
            print("    Add molecular gas mass")
        t.add_obj_mass(
            # column to save the output
            colname=f"<M_mol_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, alpha_CO=t[colname_alphaCO],
            complete_corr=t[f"corr_I_CO21_pix_{res_str}"])

        # CO line width
        if verbose:
            print("    Add CO line width")
        t.add_obj_co_linewidth(
            # column to save the output
            colname=f"<sigmav_CO21_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, vdisp=vdisp,
            complete_corr=None)

        # molecular gas velocity dispersion
        if verbose:
            print("    Add molecular gas velocity dispersion")
        t.add_obj_vel_disp(
            # column to save the output
            colname=f"<vdisp_mol_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, vdisp=vdisp,
            cosi=gal_cosi, complete_corr=None)

        # radius
        if verbose:
            print("    Add radius")
        t.add_obj_radius(
            # column to save the output
            colname=f"<R_2d_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, radius=radius2d,
            complete_corr=None)
        t.add_obj_radius(
            # column to save the output
            colname=f"<R_3d_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, radius=radius3d,
            complete_corr=None)

        # projected area
        if verbose:
            print("    Add deprojected area")
        t.add_obj_area(
            # column to save the output
            colname=f"<Area_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, radius=radius2d,
            cosi=gal_cosi, complete_corr=None)

        # molecular gas surface density
        if verbose:
            print("    Add molecular gas surface density")
        t.add_obj_surf_density(
            # column to save the output
            colname=f"<Sigma_mol_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, radius=radius2d,
            alpha_CO=t[colname_alphaCO], cosi=gal_cosi,
            complete_corr=t[f"corr_I_CO21_pix_{res_str}"])

        # free-fall time
        if verbose:
            print("    Add free-fall time")
        t.add_obj_freefall_time(
            # column to save the output
            colname=f"<t_ff^-1_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, radius=radius3d,
            alpha_CO=t[colname_alphaCO],
            complete_corr=t[f"corr_t_ff^-1_pix_{res_str}"])

        # crossing time
        if verbose:
            print("    Add crossing time")
        t.add_obj_crossing_time(
            # column to save the output
            colname=f"<t_cross^-1_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, vdisp=vdisp, radius=radius3d,
            cosi=gal_cosi, complete_corr=None)

        # virial parameter
        if verbose:
            print("    Add virial parameter")
        t.add_obj_virial_param(
            # column to save the output
            colname=f"<alpha_vir_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, vdisp=vdisp, radius=radius3d,
            alpha_CO=t[colname_alphaCO], cosi=gal_cosi,
            complete_corr=None)

        # internal turbulent pressure
        if verbose:
            print("    Add internal turbulent pressure")
        t.add_obj_turb_pressure(
            # column to save the output
            colname=f"<P_turb_obj_{res_str}>",
            # input parameters
            ra=ra, dec=dec, flux=flux, vdisp=vdisp, radius=radius3d,
            alpha_CO=t[colname_alphaCO], cosi=gal_cosi,
            complete_corr=t[f"corr_I_CO21_pix_{res_str}"])


def calc_high_level_params_in_table(
        t: PhangsAlmaMegaTable, res_pcs=None, verbose=True):

    if verbose:
        print("  Calculate high-level parameters")

    # CO-to-H2 conversion factors
    if verbose:
        print("    Calculate CO-to-H2 conversion factors")
    res_str = f"{res_pcs[-1]}pc"
    t.calc_co_conversion(
        # column to save the output
        colname='alpha_CO21_N12',
        # input parameters
        method='N12', Zprime=t['Zprime_scaling'],
        I_CO_cloud=t[f"<I_CO21_pix_{res_str}>"])
    t.calc_co_conversion(
        # column to save the output
        colname='alpha_CO21_B13',
        # input parameters
        method='B13', Zprime=t['Zprime_scaling'], I_CO_kpc=t['I_CO21'],
        Sigma_else_kpc=t['Sigma_star']+t['Sigma_atom'])
    t.calc_co_conversion(
        # column to save the output
        colname='alpha_CO21_T24',
        # input parameters
        method='T24', vdisp_mol_150pc=t["<vdisp_mol_pix_150pc>"])
    t.calc_co_conversion(
        # column to save the output
        colname='alpha_CO21_T24alt',
        # input parameters
        method='T24alt', vdisp_mol_150pc=t["<vdisp_mol_pix_150pc>"],
        Zprime=t['Zprime_scaling'])


def build_alma_table_from_base_table(
        t, data_paths=None, output_format=None, resolutions=None,
        writefile=None, verbose=True):

    if resolutions is None:
        resolutions = (None, 60, 90, 120, 150)

    # ------------------------------------------------
    # add pixel-based GMC statistics
    # ------------------------------------------------

    add_pixel_stats_to_table(
        t, data_paths=data_paths, res_pcs=resolutions, verbose=verbose)

    # ------------------------------------------------
    # add object-based GMC statistics
    # ------------------------------------------------

    add_object_stats_to_table(
        t, data_paths=data_paths, res_pcs=resolutions, verbose=verbose)

    # ------------------------------------------------
    # calculate high-level parameters
    # ------------------------------------------------

    calc_high_level_params_in_table(
        t, res_pcs=resolutions, verbose=verbose)

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
    t_format = Table.read(config_dir / "format_phangsalma.csv")

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
        tessell_alma_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content='phangsalma',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        if (tessell_base_table_file.is_file() and not
                tessell_alma_table_file.is_file()):
            print("Enhancing tessellation statistics table...")
            t = PhangsAlmaTessellMegaTable.read(tessell_base_table_file)
            build_alma_table_from_base_table(
                t, data_paths=data_paths, output_format=t_format,
                writefile=tessell_alma_table_file)
            print("Done\n")

        # RadialMegaTable
        annulus_width_str = (
            str(table_configs['radial_annulus_width']).replace('.', 'p') +
            table_configs['radial_annulus_width_unit'])
        radial_base_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content='base',
                annulus_width_str=annulus_width_str))
        radial_alma_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content='phangsalma',
                annulus_width_str=annulus_width_str))
        if (radial_base_table_file.is_file() and not
                radial_alma_table_file.is_file()):
            print("Enhancing radial statistics table...")
            t = PhangsAlmaRadialMegaTable.read(radial_base_table_file)
            build_alma_table_from_base_table(
                t, data_paths=data_paths, output_format=t_format,
                writefile=radial_alma_table_file)
            print("Done\n")

    # logging settings
    if logging:
        sys.stdout = orig_stdout
        log.close()
