import re
import os
from pathlib import Path
import numpy as np
from scipy import interpolate
from scipy.special import erf, erfinv
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.table import Table
from reproject import reproject_interp
from mega_table.core import StatsTable
from mega_table.table import (
    RadialMegaTable, ApertureMegaTable, TessellMegaTable)
from mega_table.utils import nanaverage, deproject

from CO_conversion_factor import metallicity, alphaCO

# --------------------------------------------------------------------


def get_data_path(datatype, galname=None, lin_res=None, ext='fits'):
    """
    Get the path to any required data on disk.
    """
    datatypes = datatype.split(':')
    fname_seq = []

    if lin_res is None or lin_res == 0:
        res = None
    else:
        res = lin_res

    # PHANGS data parent directory
    PHANGSdir = Path(os.getenv('PHANGSWORKDIR'))

    if datatypes[0] == 'sample':
        # sample table
        return Path('.') / 'sample.ecsv'

    elif datatypes[0] == 'ALMA':
        # PHANGS-ALMA data
        basedir = PHANGSdir / 'ALMA'
        if datatypes[1] == 'CO':
            # PHANGS-ALMA CO map
            basedir /= 'v4-temp-processed'
            fname_seq = [galname, 'CO21'] + datatypes[2:]
            if res is None:
                fname_seq += ['native']
            else:
                fname_seq += [f"{res.to('pc').value:.0f}pc"]
            else:
                fname_seq += ['native']
        elif datatypes[1] == 'CPROPS':
            # PHANGS-ALMA CPROPS catalog
            basedir = basedir / 'v4-CPROPS' / 'native'
            fname_seq = [
                galname.lower(),
                '12m+7m+tp', 'co21', 'native', 'props']
            if res is not None:
                basedir = (
                    basedir / '..' /
                    f"{res.to('pc').value:.0f}pc_matched")
                fname_seq[-2] = f"{res.to('pc').value:.0f}pc"

    elif datatypes[0] == 'HI':
        # HI data
        basedir = PHANGSdir / 'HI'
        fname_seq = [galname, '21cm'] + datatypes[1:]
        if res is not None:
            fname_seq += [f"{res.to('pc').value:.0f}pc"]

    elif datatypes[0] == 'z0MGS':
        # z0MGS data
        basedir = PHANGSdir / 'z0MGS'
        fname_seq = [galname] + datatypes[1:]
        if res is not None:
            fname_seq += [f"{res.to('pc').value:.0f}pc"]

    elif datatypes[0] == 'IRAC':
        # IRAC data
        basedir = PHANGSdir / 'IRAC'
        if datatypes[1] == 'env_mask':
            # S4G morphological maps
            basedir /= 'environmental_masks'
            fname_seq = [galname, 'mask'] + datatypes[2:]
        else:
            fname_seq = [galname, 'IRAC'] + datatypes[1:]
            if res is not None:
                fname_seq += [f"{res.to('pc').value:.0f}pc"]

    elif datatypes[0] == 'Halpha':
        # narrow band Halpha data
        basedir = PHANGSdir / 'Halpha'
        fname_seq = [galname] + datatypes[1:]
        if res is not None:
            fname_seq += [f"{res.to('pc').value:.0f}pc"]

    elif datatypes[0] == 'rotcurve':
        # rotation curve
        basedir = PHANGSdir / 'rotcurve'
        fname_seq = [galname] + datatypes[1:]

    else:
        raise ValueError("Unrecognized dataset")

    return basedir / ('_'.join(fname_seq) + '.' + ext)


# --------------------------------------------------------------------


class PhangsMegaTable(StatsTable):

    """
    PHANGS-specific mega-table construction toolkit
    """

    # ----------------------------------------------------------------

    def add_rotcurve(
            self, modelfile=None, modelcol=None, r_gal_angle=None,
            colname=None, unit=None):
        t_model = Table.read(modelfile)
        model = interpolate.interp1d(
            t_model['r_gal'].quantity.to('arcsec').value,
            t_model[modelcol].quantity.value,
            bounds_error=False, fill_value=np.nan)
        self[colname] = (
            model(r_gal_angle.to('arcsec').value) *
            t_model[modelcol].quantity.unit).to(unit)

    # ----------------------------------------------------------------

    def add_metallicity(
            self, Mstar=None,
            r_gal=None, Rdisk=None, logOH_solar=None,
            colname=None, unit=None):
        logOH_Re = metallicity.predict_logOH_SAMI19(
            Mstar * 10**0.10)  # Fig. A1 in Sanchez+19
        logOH = metallicity.extrapolate_logOH_radially(
            logOH_Re, gradient='CALIFA14',
            Rgal=r_gal, Re=Rdisk*1.68)  # Eq. A.3 in Sanchez+14
        self[colname] = (
            10**(logOH - logOH_solar) * u.Unit('')).to(unit)

    # ----------------------------------------------------------------

    def add_co_conversion(
            self, method=None, Zprime=None,
            COsm0file=None, CObm0file=None, Sigmaelse=None,
            colname=None, unit=None):
        if method == 'Galactic':
            self[colname] = alphaCO.alphaCO10_Galactic.to(unit)
        elif method == 'PHANGS':
            self[colname] = alphaCO.predict_alphaCO10_PHANGS(
                Zprime=Zprime).to(unit)
        elif method == 'N12':
            with fits.open(COsm0file) as hdul:
                COsm0map = hdul[0].data
                COhdr = hdul[0].header
                COm0unit = u.Unit(COhdr['BUNIT'])
                self.calc_image_stats(
                    COsm0map, header=COhdr, weight=COsm0map,
                    stat_func=nanaverage,
                    colname='_WCO_GMC', unit=COm0unit)
            self[colname] = alphaCO.predict_alphaCO10_N12(
                Zprime=Zprime, WCO10GMC=self['_WCO_GMC']).to(unit)
            self.table.remove_column('_WCO_GMC')
        elif method == 'B13':
            with fits.open(COsm0file) as hdul:
                COsm0map = hdul[0].data
                COhdr = hdul[0].header
                COm0unit = u.Unit(COhdr['BUNIT'])
                self.calc_image_stats(
                    COsm0map, header=COhdr, weight=COsm0map,
                    stat_func=nanaverage,
                    colname='_WCO_GMC', unit=COm0unit)
            self.calc_image_stats(
                CObm0file, stat_func=nanaverage,
                colname='_WCO_kpc', unit=COm0unit)
            self[colname] = alphaCO.predict_alphaCO10_B13(
                Zprime=Zprime, WCO10GMC=self['_WCO_GMC'],
                WCO10kpc=self['_WCO_kpc'], Sigmaelsekpc=Sigmaelse,
                suppress_error=True).to(unit)
            self.table.remove_columns(['_WCO_GMC', '_WCO_kpc'])
        else:
            raise ValueError(f"Unrecognized `method`: {method}")

    # ----------------------------------------------------------------

    def add_surf_dens_mol(
            self, COm0file=None, alpha_CO=None, cosi=1.,
            colname=None, unit=None):
        self.calc_image_stats(
            COm0file, stat_func=nanaverage,
            colname='_I_CO', unit='header')
        self[colname] = (self['_I_CO'] * alpha_CO * cosi).to(unit)
        self.table.remove_column('_I_CO')

    # ----------------------------------------------------------------

    def add_surf_dens_atom(
            self, HIm0file=None, alpha_21cm=None, cosi=1.,
            colname=None, unit=None):
        self.calc_image_stats(
            HIm0file, stat_func=nanaverage,
            colname='_I_21cm', unit='header')
        self[colname] = (self['_I_21cm'] * alpha_21cm * cosi).to(unit)
        self.table.remove_column('_I_21cm')

    # ----------------------------------------------------------------

    def add_surf_dens_sfr(
            self, SFRfile=None, cosi=1.,
            colname=None, unit=None):
        self.calc_image_stats(
            SFRfile, stat_func=nanaverage,
            colname='_SFR', unit='header')
        self[colname] = (self['_SFR'] * cosi).to(unit)
        self.table.remove_column('_SFR')

    # ----------------------------------------------------------------

    def add_surf_dens_star(
            self, IRfile,
            MtoL=None, band='3p4um', Lsun_IR=None, cosi=1.,
            colname=None, unit=None):
        self.calc_image_stats(
            IRfile, stat_func=nanaverage,
            colname='_I_IR', unit='header')
        if band[-2:] == 'um':
            freq = (
                float(band[:-2].replace('p', '.')) * u.um
            ).to('Hz', equivalencies=u.spectral())
        else:
            raise ValueError(
                f"Unrecognized wave band name: {band}")
        alpha_IR = (
            MtoL * u.Lsun / Lsun_IR * 4*np.pi*u.sr * freq
        ).to('Msun pc-2 MJy-1 sr')
        self[colname] = (self['_I_IR'] * cosi * alpha_IR).to(unit)
        self.table.remove_column('_I_IR')

    # ----------------------------------------------------------------

    def add_vol_dens_star(
            self, Sigma_star=None, r_gal=None,
            Rstar=None, diskshape='flat',
            colname=None, unit=None):
        flat_ratio = 7.3  # Kregel+02, Sun+20a
        if diskshape == 'flat':
            h_star = Rstar / flat_ratio
        elif diskshape == 'flared':
            h_star = Rstar / flat_ratio * np.exp(
                (r_gal / Rstar).to('').value - 1)
        else:
            raise ValueError(f"Unrecognized `diskshape`: {diskshape}")
        self[colname] = (Sigma_star / 4 / h_star).to(unit)

    # ----------------------------------------------------------------

    def add_dyn_eq_pressure(
            self, scale='kpc', rho_star_mp=None,
            Sigma_atom=None, vdisp_atom_z=None,
            Sigma_mol=None, vdisp_mol_z=None,
            Sigma_cloud=None, P_selfg_cloud=None, R_cloud=None,
            colname=None, unit=None):
        if scale == 'kpc':
            if vdisp_atom_z is None:  # Leroy+08, Sun+20a
                vdisp_atom_z = 10 * u.Unit('km s-1')
            if vdisp_mol_z is None:  # Leroy+08
                vdisp_mol_z = vdisp_atom_z
            Sigma_gas = Sigma_atom + Sigma_mol
            vdisp_gas_z = (  # Sun+20a Eq.14
                Sigma_mol * vdisp_mol_z +
                Sigma_atom * vdisp_atom_z) / Sigma_gas
            self[colname] = (  # Sun+20a Eq.12
                (np.pi / 2 * const.G * Sigma_gas**2 +
                 Sigma_gas * vdisp_gas_z *
                 np.sqrt(2 * const.G * rho_star_mp)) /
                const.k_B).to(unit)
        elif scale == 'cloud':
            if vdisp_atom_z is None:  # Leroy+08, Sun+20a
                vdisp_atom_z = 10 * u.Unit('km s-1')
            P_DE_cloud = (  # Sun+20a Eq.16
                P_selfg_cloud +
                np.pi / 2 * const.G / const.k_B *
                Sigma_cloud * Sigma_mol +
                3 / 4 * np.pi * const.G / const.k_B *
                Sigma_cloud * rho_star_mp * R_cloud*2)
            P_DE_atom = (  # Sun+20a Eq.18
                np.pi / 2 * const.G * Sigma_atom**2 +
                np.pi * const.G * Sigma_atom * Sigma_mol +
                Sigma_atom * vdisp_atom_z *
                np.sqrt(2 * const.G * rho_star_mp)) / const.k_B
            self[colname] = (P_DE_cloud + P_DE_atom).to(unit)
        else:
            raise ValueError(f"Unrecognized `scale`: {scale}")

    # ----------------------------------------------------------------

    def add_pix_stats(
            self, header=None,
            tpkmap=None, bm0map=None, sm0map=None, sewmap=None,
            alpha_CO=None, FWHM_beam=None, H_los=None,
            colname=None, unit=None):

        if not (tpkmap.shape == bm0map.shape ==
                sm0map.shape == sewmap.shape):
            raise ValueError(
                "Input maps have inconsistent shape")

        tpeak = tpkmap * u.K
        tpeak[~np.isfinite(tpeak)] = np.nan
        bmom0 = bm0map * u.Unit('K km s-1')
        bmom0[~np.isfinite(bmom0)] = np.nan
        wt = sm0map.copy().astype('float')
        wt[~np.isfinite(sewmap) | (sewmap <= 0) |
           ~np.isfinite(sm0map) | (sm0map <= 0)] = 0
        vdisp = sewmap * u.Unit('km s-1')
        vdisp[wt == 0] = np.nan
        smom0 = sm0map * u.Unit('K km s-1')
        smom0[wt == 0] = np.nan
        alpha_CO_fid = alphaCO.alphaCO10_Galactic
        surfdens_fid = (smom0 * alpha_CO_fid).to('Msun pc-2')
        mass_fid = (
            np.pi / (4 * np.log(2)) *
            surfdens_fid * FWHM_beam**2).to('Msun')
        radius = FWHM_beam / 2
        if radius > H_los/2:
            radius3d = np.cbrt(radius**2 * H_los/2)
        else:
            radius3d = radius

        # area coverage fraction of the strict CO map
        if colname[:5] == 'fracA':
            self.calc_image_stats(
                np.isfinite(smom0).astype('float'),
                header=header, stat_func=np.nansum,
                colname='_A_strict', unit='')
            self.calc_image_stats(
                np.isfinite(sm0map).astype('float'),
                header=header, stat_func=np.nansum,
                colname='_A_total', unit='')
            self[colname] = (
                self['_A_strict'] / self['_A_total']).to(unit)
            self.table.remove_columns(['_A_strict', '_A_total'])

        # flux recovery fraction of the strict CO map
        elif colname[:5] == 'fracF':
            self.calc_image_stats(
                smom0.value,
                header=header, stat_func=np.nansum,
                colname='_F_strict', unit='')
            self.calc_image_stats(
                bmom0.value,
                header=header, stat_func=np.nansum,
                colname='_F_broad', unit='')
            self[colname] = (
                self['_F_strict'] / self['_F_broad']).to(unit)
            self.table.remove_columns(['_F_strict', '_F_broad'])

        # completeness correction factors
        elif colname[:5] == 'fcorr':
            colname_fracA = (
                'fracA' + re.findall(r'_CO21_\d+pc$', colname)[0])
            colname_fracF = colname_fracA.replace('fracA', 'fracF')
            if not (colname_fracA in self.colnames and
                    colname_fracF in self.colnames):
                self[colname] = np.nan
            else:
                fA = self[colname_fracA].value.copy()
                fF = self[colname_fracF].value.copy()
                fF[fF < 0] = 0
                fF[fF > 1] = 1
                fF[fF < fA] = fA[fF < fA]
                fICOsq = 1/2 * (
                    1 - erf(2 * erfinv(1-2*fF) - erfinv(1-2*fA)))
                quantity = re.match(
                    r'^fcorr_(.+)_CO21_\d+pc$', colname).group(1)
                if quantity == 'I':
                    self[colname] = fF / fICOsq
                elif quantity == 'clumping':
                    self[colname] = fF**2 / fICOsq
                else:
                    raise ValueError(
                        f"Unrecognized column name: {colname}")

        # clumping factor of the strict CO map
        elif colname[:8] == 'clumping':
            self.calc_image_stats(
                wt, weight=wt,
                header=header, stat_func=nanaverage,
                colname='_F<ICO>', unit='')
            self.calc_image_stats(
                wt, weight=None,
                header=header, stat_func=nanaverage,
                colname='_A<ICO>', unit='')
            self[colname] = (
                self['_F<ICO>'] / self['_A<ICO>']).to(unit)
            self.table.remove_columns(['_F<ICO>', '_A<ICO>'])

        # weighted average cloud-scale gas properties
        else:
            quantity = colname[2:-1].split('_pix_')[0]
            if colname[0] == 'F':
                weight = wt
            else:
                weight = None

            # CO line peak temperature
            if quantity == 'T_peak':
                self.calc_image_stats(
                    tpeak.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # CO line integrated intensity
            elif quantity == 'I_CO21':
                self.calc_image_stats(
                    smom0.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # surface density
            elif quantity == 'Sigma_mol':
                self.calc_image_stats(
                    surfdens_fid.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')

            # velocity dispersion
            elif quantity == 'vdisp_mol':
                self.calc_image_stats(
                    vdisp.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # turbulent pressure
            elif quantity == 'P_turb_sph':
                pturb = (
                    3 / (4 * np.pi) / const.k_B *
                    mass_fid * vdisp**2 / radius**3)
                self.calc_image_stats(
                    pturb.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')
            elif quantity == 'P_turb_r3d':
                pturb = (
                    3 / (4 * np.pi) / const.k_B *
                    mass_fid * vdisp**2 / radius3d**3)
                self.calc_image_stats(
                    pturb.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')

            # cloud weight due to self-gravity
            elif quantity == 'P_selfg_sph':
                pself = (
                    3/8 / np.pi * const.G / const.k_B *
                    mass_fid**2 / radius**4)
                self.calc_image_stats(
                    pself.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**2
            elif quantity == 'P_selfg_r3d':
                pself = (
                    3/8 / np.pi * const.G / const.k_B *
                    mass_fid**2 / radius3d**4)
                self.calc_image_stats(
                    pself.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**2

            # virial parameter
            elif quantity == 'alpha_vir_sph':
                avir = (
                    5 * vdisp**2 * radius / (const.G * mass_fid))
                self.calc_image_stats(
                    avir.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**-1
            elif quantity == 'alpha_vir_r3d':
                avir = (
                    5 * vdisp**2 * radius3d / (const.G * mass_fid))
                self.calc_image_stats(
                    avir.to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**-1

            # free-fall time
            elif quantity == 't_ff_sph^-1':
                tff = np.sqrt(
                    np.pi**2 * radius**3 / (8 * const.G * mass_fid))
                self.calc_image_stats(
                    (tff**-1).to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**0.5
            elif quantity == 't_ff_r3d^-1':
                tff = np.sqrt(
                    np.pi**2 * radius3d**3 / (8 * const.G * mass_fid))
                self.calc_image_stats(
                    (tff**-1).to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**0.5

            # crossing time
            elif quantity == 't_cross_sph^-1':
                tcross = radius / vdisp / np.sqrt(3)
                self.calc_image_stats(
                    (tcross**-1).to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
            elif quantity == 't_cross_r3d^-1':
                tcross = radius3d / vdisp / np.sqrt(3)
                self.calc_image_stats(
                    (tcross**-1).to(unit).value, header=header,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            else:
                raise ValueError(
                    f"Unrecognized column name: {colname}")

    # ----------------------------------------------------------------

    def add_cprops_stats(
            self, cpropscat=None,
            alpha_CO=None, H_los=None, gal_dist=None,
            colname=None, unit=None):

        ra = np.array(cpropscat['XCTR_DEG'])
        dec = np.array(cpropscat['YCTR_DEG'])
        radius = (
            np.array(cpropscat['RAD_PC']) /
            np.array(cpropscat['DISTANCE_PC']) *
            gal_dist).to('pc')
        flux = (
            np.array(cpropscat['FLUX_KKMS_PC2']) /
            np.array(cpropscat['DISTANCE_PC'])**2 *
            u.Unit('K km s-1') * gal_dist**2).to('K km s-1 pc2')
        alpha_CO_fid = alphaCO.alphaCO10_Galactic
        mass_fid = (flux * alpha_CO_fid).to('Msun')
        vdisp = np.array(cpropscat['SIGV_KMS']) * u.Unit('km s-1')
        wt = flux.value.copy()
        wt[~np.isfinite(radius)] = 0
        flux[~np.isfinite(radius)] = np.nan
        mass_fid[~np.isfinite(radius)] = np.nan
        vdisp[~np.isfinite(radius)] = np.nan
        radius[~np.isfinite(radius)] = np.nan
        radius3d = radius.copy()
        radius3d[radius > H_los/2] = np.cbrt(
            radius**2 * H_los/2)[radius > H_los/2]

        # number of clouds in the CPROPS catalog
        if colname[:5] == 'N_obj':
            self.calc_catalog_stats(
                np.isfinite(flux).astype('int'), ra, dec,
                stat_func=np.nansum, colname=colname, unit=unit)

        # weighted average cloud properties
        else:
            quantity = colname[2:-1].split('_CPROPS_')[0]
            if colname[0] == 'F':
                weight = wt
            else:
                weight = None

            # radius
            if quantity == 'R_2d':
                self.calc_catalog_stats(
                    radius.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
            elif quantity == 'R_3d':
                self.calc_catalog_stats(
                    radius3d.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # area
            elif quantity == 'Area_2d':
                area = np.pi / (2 * np.log(2)) * radius**2
                self.calc_catalog_stats(
                    area.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # mass
            elif quantity == 'M_mol':
                self.calc_catalog_stats(
                    mass_fid.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')

            # velocity dispersion
            elif quantity == 'vdisp_mol':
                self.calc_catalog_stats(
                    vdisp.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # surface density
            elif quantity == 'Sigma_mol':
                surfdens_fid = (  # Rosolowsky+21, text following Eq.12
                    mass_fid / radius**2 / (2*np.pi))
                self.calc_catalog_stats(
                    surfdens_fid.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')

            # turbulent pressure
            elif quantity == 'P_turb_sph':
                pturb = (  # Rosolowsky+21, Eq.16
                    3 / (8*np.pi) / const.k_B *
                    mass_fid * vdisp**2 / radius**3)
                self.calc_catalog_stats(
                    pturb.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')
            elif quantity == 'P_turb_r3d':
                pturb = (  # Rosolowsky+21, Eq.16
                    3 / (8*np.pi) / const.k_B *
                    mass_fid * vdisp**2 / radius3d**3)
                self.calc_catalog_stats(
                    pturb.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')

            # virial parameter
            elif quantity == 'alpha_vir_sph':
                avir = (  # Rosolowsky+21, text following Eq.10
                    10 * vdisp**2 * radius / (const.G * mass_fid))
                self.calc_catalog_stats(
                    avir.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**-1
            elif quantity == 'alpha_vir_r3d':
                avir = (  # Rosolowsky+21, text following Eq.10
                    10 * vdisp**2 * radius3d / (const.G * mass_fid))
                self.calc_catalog_stats(
                    avir.to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**-1

            # free-fall time
            elif quantity == 't_ff_sph^-1':
                tff = np.sqrt(  # Rosolowsky+21, Eq.17
                    np.pi**2 * radius**3 / (4 * const.G * mass_fid))
                self.calc_catalog_stats(
                    (tff**-1).to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**0.5
            elif quantity == 't_ff_r3d^-1':
                tff = np.sqrt(  # Rosolowsky+21, Eq.17
                    np.pi**2 * radius3d**3 / (4 * const.G * mass_fid))
                self.calc_catalog_stats(
                    (tff**-1).to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
                self[colname] *= (alpha_CO / alpha_CO_fid).to('')**0.5

            # crossing time
            elif quantity == 't_cross_sph^-1':
                tcross = radius / vdisp / np.sqrt(3)
                self.calc_catalog_stats(
                    (tcross**-1).to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)
            elif quantity == 't_cross_r3d^-1':
                tcross = radius3d / vdisp / np.sqrt(3)
                self.calc_catalog_stats(
                    (tcross**-1).to(unit).value, ra, dec,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            else:
                raise ValueError(
                    f"Unrecognized column name: {colname}")

    # ----------------------------------------------------------------

    def add_env_frac(
            self, envfile=None, wfile=None,
            colname=None, unit=None):
        if wfile is None:
            with fits.open(envfile) as hdul:
                envmap = hdul[0].data.copy()
                hdr = hdul[0].header.copy()
                wmap = None
        else:
            with fits.open(wfile) as hdul:
                wmap = hdul[0].data.copy()
                wmap[~np.isfinite(wmap) | (wmap < 0)] = 0
                hdr = hdul[0].header.copy()
            with fits.open(envfile) as hdul:
                envmap, footprint = reproject_interp(
                    hdul[0], hdr, order=0)
                envmap[~footprint.astype('?')] = 0
        self.calc_image_stats(
            (envmap > 0).astype('float'), header=hdr,
            stat_func=nanaverage, weight=wmap,
            colname=colname, unit=unit)


# --------------------------------------------------------------------


class PhangsRadialMegaTable(PhangsMegaTable, RadialMegaTable):

    """
    RadialMegaTable enhanced with PHANGS-specific tools.
    """

    def add_coord(
            self, colname=None, unit=None, gal_dist=None,
            r_gal_angl_min=None, r_gal_angl_max=None):
        if r_gal_angl_min is not None:
            self[colname] = (
                r_gal_angl_min.to('rad').value * gal_dist).to(unit)
        elif r_gal_angl_max is not None:
            self[colname] = (
                r_gal_angl_max.to('rad').value * gal_dist).to(unit)


# --------------------------------------------------------------------


class PhangsApertureMegaTable(PhangsMegaTable, ApertureMegaTable):

    """
    ApertureMegaTable enhanced with PHANGS-specific tools.
    """

    def add_coord(
            self, colname_rgal=None, unit_rgal=None,
            colname_phigal=None, unit_phigal=None,
            gal_dist=None, **kwargs):
        r_gal, phi_gal = deproject(**kwargs)
        if colname_rgal is not None and unit_rgal is not None:
            self[colname_rgal] = (
                r_gal * u.deg / u.rad * gal_dist).to(unit_rgal)
        if colname_phigal is not None and unit_phigal is not None:
            self[colname_phigal] = (phi_gal * u.deg).to(unit_phigal)


# --------------------------------------------------------------------


class PhangsTessellMegaTable(PhangsMegaTable, TessellMegaTable):

    """
    TessellMegaTable enhanced with PHANGS-specific tools.
    """

    def add_coord(
            self, colname_rgal=None, unit_rgal=None,
            colname_phigal=None, unit_phigal=None,
            gal_dist=None, **kwargs):
        r_gal, phi_gal = deproject(**kwargs)
        if colname_rgal is not None and unit_rgal is not None:
            self[colname_rgal] = (
                r_gal * u.deg / u.rad * gal_dist).to(unit_rgal)
        if colname_phigal is not None and unit_phigal is not None:
            self[colname_phigal] = (phi_gal * u.deg).to(unit_phigal)


# --------------------------------------------------------------------


def add_columns_to_mega_table(
        t, config, gal_params={}, phys_params={}, verbose=True):

    # galaxy parameters
    gal_name = gal_params['name']
    gal_cosi = np.cos(np.deg2rad(gal_params['incl_deg']))
    gal_dist = gal_params['dist_Mpc'] * u.Mpc
    gal_Rstar = (
        (gal_params['Rstar_arcsec'] * u.arcsec).to('rad').value *
        gal_params['dist_Mpc'] * u.Mpc).to('kpc')

    if 'coord' in config['group']:
        if type(t) == PhangsRadialMegaTable:
            # ring inner/outer boundaries
            if verbose:
                print("  Calculating ring inner/outer boundaries")
            for row in config[config['group'] == 'coord']:
                if row['colname'] == 'r_gal_min':
                    t.add_coord(
                        colname=row['colname'], unit=row['unit'],
                        gal_dist=gal_dist,
                        r_gal_angl_min=t['r_gal_angl_min'])
                elif row['colname'] == 'r_gal_max':
                    t.add_coord(
                        colname=row['colname'], unit=row['unit'],
                        gal_dist=gal_dist,
                        r_gal_angl_max=t['r_gal_angl_max'])
                else:
                    raise ValueError(
                        f"Unrecognized column name: {row['colname']}")
        elif type(t) in (
                PhangsApertureMegaTable, PhangsTessellMegaTable):
            # galactic radius and projected angle
            if verbose:
                print(
                    "  Calculating galactic radius "
                    "and projected angle")
            for row in config[config['group'] == 'coord']:
                if row['colname'] in ('RA', 'DEC'):
                    continue
                if row['colname'] == 'r_gal':
                    t.add_coord(
                        colname_rgal=row['colname'],
                        unit_rgal=row['unit'],
                        gal_dist=gal_dist,
                        center_ra=gal_params['ra_deg'],
                        center_dec=gal_params['dec_deg'],
                        incl=gal_params['incl_deg'],
                        pa=gal_params['posang_deg'],
                        ra=t['RA'], dec=t['DEC'])
                elif row['colname'] == 'phi_gal':
                    t.add_coord(
                        colname_phigal=row['colname'],
                        unit_phigal=row['unit'],
                        center_ra=gal_params['ra_deg'],
                        center_dec=gal_params['dec_deg'],
                        incl=gal_params['incl_deg'],
                        pa=gal_params['posang_deg'],
                        ra=t['RA'], dec=t['DEC'])
                else:
                    raise ValueError(
                        f"Unrecognized column name: {row['colname']}")
        else:
            raise ValueError(f"Unknown mega-table type: {type(t)}")

    if 'rotcurve' in config['group']:
        # rotation curve-related quantities
        if verbose:
            print("  Calculating rotation curve-related quantities")
        for row in config[config['group'] == 'rotcurve']:
            modelcol = '_'.join(row['colname'].split('_')[:-1])
            if modelcol in ('V_circ', 'beta'):
                modelfile = get_data_path(
                    row['source'], gal_name, ext='ecsv')
                if not modelfile.is_file():
                    t[row['colname']] = np.nan * u.Unit(row['unit'])
                    continue
                if ('r_gal_min' in t.colnames and
                        'r_gal_max' in t.colnames):
                    r_gal_angle = (
                        (t['r_gal_min'] + t['r_gal_max']) / 2 /
                        gal_dist * u.rad).to('arcsec')
                elif 'r_gal' in t.colnames:
                    r_gal_angle = (
                        t['r_gal'] / gal_dist * u.rad).to('arcsec')
                else:
                    raise ValueError("No coordinate info found")
                t.add_rotcurve(
                    modelfile=modelfile, modelcol=modelcol,
                    r_gal_angle=r_gal_angle,
                    colname=row['colname'], unit=row['unit'])
            else:
                raise ValueError(
                    f"Unrecognized column name: {row['colname']}")

    if 'MtoL' in config['group']:
        # stellar M/L ratio
        if verbose:
            print("  Calculating stellar M/L ratio")
        for row in config[config['group'] == 'MtoL']:
            if row['colname'] == 'MtoL_3p4um':
                t.calc_image_stats(
                    get_data_path(
                        row['source'], gal_name, row['res_pc']*u.pc),
                    colname=row['colname'], unit=u.Unit(row['unit']),
                    suppress_error=True, stat_func=nanaverage)
            else:
                raise ValueError(
                    f"Unrecognized column name: {row['colname']}")

    if 'star' in config['group']:
        # stellar mass surface density
        if verbose:
            print("  Calculating stellar mass surface density")
        t['Sigma_star'] = np.nan * u.Unit('Msun pc-2')
        for row in config[config['group'] == 'star']:
            if row['colname'] == 'Sigma_star':
                continue
            if verbose:
                print(f"    {row['colname']}")
            band = row['colname'][11:16]
            Lsun = (
                phys_params[f"IR_Lsun{band}"] *
                u.Unit(phys_params[f"IR_Lsun{band}_unit"]))
            IRfile = get_data_path(
                row['source'], gal_name, row['res_pc']*u.pc)
            if not IRfile.is_file():
                t[row['colname']] = np.nan * u.Unit(row['unit'])
                continue
            if row['colname'] == 'Sigma_star_3p6umICA':
                MtoL = (
                    phys_params["IR_MtoL_S4GICA"] *
                    u.Unit(phys_params["IR_MtoL_S4GICA_unit"]))
            else:
                if 'MtoL_3p4um' not in t.colnames:
                    raise ValueError(
                        "No stellar M/L ratio info found")
                MtoL = t['MtoL_3p4um'].to('Msun Lsun-1')
            t.add_surf_dens_star(
                IRfile,
                MtoL=MtoL, band=band, Lsun_IR=Lsun, cosi=gal_cosi,
                colname=row['colname'], unit=u.Unit(row['unit']))
            if np.isfinite(t['Sigma_star']).sum() == 0:
                t['Sigma_star'] = t[row['colname']]
                t['Sigma_star'].description = (
                    f"({row['colname'].replace('Sigma_star_', '')})")

    if 'HI' in config['group']:
        # atomic gas mass surface density
        if verbose:
            print("  Calculating HI gas surface density")
        for row in config[config['group'] == 'HI']:
            if row['colname'] == 'I_21cm':
                t.calc_image_stats(
                    get_data_path(
                        row['source'], gal_name, row['res_pc']*u.pc),
                    colname=row['colname'], unit=row['unit'],
                    suppress_error=True, stat_func=nanaverage)
            elif row['colname'] == 'Sigma_atom':
                HIm0file = get_data_path(
                    row['source'], gal_name, row['res_pc']*u.pc)
                if not HIm0file.is_file():
                    t[row['colname']] = np.nan * u.Unit(row['unit'])
                    continue
                alpha_21cm = (
                    phys_params['HI_alpha21cm'] *
                    u.Unit(phys_params['HI_alpha21cm_unit']))
                t.add_surf_dens_atom(
                    HIm0file=HIm0file,
                    alpha_21cm=alpha_21cm, cosi=gal_cosi,
                    colname=row['colname'], unit=row['unit'])
            else:
                raise ValueError(
                    f"Unrecognized column name: {row['colname']}")

    if 'metal' in config['group']:
        # metallicity
        if verbose:
            print("  Calculating metallicity")
        t['Zprime'] = np.nan
        for row in config[config['group'] == 'metal']:
            if row['colname'] == 'Zprime':
                continue
            elif row['colname'] == 'Zprime_MZR+GRD':
                if ('r_gal_min' in t.colnames and
                        'r_gal_max' in t.colnames):
                    t.add_metallicity(
                        Mstar=gal_params['Mstar_Msun'] * u.Msun,
                        r_gal=(t['r_gal_min']+t['r_gal_max'])/2,
                        Rdisk=gal_Rstar,
                        logOH_solar=phys_params['abundance_solar'],
                        colname=row['colname'], unit=row['unit'])
                elif 'r_gal' in t.colnames:
                    t.add_metallicity(
                        Mstar=gal_params['Mstar_Msun'] * u.Msun,
                        r_gal=t['r_gal'], Rdisk=gal_Rstar,
                        logOH_solar=phys_params['abundance_solar'],
                        colname=row['colname'], unit=row['unit'])
                else:
                    raise ValueError("No coordinate info found")
            else:
                raise ValueError(
                    f"Unrecognized column name: {row['colname']}")
            if np.isfinite(t['Zprime']).sum() == 0:
                t['Zprime'] = t[row['colname']]
                t['Zprime'].description = (
                    f"({row['colname'].replace('Zprime_', '')})")

    if 'alphaCO' in config['group']:
        # CO-to-H2 conversion factor
        if verbose:
            print("  Calculating CO-to-H2 conversion factor")
        t['alphaCO10'] = t['alphaCO21'] = (
            np.nan * u.Unit('Msun pc-2 K-1 km-1 s'))
        for row in config[config['group'] == 'alphaCO']:
            if row['colname'] in ('alphaCO10', 'alphaCO21'):
                continue
            elif row['colname'] in (
                    'alphaCO10_Galactic', 'alphaCO10_PHANGS',
                    'alphaCO10_N12', 'alphaCO10_B13'):
                if verbose:
                    print(f"    {row['colname']}")
                method = row['colname'].split('_')[-1]
                if method in ('PHANGS', 'N12', 'B13'):
                    if 'Zprime' not in t.colnames:
                        raise ValueError("No metallicity data found")
                    Zprime = t['Zprime']
                else:
                    Zprime = None
                if method in ('N12', 'B13'):
                    CObm0file = get_data_path(
                        row['source'].split('&')[0], gal_name,
                        row['res_pc']*u.pc)
                    COsm0file = get_data_path(
                        row['source'].split('&')[1], gal_name,
                        row['res_pc']*u.pc)
                    if not COsm0file.is_file():
                        t[row['colname']] = (
                            np.nan * u.Unit(row['unit']))
                        continue
                else:
                    CObm0file = COsm0file = None
                if method == 'B13':
                    if 'Sigma_atom' not in t.colnames:
                        raise ValueError("No HI data found")
                    if 'Sigma_star' not in t.colnames:
                        raise ValueError("No stellar mass data found")
                    Sigmaelse = t['Sigma_star'] + t['Sigma_atom']
                else:
                    Sigmaelse = None
                t.add_co_conversion(
                    method=method, Zprime=Zprime, Sigmaelse=Sigmaelse,
                    COsm0file=COsm0file, CObm0file=CObm0file,
                    colname=row['colname'], unit=row['unit'])
            else:
                raise ValueError(
                    f"Unrecognized column name: {row['colname']}")
            if np.isfinite(t['alphaCO10']).sum() == 0:
                t['alphaCO10'] = t[row['colname']]
                t['alphaCO10'].description = (
                    f"({row['colname'].replace('alphaCO10_', '')})")
                t['alphaCO21'] = (
                    t['alphaCO10'] / phys_params['CO_R21'])
                t['alphaCO21'].description = (
                    t['alphaCO10'].description)

    if 'H2' in config['group']:
        # molecular gas mass surface density
        if verbose:
            print("  Calculating H2 gas surface density")
        for row in config[config['group'] == 'H2']:
            if row['colname'] == 'I_CO21':
                t.calc_image_stats(
                    get_data_path(
                        row['source'], gal_name, row['res_pc']*u.pc),
                    colname=row['colname'], unit=row['unit'],
                    suppress_error=True, stat_func=nanaverage)
            elif row['colname'] == 'Sigma_mol':
                if 'alphaCO21' not in t.colnames:
                    raise ValueError("No alphaCO column found")
                COm0file = get_data_path(
                    row['source'], gal_name, row['res_pc']*u.pc)
                if not COm0file.is_file():
                    t[row['colname']] = np.nan * u.Unit(row['unit'])
                    continue
                t.add_surf_dens_mol(
                    COm0file=COm0file,
                    alpha_CO=t['alphaCO21'], cosi=gal_cosi,
                    colname=row['colname'], unit=row['unit'])
            else:
                raise ValueError(
                    f"Unrecognized column name: {row['colname']}")

    if 'CPROPS_stats' in config['group']:
        # CPROPS cloud statistics
        if verbose:
            print("  Calculating CPROPS cloud statistics")
        if 'alphaCO21' not in t.colnames:
            raise ValueError("No alphaCO column found")
        H_los = (
            phys_params['CO_full_height'] *
            u.Unit(phys_params['CO_full_height_unit']))
        rows = config[config['group'] == 'CPROPS_stats']
        for res_pc in np.unique(rows['res_pc']):
            res = res_pc * u.pc
            cpropsfile = get_data_path(
                rows['source'][0], gal_name, res)
            if not cpropsfile.is_file():
                cpropscat = None
            else:
                cpropscat = Table.read(cpropsfile)
            for row in rows[rows['res_pc'] == res_pc]:
                if verbose:
                    print(f"    {row['colname']}")
                if cpropscat is None:
                    t[row['colname']] = np.nan * u.Unit(row['unit'])
                else:
                    t.add_cprops_stats(
                        colname=row['colname'], unit=row['unit'],
                        cpropscat=cpropscat, alpha_CO=t['alphaCO21'],
                        H_los=H_los, gal_dist=gal_dist)
            del cpropsfile

    if 'CO_stats' in config['group']:
        # CO pixel statistics
        if verbose:
            print("  Calculating CO pixel statistics")
        if 'alphaCO21' not in t.colnames:
            raise ValueError("No alphaCO column found")
        H_los = (
            phys_params['CO_full_height'] *
            u.Unit(phys_params['CO_full_height_unit']))
        rows = config[config['group'] == 'CO_stats']
        for res_pc in np.unique(rows['res_pc']):
            res = res_pc * u.pc
            tpksrc, bm0src, sm0src, sewsrc = (
                rows['source'][0].split('&'))
            tpkfile = get_data_path(tpksrc, gal_name, res)
            bm0file = get_data_path(bm0src, gal_name, res)
            sm0file = get_data_path(sm0src, gal_name, res)
            sewfile = get_data_path(sewsrc, gal_name, res)
            if not (tpkfile.is_file() and bm0file.is_file() and
                    sm0file.is_file() and sewfile.is_file()):
                hdr = tpkmap = bm0map = sm0map = sewmap = None
            else:
                with fits.open(tpkfile) as hdul:
                    hdr = hdul[0].header.copy()
                    tpkmap = hdul[0].data.copy()
                with fits.open(bm0file) as hdul:
                    bm0map = hdul[0].data.copy()
                with fits.open(sm0file) as hdul:
                    sm0map = hdul[0].data.copy()
                with fits.open(sewfile) as hdul:
                    sewmap = hdul[0].data.copy()
            for row in rows[rows['res_pc'] == res_pc]:
                if verbose:
                    print(f"    {row['colname']}")
                if hdr is None:
                    t[row['colname']] = np.nan * u.Unit(row['unit'])
                else:
                    t.add_pix_stats(
                        colname=row['colname'], unit=row['unit'],
                        header=hdr, tpkmap=tpkmap, bm0map=bm0map,
                        sm0map=sm0map, sewmap=sewmap,
                        alpha_CO=t['alphaCO21'],
                        FWHM_beam=res, H_los=H_los)
            del hdr, tpkmap, bm0map, sm0map, sewmap

    if 'env_frac' in config['group']:
        # CO fractional contribution by environments
        if verbose:
            print("  Calculating environmental fractions")
        for row in config[config['group'] == 'env_frac']:
            if verbose:
                print(f"    {row['colname']}")
            envsrc, wsrc = row['source'].split('&')
            envfile = get_data_path(envsrc, gal_name)
            wfile = get_data_path(wsrc, gal_name, row['res_pc']*u.pc)
            if not envfile.is_file() or not wfile.is_file():
                t[row['colname']] = np.nan * u.Unit(row['unit'])
            else:
                t.add_env_frac(
                    envfile=envfile, wfile=wfile,
                    colname=row['colname'], unit=row['unit'])

    if 'P_DE' in config['group']:
        # dynamical equilibrium pressure
        if verbose:
            print("  Calculating dynamical equilibrium pressure")
        for row in config[config['group'] == 'P_DE']:
            if verbose:
                print(f"    {row['colname']}")
            if row['colname'] == 'rho_star_mp':
                if 'Sigma_star' not in t.colnames:
                    raise ValueError("No Sigma_star column found")
                t.add_vol_dens_star(
                    Sigma_star=t['Sigma_star'],
                    Rstar=gal_Rstar, diskshape='flat',
                    colname=row['colname'], unit=row['unit'])
            elif row['colname'] in ('P_DE_L08', 'P_DE_S20'):
                for colname in (
                        'Sigma_mol', 'Sigma_atom', 'rho_star_mp'):
                    if colname not in t.colnames:
                        raise ValueError(
                            f"No `{colname}` column found")
                if row['colname'] == 'P_DE_L08':
                    t.add_dyn_eq_pressure(
                        scale='kpc', rho_star_mp=t['rho_star_mp'],
                        Sigma_mol=t['Sigma_mol'],
                        Sigma_atom=t['Sigma_atom'],
                        colname=row['colname'], unit=row['unit'])
                else:
                    if 'CO_stats' in config['group']:
                        rows = config[config['group'] == 'CO_stats']
                        res_fid = np.max(rows['res_pc'])
                        vdisp_col = f"F<vdisp_mol_pix_{res_fid}pc>"
                        if vdisp_col not in t.colnames:
                            raise ValueError(
                                f"No `{vdisp_col}` column found")
                        vdisp_mol_z = t[vdisp_col]
                    else:
                        vdisp_mol_z = np.nan * u.Unit('km s-1')
                    t.add_dyn_eq_pressure(
                        scale='kpc', rho_star_mp=t['rho_star_mp'],
                        Sigma_mol=t['Sigma_mol'],
                        Sigma_atom=t['Sigma_atom'],
                        vdisp_mol_z=vdisp_mol_z,
                        colname=row['colname'], unit=row['unit'])
            elif re.fullmatch(
                    r'F<P_DE_sph_pix_\d+pc>', row['colname']):
                vdisp_col = f"F<vdisp_mol_pix_{row['res_pc']}pc>"
                Sigma_col = f"F<Sigma_mol_pix_{row['res_pc']}pc>"
                P_selfg_col = f"F<P_selfg_sph_pix_{row['res_pc']}pc>"
                for colname in (
                        'Sigma_mol', 'Sigma_atom', 'rho_star_mp',
                        vdisp_col, Sigma_col, P_selfg_col):
                    if colname not in t.colnames:
                        raise ValueError(
                            f"No `{colname}` column found")
                R_cloud = row['res_pc'] * u.pc
                t.add_dyn_eq_pressure(
                    scale='cloud', rho_star_mp=t['rho_star_mp'],
                    Sigma_atom=t['Sigma_atom'],
                    Sigma_mol=t['Sigma_mol'],
                    vdisp_mol_z=t[vdisp_col],
                    Sigma_cloud=t[Sigma_col],
                    P_selfg_cloud=t[P_selfg_col],
                    R_cloud=R_cloud,
                    colname=row['colname'], unit=row['unit'])
            else:
                raise ValueError(
                    f"Unrecognized column name: {row['colname']}")

    if 'SFR' in config['group']:
        # SFR surface density
        if verbose:
            print("  Calculating SFR surface density")
        t['Sigma_SFR'] = np.nan * u.Unit('Msun yr-1 kpc-2')
        for row in config[config['group'] == 'SFR']:
            if row['colname'] == 'Sigma_SFR':
                continue
            if verbose:
                print(f"    {row['colname']}")
            SFRfile = get_data_path(
                row['source'], gal_name, row['res_pc']*u.pc)
            if not SFRfile.is_file():
                t[row['colname']] = np.nan * u.Unit(row['unit'])
                continue
            t.add_surf_dens_sfr(
                SFRfile=SFRfile, cosi=gal_cosi,
                colname=row['colname'], unit=u.Unit(row['unit']))
            if np.isfinite(t['Sigma_SFR']).sum() == 0:
                t['Sigma_SFR'] = t[row['colname']]
                t['Sigma_SFR'].description = (
                    f"({row['colname'].replace('Sigma_SFR_', '')})")
