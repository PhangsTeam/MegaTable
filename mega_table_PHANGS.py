import os
from pathlib import Path
import numpy as np
from scipy import interpolate
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.table import Table
from reproject import reproject_interp
from mega_table.core import StatsTable
from mega_table.utils import nanaverage
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
            basedir /= 'v3p4-processed'
            fname_seq = [galname, 'CO21'] + datatypes[2:]
            if res is not None:
                fname_seq += [f"{res.to('pc').value:.0f}pc"]
        elif datatypes[1] == 'CPROPS':
            # PHANGS-ALMA CPROPS catalog
            basedir = basedir / 'v3p4-CPROPS' / 'STv1p2' / 'native'
            fname_seq = [
                galname.lower(),
                '12m+7m+tp', 'co21', 'native', 'props']
            if res is not None:
                basedir = (
                    basedir / '..' /
                    f"{res.to('pc').value:.0f}pc_matched")
                fname_seq[-2] = f"{res.to('pc').value:.0f}pc"
        elif datatypes[1] == 'rotcurve':
            # PHANGS-ALMA rotation curve
            basedir = basedir / 'rotation_curve'
            fname_seq = [galname] + datatypes[2:]

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

    elif datatypes[0] == 'S4G':
        # S4G data
        basedir = PHANGSdir / 'S4G'
        if datatypes[1] == 'env_mask':
            # S4G morphological maps
            basedir /= 'environmental_masks'
            fname_seq = [galname, 'mask'] + datatypes[2:]
        else:
            fname_seq = [galname, 'S4G'] + datatypes[1:]
            if res is not None:
                fname_seq += [f"{res.to('pc').value:.0f}pc"]

    elif datatypes[0] == 'Halpha':
        # narrow band Halpha data
        basedir = PHANGSdir / 'Halpha'
        fname_seq = [galname] + datatypes[1:]
        if res is not None:
            fname_seq += [f"{res.to('pc').value:.0f}pc"]

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
            self, modelfile=None, r_gal_angle=None,
            colname=None, unit=None):
        t_model = Table.read(modelfile)
        model = interpolate.interp1d(
            t_model['r_gal'].quantity.to('arcsec').value,
            t_model[colname.lower()].quantity.value,
            bounds_error=False, fill_value=np.nan)
        self[colname] = (
            model(r_gal_angle.to('arcsec').value) *
            t_model[colname.lower()].quantity.unit).to(unit)

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

    def add_alphaCO(
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

    def add_Sigma_mol(
            self, COm0file=None, alpha_CO=None, cosi=1.,
            colname=None, unit=None):
        self.calc_image_stats(
            COm0file, stat_func=nanaverage,
            colname='_I_CO', unit='header')
        self[colname] = (self['_I_CO'] * alpha_CO * cosi).to(unit)
        self.table.remove_column('_I_CO')

    # ----------------------------------------------------------------

    def add_Sigma_atom(
            self, HIm0file=None, alpha_21cm=None, cosi=1.,
            colname=None, unit=None):
        self.calc_image_stats(
            HIm0file, stat_func=nanaverage,
            colname='_I_21cm', unit='header')
        self[colname] = (self['_I_21cm'] * alpha_21cm * cosi).to(unit)
        self.table.remove_column('_I_21cm')

    # ----------------------------------------------------------------

    def add_Sigma_SFR(
            self, SFRfile=None, cosi=1.,
            colname=None, unit=None):
        self.calc_image_stats(
            SFRfile, stat_func=nanaverage,
            colname='_SFR', unit='header')
        self[colname] = (self['_SFR'] * cosi).to(unit)
        self.table.remove_column('_SFR')

    # ----------------------------------------------------------------

    def add_Sigma_star(
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

    def add_rho_star(
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

    def add_P_DE(
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
                Sigma_cloud * rho_star_mp * 2 * R_cloud)
            P_DE_atom = (  # Sun+20a Eq.18
                np.pi / 2 * const.G * Sigma_atom**2 +
                np.pi * const.G * Sigma_atom * Sigma_mol +
                Sigma_atom * vdisp_atom_z *
                np.sqrt(2 * const.G * rho_star_mp)) / const.k_B
            self[colname] = (P_DE_cloud + P_DE_atom).to(unit)
        else:
            raise ValueError(f"Unrecognized `scale`: {scale}")

    # ----------------------------------------------------------------

    def add_CO_stats(
            self, header=None,
            tpkmap=None, bm0map=None, sm0map=None, sewmap=None,
            alpha_CO=None, R_cloud=None, H_los=None,
            colname=None, unit=None):

        if not (tpkmap.shape == bm0map.shape ==
                sm0map.shape == sewmap.shape):
            raise ValueError(
                "Input maps have inconsistent shape")
        tpkmap[~np.isfinite(tpkmap)] = np.nan
        bm0map[~np.isfinite(bm0map)] = np.nan
        wtmap = sm0map.copy().astype('float')
        wtmap[~np.isfinite(sm0map) | (sm0map <= 0) |
              ~np.isfinite(sewmap) | (sewmap <= 0)] = 0
        sm0map[wtmap == 0] = np.nan
        sewmap[wtmap == 0] = np.nan

        # area coverage fraction of the strict CO map
        if colname[:5] == 'fracA':
            self.calc_image_stats(
                (sm0map > 0).astype('float'),
                header=header, stat_func=np.nansum,
                colname='_A_strict', unit='')
            self.calc_image_stats(
                np.ones_like(sm0map).astype('float'),
                header=header, stat_func=np.nansum,
                colname='_A_total', unit='')
            self[colname] = (
                self['_A_strict'] / self['_A_total']).to(unit)
            self.table.remove_columns(['_A_strict', '_A_total'])

        # flux recovery fraction of the strict CO map
        elif colname[:5] == 'fracF':
            self.calc_image_stats(
                sm0map.astype('float'),
                header=header, stat_func=np.nansum,
                colname='_F_strict', unit='')
            self.calc_image_stats(
                bm0map.astype('float'),
                header=header, stat_func=np.nansum,
                colname='_F_broad', unit='')
            self[colname] = (
                self['_F_strict'] / self['_F_broad']).to(unit)
            self.table.remove_columns(['_F_strict', '_F_broad'])

        # clumping factor of the strict CO map
        elif colname[:8] == 'clumping':
            self.calc_image_stats(
                sm0map.astype('float'), weight=wtmap,
                header=header, stat_func=nanaverage,
                colname='_F<ICO>', unit='')
            self.calc_image_stats(
                wtmap, weight=None,
                header=header, stat_func=nanaverage,
                colname='_A<ICO>', unit='')
            self[colname] = (
                self['_F<ICO>'] / self['_A<ICO>']).to(unit)
            self.table.remove_columns(['_F<ICO>', '_A<ICO>'])

        else:
            quantity = colname[2:-1].split('_pix_')[0]
            if colname[0] == 'F':
                weight = wtmap
            else:
                weight = None

            # CO line peak temperature
            if quantity == 'T_peak':
                self.calc_image_stats(
                    tpkmap.astype('float'), weight=weight,
                    header=header, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # CO line integrated intensity
            elif quantity == 'I_CO21':
                self.calc_image_stats(
                    sm0map.astype('float'), weight=weight,
                    header=header, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # molecular gas surface density
            elif quantity == 'Sigma_mol':
                colname_ICO = colname.replace('Sigma_mol', 'I_CO21')
                if colname_ICO in self.colnames:
                    self[colname] = (
                        self[colname_ICO] * alpha_CO).to(unit)
                else:
                    self.calc_image_stats(
                        sm0map.astype('float'), weight=weight,
                        header=header, stat_func=nanaverage,
                        colname='_<ICO>', unit='K km s-1')
                    self[colname] = (
                        self['_<ICO>'] * alpha_CO).to(unit)
                    self.table.remove_column('_<ICO>')

            # velocity dispersion = CO line width
            elif quantity == 'vdisp_mol':
                self.calc_image_stats(
                    sewmap.astype('float'), weight=weight,
                    header=header, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # turbulent pressure
            elif quantity == 'P_turb_sph':
                self.calc_image_stats(
                    (sm0map.astype('float') *
                     sewmap.astype('float')**2),
                    header=header,
                    weight=weight, stat_func=nanaverage,
                    colname='_<I*sigv^2>', unit='K km3 s-3')
                self[colname] = (  # Sun+20a Eq.10
                    3/2 * self['_<I*sigv^2>'] / (2*R_cloud) *
                    alpha_CO / const.k_B).to(unit)
                self.table.remove_column('_<I*sigv^2>')
            elif quantity == 'P_turb_cyl':
                colname_sph = colname.replace('cyl', 'sph')
                if colname_sph not in self.colnames:
                    raise ValueError(
                        f"No column in table named {colname_sph}")
                self[colname] = (  # Sun+18 Eq.14
                    self[colname_sph] * 2/3).to(unit)

            # cloud weight due to self-gravity
            elif quantity == 'P_selfg_sph':
                self.calc_image_stats(
                    sm0map.astype('float')**2,
                    header=header,
                    weight=weight, stat_func=nanaverage,
                    colname='_<I^2>', unit='K2 km2 s-2')
                self[colname] = (  # Sun+20a Eq.16
                    3/8 * np.pi * const.G * self['_<I^2>'] *
                    alpha_CO**2 / const.k_B).to(unit)
                self.table.remove_column('_<I^2>')
            elif quantity == 'P_selfg_cyl':
                colname_sph = colname.replace('cyl', 'sph')
                if colname_sph not in self.colnames:
                    raise ValueError(
                        f"No column in table named {colname_sph}")
                self[colname] = (
                    self[colname_sph] * 4/3).to(unit)

            # virial parameter
            elif quantity == 'alpha_vir_sph':
                self.calc_image_stats(
                    (sewmap.astype('float')**2 /
                     sm0map.astype('float')),
                    header=header,
                    weight=weight, stat_func=nanaverage,
                    colname='_<sigv^2/I>', unit='K-1 km s-1')
                self[colname] = (  # Sun+18 Eq.13
                    5 * np.log(2) / (10/9 * np.pi * const.G) *
                    self['_<sigv^2/I>'] /
                    R_cloud / alpha_CO).to(unit)
                self.table.remove_column('_<sigv^2/I>')

            # free-fall time
            elif quantity == 'tau_ff_sph^-1':
                self.calc_image_stats(
                    np.sqrt(sm0map.astype('float')),
                    header=header,
                    weight=weight, stat_func=nanaverage,
                    colname='_<I^0.5>',
                    unit='K(1/2) km(1/2) s(-1/2)')
                self[colname] = (  # not the same as Utomo+18 Eq.6
                    np.sqrt(
                        16/np.pi * const.G * alpha_CO / H_los) *
                    self['_<I^0.5>']).to(unit)
                self.table.remove_column('_<I^0.5>')
            elif quantity == 'tau_ff_cyl^-1':
                colname_sph = colname.replace('cyl', 'sph')
                if colname_sph not in self.colnames:
                    raise ValueError(
                        f"No column in table named {colname_sph}")
                self[colname] = (
                    self[colname_sph] * np.sqrt(2/3)).to(unit)

            else:
                raise ValueError(
                    f"Unrecognized column name: {colname}")

    # ----------------------------------------------------------------

    def add_CPROPS_stats(
            self, cpropscat=None,
            alpha_CO=None, H_los=None, gal_dist=None,
            colname=None, unit=None):

        raarr = np.array(cpropscat['XCTR_DEG'])
        decarr = np.array(cpropscat['YCTR_DEG'])
        fluxarr = (
            cpropscat['FLUX_KKMS_PC2'] / cpropscat['DISTANCE_PC']**2 *
            u.Unit('K km s-1 sr')).to('K km s-1 arcsec2').value
        sigvarr = np.array(cpropscat['SIGV_KMS'])
        radarr = (
            cpropscat['RAD_PC'] / cpropscat['DISTANCE_PC'] *
            u.rad).to('arcsec').value
        wtarr = fluxarr.copy()
        wtarr[~np.isfinite(radarr)] = 0
        fluxarr[~np.isfinite(radarr)] = np.nan
        sigvarr[~np.isfinite(radarr)] = np.nan
        radarr[~np.isfinite(radarr)] = np.nan

        # number of clouds in the CPROPS catalog
        if colname[:4] == 'Nobj':
            self.calc_catalog_stats(
                np.isfinite(fluxarr).astype('int'),
                raarr, decarr, stat_func=np.nansum,
                colname=colname, unit=unit)

        else:
            quantity = colname[2:-1].split('_CPROPS_')[0]
            if colname[0] == 'F':
                weight = wtarr
            else:
                weight = None

            # cloud mass
            if quantity == 'M_mol':
                self.calc_catalog_stats(
                    fluxarr.astype('float'), raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname='_<FCO>', unit='K km s-1 arcsec2')
                self[colname] = (
                    self['_<FCO>'] * alpha_CO * gal_dist**2 /
                    u.sr).to(unit)
                self.table.remove_column('_<FCO>')

            # cloud radius
            elif quantity == 'R':
                self.calc_catalog_stats(
                    radarr, raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname='_<R>', unit='arcsec')
                self[colname] = (
                    self['_<R>'] * gal_dist / u.rad).to(unit)
                self.table.remove_column('_<R>')

            # cloud surface density
            elif quantity == 'Sigma_mol':
                self.calc_catalog_stats(
                    fluxarr / radarr**2, raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname='_<F/R^2>', unit='K km s-1')
                self[colname] = (  # Rosolowsky+21, right after Eq.12
                    self['_<F/R^2>'] * alpha_CO / (2*np.pi)).to(unit)
                self.table.remove_column('_<F/R^2>')

            # cloud velocity dispersion
            elif quantity == 'vdisp_mol':
                self.calc_catalog_stats(
                    sigvarr, raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname=colname, unit=unit)

            # turbulent pressure
            elif quantity == 'P_turb_sph':
                self.calc_catalog_stats(
                    fluxarr * sigvarr**2 / radarr**3, raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname='_<F*sigv^2/R^3>',
                    unit='K km3 s-3 arcsec-1')
                self[colname] = (  # Rosolowsky+21, Eq.16
                    3 / (8 * np.pi) * self['_<F*sigv^2/R^3>'] *
                    alpha_CO / (gal_dist/u.rad) / const.k_B).to(unit)
                self.table.remove_column('_<F*sigv^2/R^3>')
            elif quantity == 'P_turb_cyl':
                self.calc_catalog_stats(
                    fluxarr * sigvarr**2 / radarr**2, raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname='_<F*sigv^2/R^2>',
                    unit='K km3 s-3')
                self[colname] = (  # Rosolowsky+21, Eq.16
                    3 / (4 * np.pi) * self['_<F*sigv^2/R^2>'] *
                    alpha_CO / H_los / const.k_B).to(unit)
                self.table.remove_column('_<F*sigv^2/R^2>')

            # virial parameter
            elif quantity == 'alpha_vir_sph':
                self.calc_catalog_stats(
                    radarr * sigvarr**2 / fluxarr, raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname='_<R*sigv^2/F>',
                    unit='km s-1 K-1 arcsec-1')
                self[colname] = (  # Rosolowsky+21, right above Eq.11
                    10 / const.G * self['_<R*sigv^2/F>'] /
                    alpha_CO / (gal_dist/u.rad)).to(unit)
                self.table.remove_column('_<R*sigv^2/F>')

            # free-fall time
            elif quantity == 'tau_ff_sph^-1':
                self.calc_catalog_stats(
                    np.sqrt(fluxarr / radarr**3), raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname='_<F^0.5/R^1.5>',
                    unit='K(1/2) km(1/2) s(-1/2) arcsec(-1/2)')
                self[colname] = (  # Rosolowsky+21, Eq.17
                    np.sqrt(
                        4 * const.G / np.pi**2 *
                        alpha_CO / (gal_dist/u.rad)) *
                    self['_<F^0.5/R^1.5>']).to(unit)
                self.table.remove_column('_<F^0.5/R^1.5>')
            elif quantity == 'tau_ff_cyl^-1':
                self.calc_catalog_stats(
                    np.sqrt(fluxarr / radarr**2), raarr, decarr,
                    weight=weight, stat_func=nanaverage,
                    colname='_<F^0.5/R>',
                    unit='K(1/2) km(1/2) s(-1/2)')
                self[colname] = (  # Rosolowsky+21, Eq.17
                    np.sqrt(
                        8 * const.G / np.pi ** 2 *
                        alpha_CO / H_los) *
                    self['_<F^0.5/R>']).to(unit)
                self.table.remove_column('_<F^0.5/R>')

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
