import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
from astropy import units as u, constants as const
from astropy.table import Table
from astropy.io import fits
from reproject import reproject_interp
from mega_table.table import RadialMegaTable
from mega_table.utils import nanaverage
from CO_conversion_factor import metallicity, alphaCO

warnings.filterwarnings('ignore')

logging = False

# --------------------------------------------------------------------


def get_data_path(datatype, galname=None, lin_res=None):
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
        elif datatypes[1] == 'alphaCO':
            # PHANGS alphaCO map
            basedir = basedir / 'alphaCO'  # / 'v0p1'
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

    return basedir / ('_'.join(fname_seq) + '.fits')


# --------------------------------------------------------------------


class PhangsRadialMegaTable(RadialMegaTable):

    """
    Enhanced RadialMegaTable.
    """

    # ----------------------------------------------------------------

    def add_metallicity(
            self, Mstar=None, r_gal=None, Re=None, logOH_solar=None,
            colname=None, unit=None):
        logOH_Re = metallicity.predict_logOH_SAMI19(Mstar)
        logOH = metallicity.extrapolate_logOH_radially(
            logOH_Re, gradient='CALIFA14', Rgal=r_gal, Re=Re)
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

    # ----------------------------------------------------------------

    def add_Sigma_mol(
            self, I_CO=None, alpha_CO=None, cosi=1.,
            colname=None, unit=None):
        self[colname] = (I_CO * alpha_CO * cosi).to(unit)

    # ----------------------------------------------------------------

    def add_Sigma_atom(
            self, I_21cm=None, alpha_21cm=None, cosi=1.,
            colname=None, unit=None):
        self[colname] = (I_21cm * alpha_21cm * cosi).to(unit)

    # ----------------------------------------------------------------

    def add_Sigma_star(
            self, image,
            MtoL=None, band='3p4um', Lsun_IR=None, cosi=1.,
            colname=None, unit=None, **kwargs):
        self.calc_image_stats(
            image, colname='_I_IR', unit='MJy sr-1', **kwargs)
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
            self, Sigma_gas=None, rho_star_mp=None, vdisp_gas_z=None,
            colname=None, unit=None):
        if vdisp_gas_z is None:
            vdisp_gas = 11 * u.Unit('km s-1')  # Leroy+08
        else:
            vdisp_gas = vdisp_gas_z
        self[colname] = (
            (np.pi * const.G / 2 * Sigma_gas**2 +
             Sigma_gas * vdisp_gas *
             np.sqrt(2 * const.G * rho_star_mp)) /
            const.k_B).to(unit)

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
        wtmap = np.ones_like(sm0map).astype('float')
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
                sm0map.astype('float'), weight=sm0map.astype('float'),
                header=header, stat_func=nanaverage,
                colname='_F<ICO>', unit='')
            self.calc_image_stats(
                sm0map.astype('float'), weight=None,
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
            self, envfile=None, wfile=None, colname=None, unit=None,
            **kwargs):
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
            (envmap > 0).astype('float'),
            header=hdr, stat_func=nanaverage,
            weight=wmap, colname=colname, unit=unit, **kwargs)


# --------------------------------------------------------------------


def gen_radial_mega_table(
        config, gal_params={}, phys_params={},
        rgal_bin_kpc=None, rgal_max_kpc=None,
        verbose=True, note='', version=0.0, writefile=''):

    rgal_bin_arcsec = np.rad2deg(
        rgal_bin_kpc / gal_params['dist_Mpc'] / 1e3) * 3600
    if rgal_max_kpc is None:
        rgal_max_arcsec = None
    else:
        rgal_max_arcsec = np.rad2deg(
            rgal_max_kpc / gal_params['dist_Mpc'] / 1e3) * 3600

    # galaxy parameters
    gal_name = gal_params['name']
    gal_cosi = np.cos(np.deg2rad(gal_params['incl_deg']))
    gal_dist = gal_params['dist_Mpc'] * u.Mpc
    gal_Reff = (
        (gal_params['Reff_arcsec'] * u.arcsec).to('rad').value *
        gal_params['dist_Mpc'] * u.Mpc).to('kpc')
    gal_Rstar = (
        (gal_params['Rstar_arcsec'] * u.arcsec).to('rad').value *
        gal_params['dist_Mpc'] * u.Mpc).to('kpc')

    # initialize table
    if verbose:
        print("  Initializing mega table")
    t = PhangsRadialMegaTable(
        gal_params['ra_deg'], gal_params['dec_deg'], rgal_bin_arcsec,
        rgal_max_arcsec=rgal_max_arcsec,
        gal_incl_deg=gal_params['incl_deg'],
        gal_posang_deg=gal_params['posang_deg'])

    if 'coord' in config['group']:
        # ring inner/outer boundaries
        if verbose:
            print("  Calculating ring inner/outer boundaries")
        t['r_gal_min'] = (
            t['r_gal_angl_min'].to('rad').value *
            gal_params['dist_Mpc']*u.Mpc).to('kpc')
        t['r_gal_max'] = (
            t['r_gal_angl_max'].to('rad').value *
            gal_params['dist_Mpc']*u.Mpc).to('kpc')

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
        if 'MtoL_3p4um' not in t.colnames:
            raise ValueError("No stellar M/L ratio data found")
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
            t.add_Sigma_star(
                get_data_path(
                    row['source'], gal_name, row['res_pc']*u.pc),
                MtoL=t['MtoL_3p4um'], band=band, Lsun_IR=Lsun,
                cosi=gal_cosi,
                colname=row['colname'], unit=u.Unit(row['unit']),
                suppress_error=True, stat_func=nanaverage)
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
                if 'I_21cm' not in t.colnames:
                    raise ValueError("No I_21cm column found")
                alpha_21cm = (
                    phys_params['HI_alpha21cm'] *
                    u.Unit(phys_params['HI_alpha21cm_unit']))
                t.add_Sigma_atom(
                    I_21cm=t['I_21cm'], alpha_21cm=alpha_21cm,
                    cosi=gal_cosi,
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
                if 'r_gal_min' not in t.colnames:
                    raise ValueError("No coordinate info found")
                t.add_metallicity(
                    Mstar=gal_params['Mstar_Msun']*u.Msun,
                    r_gal=(t['r_gal_min']+t['r_gal_max'])/2,
                    Re=gal_Reff,
                    logOH_solar=phys_params['abundance_solar'],
                    colname=row['colname'], unit=row['unit'])
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
                t.add_alphaCO(
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
                if 'I_CO21' not in t.colnames:
                    raise ValueError("No I_CO column found")
                if 'alphaCO21' not in t.colnames:
                    raise ValueError("No alphaCO column found")
                t.add_Sigma_mol(
                    I_CO=t['I_CO21'], alpha_CO=t['alphaCO21'],
                    cosi=gal_cosi,
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
            phys_params['CO_los_depth'] *
            u.Unit(phys_params['CO_los_depth_unit']))
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
                    t.add_CPROPS_stats(
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
            phys_params['CO_los_depth'] *
            u.Unit(phys_params['CO_los_depth_unit']))
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
                    t.add_CO_stats(
                        colname=row['colname'], unit=row['unit'],
                        header=hdr, tpkmap=tpkmap, bm0map=bm0map,
                        sm0map=sm0map, sewmap=sewmap,
                        alpha_CO=t['alphaCO21'],
                        R_cloud=res/2, H_los=H_los)
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
                t.add_rho_star(
                    Sigma_star=t['Sigma_star'],
                    Rstar=gal_Rstar, diskshape='flat',
                    colname=row['colname'], unit=row['unit'])
            elif row['colname'] in ('P_DE_L08', 'P_DE_S20'):
                if 'Sigma_mol' not in t.colnames:
                    raise ValueError("No Sigma_mol column found")
                if 'Sigma_atom' not in t.colnames:
                    raise ValueError("No Sigma_atom column found")
                if 'rho_star_mp' not in t.colnames:
                    raise ValueError("No rho_star_mp column found")
                if row['colname'] == 'P_DE_L08':
                    t.add_P_DE(
                        Sigma_gas=t['Sigma_mol']+t['Sigma_atom'],
                        rho_star_mp=t['rho_star_mp'],
                        colname=row['colname'], unit=row['unit'])
                else:
                    if 'CO_stats' in config['group']:
                        rows = config[config['group'] == 'CO_stats']
                        res_fid = np.max(rows['res_pc'])
                        vdisp_col = f"F<vdisp_mol_pix_{res_fid}pc>"
                        if vdisp_col not in t.colnames:
                            raise ValueError("No vdisp data found")
                        vdisp_gas_z = t[vdisp_col]
                    else:
                        vdisp_gas_z = np.nan * u.Unit('km s-1')
                    t.add_P_DE(
                        Sigma_gas=t['Sigma_mol'] + t['Sigma_atom'],
                        rho_star_mp=t['rho_star_mp'],
                        vdisp_gas_z=vdisp_gas_z,
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
            t.calc_image_stats(
                get_data_path(
                    row['source'], gal_name, row['res_pc']*u.pc),
                colname=row['colname'], unit=u.Unit(row['unit']),
                suppress_error=True, stat_func=nanaverage)
            if np.isfinite(t['Sigma_SFR']).sum() == 0:
                t['Sigma_SFR'] = t[row['colname']]
                t['Sigma_SFR'].description = (
                    f"({row['colname'].replace('Sigma_SFR_', '')})")

    # clean and format table
    new_table = Table(t[list(config['colname'])])
    for row in config:
        postfix = ''
        if hasattr(t[row['colname']], 'description'):
            if t[row['colname']].description is not None:
                postfix = ' ' + t[row['colname']].description
        if isinstance(t[row['colname']], u.Quantity):
            new_table[row['colname']] = (
                t[row['colname']].to(row['unit']))
        else:
            new_table[row['colname']] = t[row['colname']]
        new_table[row['colname']].info.format = str(row['format'])
        new_table[row['colname']].info.description = (
            str(row['description']) + postfix)
    t.table = new_table

    # record metadata
    t.meta['GALAXY'] = str(gal_name)
    t.meta['DIST_MPC'] = gal_params['dist_Mpc']
    t.meta['RA_DEG'] = gal_params['ra_deg']
    t.meta['DEC_DEG'] = gal_params['dec_deg']
    t.meta['INCL_DEG'] = gal_params['incl_deg']
    t.meta['PA_DEG'] = gal_params['posang_deg']
    t.meta['LOGMSTAR'] = np.log10(gal_params['Mstar_Msun'])
    t.meta['REFF_KPC'] = gal_Reff.to('kpc').value
    t.meta['RDIS_KPC'] = gal_Rstar.to('kpc').value
    t.meta['CO_R21'] = phys_params['CO_R21']
    t.meta['H_MOL_PC'] = phys_params['CO_los_depth']
    t.meta['ABUN_SUN'] = phys_params['abundance_solar']
    t.meta['TBLNOTE'] = str(note)
    t.meta['VERSION'] = float(version)

    # output
    if writefile:
        t.write(writefile, add_timestamp=True, overwrite=True)
        return writefile
    else:
        return t


######################################################################
######################################################################


if __name__ == '__main__':

    # ring (deprojected) width
    rgal_bin = 0.5 * u.kpc

    # maximal (depojected) galactic radius
    rgal_max = 25 * u.kpc

    # ----------------------------------------------------------------

    # working directory
    workdir = Path(__file__).parent

    # warnings & logging settings
    if logging:
        # output log to a file
        orig_stdout = sys.stdout
        log = open(workdir/(str(Path(__file__).stem)+'.log'), 'w')
        sys.stdout = log
    else:
        orig_stdout = log = None

    # read configuration file
    config = Table.read(
        workdir /
        f"config_annulus_{rgal_bin.to('pc').value:.0f}pc.csv")

    # read physical parameter file
    with open(workdir / "config_params.json") as f:
        phys_params = json.load(f)

    # read PHANGS sample table
    t_sample = Table.read(get_data_path('sample'))
    # only keep targets with the 'HAS_ALMA' tag
    t_sample = t_sample[t_sample['HAS_ALMA'] == 1]
    # loop through sample table
    for row in t_sample:

        # galaxy parameters
        gal_params = {
            'name': row['NAME'].strip(),
            'dist_Mpc': row['DIST'],
            'ra_deg': row['ORIENT_RA'],
            'dec_deg': row['ORIENT_DEC'],
            'incl_deg': row['ORIENT_INCL'],
            'posang_deg': row['ORIENT_POSANG'],
            'Mstar_Msun': row['MSTAR_MAP'] * 10**0.21,
            'Reff_arcsec': row['SIZE_LSTAR_MASS'] * 1.67,
            'Rstar_arcsec': row['SIZE_LSTAR_S4G'],
        }

        # skip targets with bad geometrical information
        if not (0 <= gal_params['incl_deg'] < 90 and
                np.isfinite(gal_params['posang_deg'])):
            print(
                f"Bad orientation measurement - skipping "
                f"{gal_params['name']}")
            print("")
            continue
        # skip targets with bad distance
        if not gal_params['dist_Mpc'] > 0:
            print(
                f"Bad distance measurement - skipping "
                f"{gal_params['name']}")
            print("")
            continue

        print(f"Processing data for {gal_params['name']}")

        mtfile = (
            workdir /
            f"{gal_params['name']}_annulus_stats_"
            f"{rgal_bin.to('pc').value:.0f}pc.fits")
        if not mtfile.is_file():
            print(f"Constructing mega-table for {gal_params['name']}")
            gen_radial_mega_table(
                config, gal_params=gal_params, phys_params=phys_params,
                rgal_bin_kpc=rgal_bin.to('kpc').value,
                rgal_max_kpc=rgal_max.to('kpc').value,
                note=(
                    'PHANGS-ALMA v3.4; '
                    'CPROPS catalogs v3.4 (ST1.2); '
                    'PHANGS-VLA v1.0; '
                    'PHANGS-Halpha v0.1&0.3; '
                    'sample table v1.4 (dist=v1.2)'),
                version=1.3, writefile=mtfile)

        # ------------------------------------------------------------

        mtfile_new = (
            workdir /
            f"{gal_params['name']}_annulus_stats_"
            f"{rgal_bin.to('pc').value:.0f}pc.ecsv")
        if mtfile.is_file() and not mtfile_new.is_file():
            print("Converting FITS table to ECSV format")
            t = Table.read(mtfile)
            t.write(mtfile_new, delimiter=',', overwrite=True)

        # ------------------------------------------------------------

        print(f"Finished processing data for {gal_params['name']}!")
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
