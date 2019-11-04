import os
import sys
import warnings
from pathlib import Path
import numpy as np
from astropy import units as u, constants as const
from astropy.table import Table
from AlmaTools.XCO import predict_metallicity, predict_alphaCO10
from mega_table.table import RadialMegaTable
from mega_table.utils import nanaverage


# --------------------------------------------------------------------


def get_data_path(datatype, galname=None, lin_res=None):
    """
    Get the path to any required data on disk.
    """
    datatypes = datatype.split(':')

    # PHANGS data parent directory
    PHANGSdir = Path(os.getenv('PHANGSWORKDIR'))

    if datatypes[0] == 'sample_table':
        return PHANGSdir / 'sample_v1p3_sup.ecsv'

    elif datatypes[0] == 'ALMA':
        # PHANGS-ALMA data
        basedir = PHANGSdir / 'ALMA'
        if datatypes[1] == 'CO':
            # PHANGS-ALMA CO map (v3)
            basedir /= 'v3-processed'
            fname_seq = [galname, 'CO21'] + datatypes[2:]
            if lin_res is not None:
                fname_seq += [f"{lin_res.to('pc').value:.0f}pc"]
        elif datatypes[1] == 'CPROPS':
            # PHANGS-ALMA CPROPS catalog
            basedir /= 'v3-CPROPS'
            fname_seq = [galname.lower(), 'co21', 'native', 'props']
            if lin_res is not None:
                basedir /= f"fixed_{lin_res.to('pc').value:.0f}pc"
                fname_seq[2] = f"{lin_res.to('pc').value:.0f}pc"

    elif datatypes[0] == 'HI':
        # HI data
        basedir = PHANGSdir / 'HI' / 'v0'
        fname_seq = [galname, '21cm'] + datatypes[1:]
        if lin_res is not None:
            fname_seq += [f"{lin_res.to('pc').value:.0f}pc"]

    elif datatypes[0] == 'Z0MGS':
        # Z0MGS data
        basedir = PHANGSdir / 'Z0MGS'
        fname_seq = [galname] + datatypes[1:]
        if lin_res is not None:
            fname_seq += [f"{lin_res.to('pc').value:.0f}pc"]

    elif datatypes[0] == 'S4G':
        # S4G data
        basedir = PHANGSdir / 'S4G'
        if datatypes[1] == 'env_mask':
            # S4G morphological maps
            basedir /= 'environmental_masks'
            fname_seq = [galname, 'mask'] + datatypes[2:]
        else:
            fname_seq = [galname, 'S4G'] + datatypes[1:]
            if lin_res is not None:
                fname_seq += [f"{lin_res.to('pc').value:.0f}pc"]

    return basedir / ('_'.join(fname_seq) + '.fits')


# --------------------------------------------------------------------


def get_R21():
    return 0.7  # CO(2-1)/CO(1-0) ratio

def get_alpha21cm(include_He=True):
    if include_He:  # include the extra 35% mass of Helium
        alpha21cm = 1.97e-2 * u.Msun/u.pc**2/(u.K*u.km/u.s)
    else:
        alpha21cm = 1.46e-2 * u.Msun/u.pc**2/(u.K*u.km/u.s)
    return alpha21cm

def get_alpha3p6um(ref='MS14'):
    if ref == 'MS14':  # Y3.6 = 0.47 Msun/Lsun
        alpha3p6um = 330 * u.Msun/u.pc**2/(u.MJy/u.sr)
    elif ref == 'Q15':  # Y3.6 = 0.6 Msun/Lsun
        alpha3p6um = 420 * u.Msun/u.pc**2/(u.MJy/u.sr)
    else:
        raise ValueError("")
    return alpha3p6um

def get_h_star(Rstar, diskshape='flat', Rgal=None):
    # see Leroy+08 and Ostriker+10
    flat_ratio = 7.3  # Kregel+02
    if diskshape == 'flat':
        hstar = Rstar / flat_ratio
    elif diskshape == 'flared':
        hstar = Rstar / flat_ratio * np.exp(
            (Rgal/Rstar).to('').value - 1)
    else:
        raise ValueError("`diskshape` must be 'flat' or 'flared'")
    return hstar


# --------------------------------------------------------------------


def gen_phys_props_table(
        mt, params, lin_res=None, append_raw_data=False):

    # initiate new table
    t_phys = Table()
    t_phys.meta['NAME'] = str(params['NAME'].strip())
    for key in (
            'RA_DEG', 'DEC_DEG', 'DIST', 'INCL', 'POSANG',
            'LOGMSTAR', 'RDISK_DEG', 'REFF_W1_DEG'):
        t_phys.meta[key] = params[key]

    # galaxy global parameters
    gal_cosi = np.cos(np.deg2rad(params['INCL']))
    gal_dist = params['DIST'] * u.Mpc
    gal_Mstar = 10**params['LOGMSTAR'] * u.Msun
    gal_Rstar = (
        np.deg2rad(params['RDISK_DEG'])*gal_dist).to('kpc')
    gal_Reff = (
        np.deg2rad(params['REFF_W1_DEG'])*gal_dist).to('kpc')

    # coordinates
    t_phys['REGION'] = mt['REGION']
    t_phys['r_gal_min'] = (
        mt['rang_gal_min'].quantity.to('rad').value *
        gal_dist).to('kpc')
    t_phys['r_gal_max'] = (
        mt['rang_gal_max'].quantity.to('rad').value *
        gal_dist).to('kpc')
    r_gal = (t_phys['r_gal_min'].quantity +
             t_phys['r_gal_max'].quantity) / 2

    # metallicity
    t_phys['log(O/H)_PP04'] = predict_metallicity(
        gal_Mstar, calibrator='O3N2(PP04)', MZR='Sanchez+19',
        Rgal=r_gal, Re=gal_Reff, gradient='Sanchez+14')
    t_phys['Zprime'] = Zprime = 10**(
        t_phys['log(O/H)_PP04'] - 8.69)

    # SFR surface density
    for key in ('Sigma_SFR_NUVW3', 'Sigma_SFR_W3ONLY'):
        t_phys[key] = mt[key].quantity.to('Msun kpc-2 yr-1')
        if (np.isfinite(t_phys[key]).sum() != 0 and
            'Sigma_SFR' not in t_phys.colnames):
            t_phys['Sigma_SFR'] = t_phys[key].quantity
    if 'Sigma_SFR' not in t_phys.colnames:
        t_phys['Sigma_SFR'] = np.nan * u.Unit('Msun kpc-2 yr-1')

    # stellar mass surface densities
    alpha3p6um = get_alpha3p6um(ref='MS14')
    for key in ('I_3p6um_ICA', 'I_3p6um_raw'):
        newkey = 'Sigma_star'+key[-4:]
        t_phys[newkey] = (
            mt[key].quantity * gal_cosi * alpha3p6um
        ).to('Msun/pc^2')
        if (np.isfinite(t_phys[newkey]).sum() != 0 and
            'Sigma_star' not in t_phys.colnames):
            t_phys['Sigma_star'] = t_phys[newkey].quantity
    if 'Sigma_star' not in t_phys.colnames:
        t_phys['Sigma_star'] = np.nan * u.Unit('Msun pc-2')
    Sigma_star = t_phys['Sigma_star'].quantity

    # stellar mass volume density near disk mid-plane
    t_phys['rho_star'] = rho_star = (
        t_phys['Sigma_star'] / 4 /
        get_h_star(
            gal_Rstar, diskshape='flat')).to('Msun pc-3')
    t_phys['rho_star_flared'] = rho_star_flared = (
        t_phys['Sigma_star'] / 4 /
        get_h_star(
            gal_Rstar, diskshape='flared',
            Rgal=r_gal)).to('Msun pc-3')

    # HI mass surface density
    alpha21cm = get_alpha21cm(include_He=True)
    t_phys['Sigma_atom'] = (
        mt['I_21cm'].quantity * gal_cosi * alpha21cm
    ).to('Msun pc-2')
    Sigma_atom = t_phys['Sigma_atom'].quantity

    # H2 surface density (low resolution)
    R21 = get_R21()
    rstr_fid = f"{lin_res[-1].to('pc').value:.0f}pc"  # 150 pc
    t_phys['I_CO21'] = I_CO21 = (
        mt[f"Flux_CO21_broad_{rstr_fid}"].quantity /
        mt[f"Area_CO21_total_{rstr_fid}"].quantity).to('K km s-1')
    ICO10kpc = I_CO21 * gal_cosi / R21
    ICO10GMC = mt[f"F<I_CO21_{rstr_fid}>"].quantity / R21
    t_phys['alphaCO10_MW'] = predict_alphaCO10(
        prescription='constant')
    t_phys['alphaCO10_PHANGS'] = predict_alphaCO10(
        prescription='PHANGS', PHANGS_Zprime=Zprime)
    t_phys['alphaCO10_N12'] = predict_alphaCO10(
        prescription='Narayanan+12',
        N12_Zprime=Zprime, N12_WCO10GMC=ICO10GMC)
    t_phys['alphaCO10_B13'] = predict_alphaCO10(
        prescription='Bolatto+13',
        iterative=True, suppress_error=True,
        B13_Zprime=Zprime, B13_Sigmakpc=Sigma_atom+Sigma_star,
        B13_WCO10kpc=ICO10kpc, B13_WCO10GMC=ICO10GMC)
    alphaCO21 = t_phys['alphaCO10_PHANGS'].quantity / R21
    t_phys['Sigma_mol'] = (
        I_CO21 * alphaCO21 * gal_cosi).to('Msun pc-2')
    Sigma_mol = t_phys['Sigma_mol'].quantity

    # CO contribution by environments
    for reg in regions:
        t_phys[f"frac_{reg}"] = mt[f"frac_{reg}"].quantity
    # if no environment masks are available:
    # manually flag all data points (for now)  # <----------------
    if ((np.isfinite(t_phys['frac_bulge']).sum() == 0) and
        (np.isfinite(t_phys['frac_bars']).sum() == 0)):
        t_phys['frac_bulge'] = 1.

    # CO map statistics
    for res in lin_res:
        R_cloud = res / 2
        rstr = f"{res.to('pc').value:.0f}pc"
        t_phys[f"fracA_CO21_{rstr}"] = (
            mt[f"Area_CO21_strict_{rstr}"].quantity /
            mt[f"Area_CO21_total_{rstr}"].quantity).to('')
        t_phys[f"fracF_CO21_{rstr}"] = (
            mt[f"Flux_CO21_strict_{rstr}"].quantity /
            mt[f"Flux_CO21_broad_{rstr}"].quantity).to('')
        t_phys[f"clumping_CO21_{rstr}"] = (
            mt[f"F<I_CO21_{rstr}>"].quantity /
            mt[f"A<I_CO21_{rstr}>"].quantity).to('')
        t_phys[f"A<Sigma_mol_pix_{rstr}>"] = (
            mt[f"A<I_CO21_{rstr}>"].quantity *
            alphaCO21).to('Msun pc-2')
        t_phys[f"F<Sigma_mol_pix_{rstr}>"] = (
            mt[f"F<I_CO21_{rstr}>"].quantity *
            alphaCO21).to('Msun pc-2')
        t_phys[f"A<vdisp_mol_pix_{rstr}>"] = (
            mt[f"A<sigv_CO21_{rstr}>"].quantity).to('km s-1')
        t_phys[f"F<vdisp_mol_pix_{rstr}>"] = (
            mt[f"F<sigv_CO21_{rstr}>"].quantity).to('km s-1')
        t_phys[f"A<P_turb_pix_{rstr}>"] = (  # Sun+20 Eq.4
            (3/2) * mt[f"A<I*sigv^2_CO21_{rstr}>"].quantity /
            (2*R_cloud) * alphaCO21 / const.k_B).to('K cm-3')
        t_phys[f"F<P_turb_pix_{rstr}>"] = (  # Eq.4 in Sun+20
            (3/2) * mt[f"F<I*sigv^2_CO21_{rstr}>"].quantity /
            (2*R_cloud) * alphaCO21 / const.k_B).to('K cm-3')
        t_phys[f"A<alpha_vir_pix_{rstr}>"] = (  # Eq.13 in Sun+18
            5 * np.log(2) / (10/9 * np.pi * const.G) *
            mt[f"A<sigv^2/I_CO21_{rstr}>"].quantity / R_cloud /
            alphaCO21).to('')
        t_phys[f"F<alpha_vir_pix_{rstr}>"] = (  # Eq.13 in Sun+18
            5 * np.log(2) / (10/9 * np.pi * const.G) *
            mt[f"F<sigv^2/I_CO21_{rstr}>"].quantity / R_cloud /
            alphaCO21).to('')

    # CPROPS cloud statistics
    R_factor = np.sqrt(5) / 1.91  # Rosolowsky&Leroy06 Sec.3.1
    for res in lin_res:
        rstr = f"{res.to('pc').value:.0f}pc"
        t_phys[f"Nobj_CPROPS_{rstr}"] = (
            mt[f"Nobj_CPROPS_{rstr}"].quantity).to('')
        t_phys[f"fracF_CPROPS_{rstr}"] = (
            mt[f"Flux_CPROPS_total_{rstr}"].quantity /
            mt[f"Flux_CO21_broad_{rstr}"].quantity).to('')
        t_phys[f"U<M_mol_CPROPS_{rstr}>"] = (
            # Note that F [=] K*km/s arcsec2
            mt[f"U<F_CPROPS_{rstr}>"].quantity * alphaCO21 *
            gal_dist**2 / u.sr).to('Msun')
        t_phys[f"F<M_mol_CPROPS_{rstr}>"] = (
            # Note that F [=] K*km/s arcsec2
            mt[f"F<F_CPROPS_{rstr}>"].quantity * alphaCO21 *
            gal_dist**2 / u.sr).to('Msun')
        t_phys[f"U<R_CPROPS_{rstr}>"] = (
            # Note that R [=] arcsec
            mt[f"U<R_CPROPS_{rstr}>"].quantity *
            gal_dist / u.rad).to('pc')
        t_phys[f"F<R_CPROPS_{rstr}>"] = (
            # Note that R [=] arcsec
            mt[f"F<R_CPROPS_{rstr}>"].quantity *
            gal_dist / u.rad).to('pc')
        t_phys[f"U<Sigma_mol_CPROPS_{rstr}>"] = (
            mt[f"U<F/R^2_CPROPS_{rstr}>"].quantity * alphaCO21 /
            (np.pi*R_factor**2)).to('Msun pc-2')
        t_phys[f"F<Sigma_mol_CPROPS_{rstr}>"] = (
            mt[f"F<F/R^2_CPROPS_{rstr}>"].quantity * alphaCO21 /
            (np.pi*R_factor**2)).to('Msun pc-2')
        t_phys[f"U<vdisp_mol_CPROPS_{rstr}>"] = (
            mt[f"U<sigv_CPROPS_{rstr}>"].quantity).to('km s-1')
        t_phys[f"F<vdisp_mol_CPROPS_{rstr}>"] = (
            mt[f"F<sigv_CPROPS_{rstr}>"].quantity).to('km s-1')
        t_phys[f"U<P_turb_CPROPS_{rstr}>"] = (
            3 / (4 * np.pi) *
            mt[f"U<F*sigv^2/R^3_CPROPS_{rstr}>"].quantity *
            alphaCO21 / R_factor**3 / (gal_dist / u.rad) /
            const.k_B).to('K cm-3')
        t_phys[f"F<P_turb_CPROPS_{rstr}>"] = (
            3 / (4 * np.pi) *
            mt[f"F<F*sigv^2/R^3_CPROPS_{rstr}>"].quantity *
            alphaCO21 / R_factor**3 / (gal_dist / u.rad) /
            const.k_B).to('K cm-3')
        t_phys[f"U<alpha_vir_CPROPS_{rstr}>"] = (  # Sun+18 Eq.6
            5 / const.G *
            mt[f"U<R*sigv^2/F_CPROPS_{rstr}>"].quantity /
            alphaCO21 * R_factor / (gal_dist / u.rad)).to('')
        t_phys[f"F<alpha_vir_CPROPS_{rstr}>"] = (  # Sun+18 Eq.6
            5 / const.G *
            mt[f"F<R*sigv^2/F_CPROPS_{rstr}>"].quantity /
            alphaCO21 * R_factor / (gal_dist / u.rad)).to('')

    # dynamical equilibrium pressure estimates
    Sigma_gas = Sigma_mol + Sigma_atom
    res_fid = lin_res[-2].to('pc')  # 120 pc
    rstr_fid = f"{res_fid.value:.0f}pc"
    vdisp_mol_z = t_phys[f"F<vdisp_mol_pix_{rstr_fid}>"].quantity
    vdisp_atom_z = 10 * u.Unit('km s-1')
    vdisp_z = (
        (vdisp_mol_z * Sigma_mol + vdisp_atom_z * Sigma_atom) /
        Sigma_gas).to('km s-1')
    t_phys["P_DE_classic"] = (
        (np.pi * const.G / 2 * Sigma_gas**2 +
         Sigma_gas * vdisp_z * np.sqrt(2*const.G*rho_star)) /
        const.k_B).to('K cm-3')
    t_phys["P_DE_classic_flared"] = (
        (np.pi * const.G / 2 * Sigma_gas**2 +
         Sigma_gas * vdisp_z *
         np.sqrt(2*const.G*rho_star_flared)) /
        const.k_B).to('K cm-3')
    t_phys["P_DE_classic_fixedsiggas"] = (
        (np.pi * const.G / 2 * Sigma_gas**2 +
         Sigma_gas * vdisp_atom_z *
         np.sqrt(2*const.G*rho_star)) /
        const.k_B).to('K cm-3')
    # t_phys["W_atom"] = (
    #     (np.pi * const.G / 2 * Sigma_atom**2 +
    #      np.pi * const.G * Sigma_atom * Sigma_mol +
    #      Sigma_atom * vdisp_atom_z *
    #      np.sqrt(2*const.G*rho_star)) /
    #     const.k_B).to('K cm-3')
    # t_phys["P_DE_smooth"] = (
    #     (np.pi * const.G / 2 * Sigma_mol**2 +
    #      np.pi * const.G * Sigma_mol * rho_star * res_fid/2) /
    #     const.k_B).to('K cm-3') + t_phys['W_atom'].quantity
    # for res in lin_res:
    #     R_cloud = res / 2
    #     rstr = f"{res.to('pc').value:.0f}pc"
    #     t_phys[f"A<W_cloud_self_pix_{rstr}>"] = (
    #         3/8 * np.pi * const.G *
    #         mt[f"A<I^2_CO21_{rstr}>"].quantity * alphaCO21**2 /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"F<W_cloud_self_pix_{rstr}>"] = (
    #         3/8 * np.pi * const.G *
    #         mt[f"F<I^2_CO21_{rstr}>"].quantity * alphaCO21**2 /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"A<W_cloud_mol_pix_{rstr}>"] = (
    #         np.pi * const.G / 2 * Sigma_mol *
    #         t_phys[f"A<Sigma_mol_pix_{rstr}>"] /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"F<W_cloud_mol_pix_{rstr}>"] = (
    #         np.pi * const.G / 2 * Sigma_mol *
    #         t_phys[f"F<Sigma_mol_pix_{rstr}>"] /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"A<W_cloud_star_pix_{rstr}>"] = (
    #         3/2 * np.pi * const.G * rho_star * R_cloud *
    #         t_phys[f"A<Sigma_mol_pix_{rstr}>"] /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"F<W_cloud_star_pix_{rstr}>"] = (
    #         3/2 * np.pi * const.G * rho_star * R_cloud *
    #         t_phys[f"F<Sigma_mol_pix_{rstr}>"] /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"A<P_DE_pix_{rstr}>"] = (
    #         t_phys[f"A<W_cloud_self_pix_{rstr}>"].quantity +
    #         t_phys[f"A<W_cloud_mol_pix_{rstr}>"].quantity +
    #         t_phys[f"A<W_cloud_star_pix_{rstr}>"].quantity +
    #         t_phys["W_atom"].quantity)
    #     t_phys[f"F<P_DE_pix_{rstr}>"] = (
    #         t_phys[f"F<W_cloud_self_pix_{rstr}>"].quantity +
    #         t_phys[f"F<W_cloud_mol_pix_{rstr}>"].quantity +
    #         t_phys[f"F<W_cloud_star_pix_{rstr}>"].quantity +
    #         t_phys["W_atom"].quantity)
    # for res in lin_res:
    #     rstr = f"{res.to('pc').value:.0f}pc"
    #     t_phys[f"U<W_cloud_self_CPROPS_{rstr}>"] = (
    #         3 * const.G / 8 / np.pi *
    #         mt[f"U<F^2/R^4_CPROPS_{rstr}>"].quantity *
    #         alphaCO21**2 / R_factor**4 /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"F<W_cloud_self_CPROPS_{rstr}>"] = (
    #         3 * const.G / 8 / np.pi *
    #         mt[f"F<F^2/R^4_CPROPS_{rstr}>"].quantity *
    #         alphaCO21**2 / R_factor**4 /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"U<W_cloud_mol_CPROPS_{rstr}>"] = (
    #         np.pi * const.G / 2 * Sigma_mol *
    #         t_phys[f"U<Sigma_mol_CPROPS_{rstr}>"].quantity /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"F<W_cloud_mol_CPROPS_{rstr}>"] = (
    #         np.pi * const.G / 2 * Sigma_mol *
    #         t_phys[f"F<Sigma_mol_CPROPS_{rstr}>"].quantity /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"U<W_cloud_star_CPROPS_{rstr}>"] = (
    #         3/2 * const.G * rho_star *
    #         mt[f"U<F/R_CPROPS_{rstr}>"].quantity *
    #         alphaCO21 / R_factor * (gal_dist / u.rad) /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"F<W_cloud_star_CPROPS_{rstr}>"] = (
    #         3/2 * const.G * rho_star *
    #         mt[f"F<F/R_CPROPS_{rstr}>"].quantity *
    #         alphaCO21 / R_factor * (gal_dist / u.rad) /
    #         const.k_B).to('K cm-3')
    #     t_phys[f"U<P_DE_CPROPS_{rstr}>"] = (
    #         t_phys[f"U<W_cloud_self_CPROPS_{rstr}>"].quantity +
    #         t_phys[f"U<W_cloud_mol_CPROPS_{rstr}>"].quantity +
    #         t_phys[f"U<W_cloud_star_CPROPS_{rstr}>"].quantity +
    #         t_phys["W_atom"].quantity)
    #     t_phys[f"F<P_DE_CPROPS_{rstr}>"] = (
    #         t_phys[f"F<W_cloud_self_CPROPS_{rstr}>"].quantity +
    #         t_phys[f"F<W_cloud_mol_CPROPS_{rstr}>"].quantity +
    #         t_phys[f"F<W_cloud_star_CPROPS_{rstr}>"].quantity +
    #         t_phys["W_atom"].quantity)

    if append_raw_data:
        for key in mt.colnames:
            if key not in t_phys.colnames:
                t_phys[key] = mt[key].quantity

    return t_phys


######################################################################
######################################################################
##
##  Pipeline main body starts from here
##
######################################################################
######################################################################


if __name__ == '__main__':

    # ----------------------------------------------------------------

    # working directory
    workdir = Path(__file__).parent

    # warnings & logging settings
    warnings.filterwarnings('ignore')
    logging = False
    if logging:
        # output log to a file
        orig_stdout = sys.stdout
        log = open(workdir/(str(Path(__file__).stem)+'.log'), 'w')
        sys.stdout = log

    # ----------------------------------------------------------------

    # ring (deprojected) width
    ring_width = 0.5 * u.kpc

    # maximal (depojected) galactic radius
    max_rgal = 20 * u.kpc

    # ----------------------------------------------------------------

    # (linear) resolutions of the PHANGS-ALMA data
    lin_res = np.array([60, 90, 120, 150]) * u.pc

    # list of morphological regions in environmental masks
    regions = ('disk', 'bulge', 'bars', 'rings', 'lenses', 'sp_arms')

    # ----------------------------------------------------------------

    # read PHANGS sample table
    catalog = Table.read(get_data_path('sample_table'))
    # only keep targets with the 'ALMA' tag
    catalog = catalog[catalog['ALMA'] == 1]

    # loop through sample table
    for row in catalog:

        # galaxy parameters
        name = row['NAME'].strip()
        ra = row['RA_DEG'] * u.deg
        dec = row['DEC_DEG'] * u.deg
        dist = row['DIST'] * u.Mpc
        incl = row['INCL'] * u.deg
        posang = row['POSANG'] * u.deg

        # skip targets with bad geometrical information
        if not ((incl >= 0*u.deg) and (incl < 90*u.deg) and
                np.isfinite(posang)):
            continue

        mtfile = (workdir /
                  f"{name}_radial_profile_"
                  f"{ring_width.to('pc').value:.0f}pc.ecsv")
        # skip targets with mega-table already on disk
        if mtfile.is_file():
            print(f"Table file already on disk - skipping {name}")
            continue

        print(f"Processing data for {name}")

        # initialize table
        print("  Initializing data table")
        ring_width_deg = (ring_width/dist*u.rad).to('deg').value
        mt = RadialMegaTable(
            ra.value, dec.value, ring_width_deg,
            gal_incl_deg=incl.value, gal_pa_deg=posang.value,
            max_rgal_deg = (max_rgal/dist*u.rad).to('deg').value)

        # add environmental fraction (CO flux-weighted) in table
        print("  Calculating (CO flux-weighted) "
              "environmental fraction")
        res = lin_res[-1]
        wtfile = get_data_path('ALMA:CO:mom0:strict', name, res)
        for reg in regions:
            print(f"    > fraction of {reg}")
            envfile = get_data_path('S4G:env_mask:'+reg, name)
            mt.calc_env_frac(
                envfile, wtfile, colname='frac_'+reg)

        # add Z0MGS data in table
        print("  Resampling Z0MGS data")
        infile = get_data_path('Z0MGS:SFR:NUVW3', name)
        mt.calc_image_stats(
            infile, stat_func=nanaverage,
            colname='Sigma_SFR_NUVW3',
            unit=u.Unit('Msun kpc-2 yr-1'))
        infile = get_data_path('Z0MGS:SFR:W3ONLY', name)
        mt.calc_image_stats(
            infile, stat_func=nanaverage,
            colname='Sigma_SFR_W3ONLY',
            unit=u.Unit('Msun kpc-2 yr-1'))

        # add S4G data in table
        print("  Resampling S4G data")
        infile = get_data_path('S4G:ICA3p6um', name)
        mt.calc_image_stats(
            infile, stat_func=nanaverage,
            colname='I_3p6um_ICA', unit=u.Unit('MJy sr-1'))
        infile = get_data_path('S4G:3p6um', name)
        mt.calc_image_stats(
            infile, stat_func=nanaverage,
            colname='I_3p6um_raw', unit=u.Unit('MJy sr-1'))

        # add HI data in table
        print("  Resampling HI data")
        infile = get_data_path('HI:mom0', name)
        mt.calc_image_stats(
            infile, stat_func=nanaverage,
            colname='I_21cm', unit=u.Unit('K km s-1'))

        # add statistics of high resolution CO data in table
        print("  Calculating statistics of high resolution CO data")
        for res in lin_res:
            print(f"    @ {res.to('pc').value:.0f}pc resolution")
            bm0file = get_data_path('ALMA:CO:mom0:broad', name, res)
            sm0file = get_data_path('ALMA:CO:mom0:strict', name, res)
            sewfile = get_data_path('ALMA:CO:ew:strict', name, res)
            mt.calc_CO_stats(bm0file, sm0file, sewfile, res)

        # add statistics of CPROPS clouds in table
        print("  Calculating statistics of CPROPS clouds")
        for res in lin_res:
            print(f"    @ {res.to('pc').value:.0f}pc resolution")
            cpropsfile = get_data_path('ALMA:CPROPS', name, res)
            mt.calc_cprops_stats(cpropsfile, res)

        # # only keep rings in which the total CO flux is positive
        # rstr = f"{lin_res[-1].value:.0f}pc"  # 150 pc
        # mt.clean(keep_positive=f"Flux_CO21_broad_{rstr}")

        # write table to disk
        print("  Writing table to disk")
        mt.write(mtfile, overwrite=True)
        del mt

        print(f"Finished processing data for {name}!")
        print("")

    # ----------------------------------------------------------------

    for row in catalog:

        name = row['NAME'].strip()
        mtfile = (workdir /
                  f"{name}_radial_profile_"
                  f"{ring_width.to('pc').value:.0f}pc.ecsv")
        physfile = (workdir /
                  f"{name}_radial_profile_"
                  f"{ring_width.to('pc').value:.0f}pc_phys.ecsv")
        if (not mtfile.is_file() or physfile.is_file()):
            continue

        # convert raw observables to physical properties
        print(f"Constructing physical property table for {name}")
        mt = Table.read(mtfile)
        t_phys = gen_phys_props_table(mt, row, lin_res=lin_res)

        # write table to disk
        print("  Writing table to disk")
        t_phys.write(physfile, overwrite=True)
        del mt, t_phys
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
