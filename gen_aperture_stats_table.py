import os
import sys
import warnings
from pathlib import Path
import numpy as np
from astropy import units as u, constants as const
from astropy.table import Table
from astropy.io import fits
from AlmaTools.XCO import predict_metallicity, predict_alphaCO10
from mega_table.table import TessellMegaTable
from mega_table.utils import deproject, nanaverage


# --------------------------------------------------------------------


def get_data_path(datatype, galname=None, lin_res=None):
    """
    Get the path to any required data on disk.
    """
    datatypes = datatype.split(':')

    # PHANGS data parent directory
    PHANGSdir = Path(os.getenv('PHANGSWORKDIR'))

    if datatypes[0] == 'sample_table':
        # return PHANGSdir / 'sample_v1p3_sup.ecsv'
        return (PHANGSdir / 'mega-tables' /
                'sample_table_for_Sun+19.fits')

    elif datatypes[0] == 'ALMA':
        # PHANGS-ALMA data
        basedir = PHANGSdir / 'ALMA'
        if datatypes[1] == 'CO':
            # PHANGS-ALMA CO map (v4)
            basedir /= 'v4-processed'
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
        basedir = PHANGSdir / 'HI'
        fname_seq = [galname, '21cm'] + datatypes[1:]
        if lin_res is not None:
            fname_seq += [f"{lin_res.to('pc').value:.0f}pc"]

    elif datatypes[0] == 'z0MGS':
        # z0MGS data
        basedir = PHANGSdir / 'z0MGS'
        fname_seq = [galname] + datatypes[1:]
        if lin_res is not None:
            fname_seq += [f"{lin_res.to('pc').value:.0f}pc"]
        else:
            fname_seq += ['gauss15']

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


def gen_raw_measurement_table(
        gal_name, aperture_shape, aperture_size_kpc,
        CO_res_pc=[], env_regions=[],
        gal_ra_deg=None, gal_dec_deg=None, gal_dist_Mpc=None,
        gal_incl_deg=None, gal_posang_deg=None,
        verbose=True, writefile=''):

    cell_size_deg = np.rad2deg(aperture_size_kpc / gal_dist_Mpc / 1e3)

    low_res = aperture_size_kpc * u.kpc
    CO_high_res = np.array(CO_res_pc) * u.pc

    # initialize table
    if verbose:
        print("  Initializing data table")
    infile = get_data_path('ALMA:CO:tpeak', gal_name, low_res)
    if not infile.is_file():
        if verbose:
            print(f"No CO low resolution data found for {gal_name}")
            print("")
        return
    with fits.open(infile) as hdul:
        mt = TessellMegaTable(
            hdul[0].header, 
            cell_shape=aperture_shape,
            cell_size_deg=cell_size_deg,
            gal_ra_deg=gal_ra_deg, gal_dec_deg=gal_dec_deg)

    # add galactic radii and projected angles in table
    if verbose:
        print("  Calculating r_gal and phi_gal")
    radii, projang = deproject(
        center_ra=gal_ra_deg, center_dec=gal_dec_deg,
        incl=gal_incl_deg, pa=gal_posang_deg,
        ra=mt['RA'], dec=mt['DEC'])
    mt['r_gal_angl'] = (radii * u.deg).to('arcsec')
    mt['phi_gal'] = projang * u.deg
    # sort rows by galactic radii
    mt[:] = mt[np.argsort(mt['r_gal_angl'])]

    # add low resolution CO data in table
    # note: resampling 1kpc resolution maps as they are more complete
    if verbose:
        print("  Resampling low resolution CO data")
    infile = get_data_path('ALMA:CO:mom0:strict', gal_name, low_res)
    mt.resample_image(
        infile, suppress_error=True, fill_outside=np.nan,
        colname='I_CO21_1kpc_samp', unit=u.Unit('K km s-1'))
    if np.isfinite(mt['I_CO21_1kpc_samp']).sum() == 0:
        if verbose:
            print(f"No CO detection in any aperture -- "
                  f"skip {gal_name}")
            print("")
        return
    
    # add HI data in table
    if verbose:
        print("  Resampling HI data")
    infile = get_data_path('HI:mom0', gal_name, low_res)
    mt.resample_image(
        infile, suppress_error=True, fill_outside=np.nan,
        colname='I_21cm_1kpc_samp', unit=u.Unit('K km s-1'))
    infile = get_data_path('HI:mom0', gal_name)
    mt.resample_image(
        infile, suppress_error=True, fill_outside=np.nan,
        colname='I_21cm_nat_samp', unit=u.Unit('K km s-1'))
    mt.calc_image_stats(
        infile, suppress_error=True, stat_func=nanaverage,
        colname='I_21cm_nat_avg', unit=u.Unit('K km s-1'))
    # mask all values below 20 K km s-1 (for now)
    thres = 20 * u.Unit('K km s-1')
    for key in (
            'I_21cm_1kpc_samp', 'I_21cm_nat_samp', 'I_21cm_nat_avg'):
        mt._table[key][mt[key].quantity < thres] = 0

    # add S4G data in table
    if verbose:
        print("  Resampling S4G data")
    infile = get_data_path('S4G:ICA3p6um', gal_name, low_res)
    mt.resample_image(
        infile, suppress_error=True, fill_outside=np.nan,
        colname='I_3p6um_ICA_1kpc_samp', unit=u.Unit('MJy sr-1'))
    infile = get_data_path('S4G:3p6um', gal_name, low_res)
    mt.resample_image(
        infile, suppress_error=True, fill_outside=np.nan,
        colname='I_3p6um_raw_1kpc_samp', unit=u.Unit('MJy sr-1'))
    infile = get_data_path('S4G:ICA3p6um', gal_name)
    mt.calc_image_stats(
        infile, suppress_error=True, stat_func=nanaverage,
        colname='I_3p6um_ICA_nat_avg', unit=u.Unit('MJy sr-1'))
    infile = get_data_path('S4G:3p6um', gal_name)
    mt.calc_image_stats(
        infile, suppress_error=True, stat_func=nanaverage,
        colname='I_3p6um_raw_nat_avg', unit=u.Unit('MJy sr-1'))

    # add z0MGS data in table
    if verbose:
        print("  Resampling z0MGS data")
    infile = get_data_path('z0MGS:SFR:NUVW3', gal_name, low_res)
    mt.resample_image(
        infile, suppress_error=True, fill_outside=np.nan,
        colname='Sigma_SFR_NUVW3_1kpc_samp',
        unit=u.Unit('Msun kpc-2 yr-1'))
    infile = get_data_path('z0MGS:SFR:W3ONLY', gal_name, low_res)
    mt.resample_image(
        infile, suppress_error=True, fill_outside=np.nan,
        colname='Sigma_SFR_W3ONLY_1kpc_samp',
        unit=u.Unit('Msun kpc-2 yr-1'))
    infile = get_data_path('z0MGS:SFR:NUVW3', gal_name)
    mt.calc_image_stats(
        infile, suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_NUVW3_nat_avg',
        unit=u.Unit('Msun kpc-2 yr-1'))
    infile = get_data_path('z0MGS:SFR:FUVW4', gal_name)
    mt.calc_image_stats(
        infile, suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_FUVW4_nat_avg',
        unit=u.Unit('Msun kpc-2 yr-1'))
    infile = get_data_path('z0MGS:SFR:W3ONLY', gal_name)
    mt.calc_image_stats(
        infile, suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_W3ONLY_nat_avg',
        unit=u.Unit('Msun kpc-2 yr-1'))
    infile = get_data_path('z0MGS:SFR:W4ONLY', gal_name)
    mt.calc_image_stats(
        infile, suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_W4ONLY_nat_avg',
        unit=u.Unit('Msun kpc-2 yr-1'))

    # add environmental fraction (CO flux-weighted) in table
    if verbose:
        print("  Calculating (CO flux-weighted) "
              "environmental fraction")
    res = CO_high_res[-1]
    wtfile = get_data_path('ALMA:CO:mom0:strict', gal_name, res)
    for reg in env_regions:
        if verbose:
            print(f"    > fraction of {reg}")
        envfile = get_data_path('S4G:env_mask:'+reg, gal_name)
        mt.calc_env_frac(
            envfile, wtfile, suppress_error=True,
            colname='frac_'+reg)

    # add statistics of high resolution CO data in table
    if verbose:
        print("  Calculating statistics of high resolution CO data")
    for res in CO_high_res:
        if verbose:
            print(f"    @ {res.to('pc').value:.0f}pc resolution")
        # search for "parts" if the full mosaic is not available
        # at the requested linear resolution
        for part_str in ['', '_1', '_2', '_3']:
            bm0file = get_data_path(
                'ALMA:CO:mom0:broad', gal_name + part_str, res)
            sm0file = get_data_path(
                'ALMA:CO:mom0:strict', gal_name + part_str, res)
            sewfile = get_data_path(
                'ALMA:CO:ew:strict', gal_name + part_str, res)
            if bm0file.is_file():
                break
        mt.calc_CO_stats(
            bm0file, sm0file, sewfile, res, suppress_error=True)

    # add statistics of CPROPS clouds in table
    if verbose:
        print("  Calculating statistics of CPROPS clouds")
    for res in CO_high_res:
        if verbose:
            print(f"    @ {res.to('pc').value:.0f}pc resolution")
        cpropsfile = get_data_path('ALMA:CPROPS', gal_name, res)
        mt.calc_cprops_stats(
            cpropsfile, res, suppress_error=True)

    # write table to disk
    if writefile:
        if verbose:
            print("  Writing table to disk")
        mt.write(writefile, overwrite=True)
        return writefile
    else:
        return mt
    

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
        mtfile, CO_res_pc=[],
        gal_name=None, gal_ra_deg=None, gal_dec_deg=None,
        gal_incl_deg=None, gal_posang_deg=None,
        gal_dist_Mpc=None, gal_logMstar=None,
        gal_Rstar_arcsec=None, gal_Reff_arcsec=None,
        append_raw_data=False, note='', version=0, writefile=''):

    mt = TessellMegaTable.read(mtfile)
    
    # mask rows where CO(2-1) intensity at 1kpc res. is not positive
    mt.clean(keep_positive='I_CO21_1kpc_samp')

    CO_high_res = np.array(CO_res_pc) * u.pc

    # initiate new table
    t_phys = Table()
    t_phys.meta['NAME'] = str(gal_name)
    t_phys.meta['RA_DEG'] = gal_ra_deg
    t_phys.meta['DEC_DEG'] = gal_dec_deg
    t_phys.meta['INCL_DEG'] = gal_incl_deg
    t_phys.meta['PA_DEG'] = gal_posang_deg
    t_phys.meta['DIST_MPC'] = gal_dist_Mpc
    t_phys.meta['LOGMSTAR'] = gal_logMstar
    t_phys.meta['RDISK_AS'] = gal_Rstar_arcsec
    t_phys.meta['REFF_AS'] = gal_Reff_arcsec
    t_phys.meta['TBL_NOTE'] = note
    t_phys.meta['VERSION'] = f"ver{version:.1f}"

    # galaxy global parameters
    gal_cosi = np.cos(np.deg2rad(gal_incl_deg))
    gal_dist = gal_dist_Mpc * u.Mpc
    gal_Mstar = 10**gal_logMstar * u.Msun
    gal_Rstar = (np.deg2rad(gal_Rstar_arcsec/3600)*gal_dist).to('kpc')
    gal_Reff = (np.deg2rad(gal_Reff_arcsec/3600)*gal_dist).to('kpc')

    # coordinates
    t_phys['RA'] = mt['RA'].quantity.to('deg')
    t_phys['DEC'] = mt['DEC'].quantity.to('deg')
    t_phys['r_gal'] = r_gal = (
        mt['r_gal_angl'].quantity.to('rad').value *
        gal_dist).to('kpc')
    t_phys['phi_gal'] = mt['phi_gal'].quantity.to('deg')

    # metallicity
    t_phys['log(O/H)_PP04'] = predict_metallicity(
        gal_Mstar, calibrator='O3N2(PP04)', MZR='Sanchez+19',
        Rgal=r_gal, Re=gal_Reff, gradient='Sanchez+14')
    t_phys['Zprime'] = Zprime = 10**(
        t_phys['log(O/H)_PP04'] - 8.69)

    # SFR surface density
    for key in (
            'Sigma_SFR_NUVW3_1kpc_samp',
            'Sigma_SFR_W3ONLY_1kpc_samp'):
        newkey = key[:-10]
        t_phys[newkey] = mt[key].quantity.to('Msun kpc-2 yr-1')
        if (np.isfinite(t_phys[newkey]).sum() != 0 and
            'Sigma_SFR' not in t_phys.colnames):
            t_phys['Sigma_SFR'] = t_phys[newkey].quantity
    if 'Sigma_SFR' not in t_phys.colnames:
        t_phys['Sigma_SFR'] = np.nan * u.Unit('Msun kpc-2 yr-1')
    # for key in (
    #         'Sigma_SFR_NUVW3_nat_avg', 'Sigma_SFR_W3ONLY_nat_avg',
    #         'Sigma_SFR_FUVW4_nat_avg', 'Sigma_SFR_W4ONLY_nat_avg'):
    #     t_phys[key] = mt[key].quantity.to('Msun kpc-2 yr-1')

    # stellar mass surface densities
    alpha3p6um = get_alpha3p6um(ref='MS14')
    for key in (
            'I_3p6um_ICA_1kpc_samp',
            'I_3p6um_raw_1kpc_samp'):
        newkey = 'Sigma_star'+key[-12:-8]
        t_phys[newkey] = (
            mt[key].quantity * gal_cosi * alpha3p6um
        ).to('Msun/pc^2')
        if (np.isfinite(t_phys[newkey]).sum() != 0 and
            'Sigma_star' not in t_phys.colnames):
            t_phys['Sigma_star'] = t_phys[newkey].quantity
    if 'Sigma_star' not in t_phys.colnames:
        t_phys['Sigma_star'] = np.nan * u.Unit('Msun pc-2')
    # for key in ('I_3p6um_ICA_nat_avg', 'I_3p6um_raw_nat_avg'):
    #     newkey = 'Sigma_star'+key[-12:]
    #     t_phys[newkey] = (
    #         mt[key].quantity * gal_cosi * alpha3p6um
    #     ).to('Msun/pc^2')
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
        mt['I_21cm_1kpc_samp'].quantity * gal_cosi * alpha21cm
    ).to('Msun pc-2')
    if np.isfinite(t_phys['Sigma_atom']).sum() == 0:
        t_phys['Sigma_atom'] = (
            mt['I_21cm_nat_samp'].quantity * gal_cosi * alpha21cm
        ).to('Msun pc-2')
    Sigma_atom = t_phys['Sigma_atom'].quantity
    # t_phys['Sigma_atom'] = (
    #     mt['I_21cm_nat_avg'].quantity * gal_cosi * alpha21cm
    # ).to('Msun pc-2')
    Sigma_atom = t_phys['Sigma_atom'].quantity

    # H2 surface density (low resolution)
    R21 = get_R21()
    rstr_fid = f"{CO_high_res[-1].to('pc').value:.0f}pc"  # 150 pc
    ICO10kpc = mt['I_CO21_1kpc_samp'].quantity * gal_cosi / R21
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
        mt['I_CO21_1kpc_samp'].quantity * gal_cosi *
        alphaCO21).to('Msun pc-2')
    Sigma_mol = t_phys['Sigma_mol'].quantity

    # CO contribution by environments
    for reg in regions:
        t_phys[f"frac_{reg}"] = mt[f"frac_{reg}"].quantity
    # manually flag the aperture on the galactic center as "bulge"
    # if no environment masks are available for this target
    if ((np.isfinite(t_phys['frac_bulge']).sum() == 0) and
        (np.isfinite(t_phys['frac_bars']).sum() == 0) and
        (np.isfinite(t_phys['frac_disk']).sum() == 0)):
        t_phys['frac_bulge'][np.nonzero(r_gal == 0)[0]] = 1.

    # CO map statistics
    for res in CO_high_res:
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
    for res in CO_high_res:
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
    res_fid = CO_high_res[-2].to('pc')  # 120 pc
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
    t_phys["W_atom"] = (
        (np.pi * const.G / 2 * Sigma_atom**2 +
         np.pi * const.G * Sigma_atom * Sigma_mol +
         Sigma_atom * vdisp_atom_z *
         np.sqrt(2*const.G*rho_star)) /
        const.k_B).to('K cm-3')
    t_phys["P_DE_smooth"] = (
        (np.pi * const.G / 2 * Sigma_mol**2 +
         np.pi * const.G * Sigma_mol * rho_star * res_fid/2) /
        const.k_B).to('K cm-3') + t_phys['W_atom'].quantity
    for res in CO_high_res:
        R_cloud = res / 2
        rstr = f"{res.to('pc').value:.0f}pc"
        t_phys[f"A<W_cloud_self_pix_{rstr}>"] = (
            3/8 * np.pi * const.G *
            mt[f"A<I^2_CO21_{rstr}>"].quantity * alphaCO21**2 /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_self_pix_{rstr}>"] = (
            3/8 * np.pi * const.G *
            mt[f"F<I^2_CO21_{rstr}>"].quantity * alphaCO21**2 /
            const.k_B).to('K cm-3')
        t_phys[f"A<W_cloud_mol_pix_{rstr}>"] = (
            np.pi * const.G / 2 * Sigma_mol *
            t_phys[f"A<Sigma_mol_pix_{rstr}>"] /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_mol_pix_{rstr}>"] = (
            np.pi * const.G / 2 * Sigma_mol *
            t_phys[f"F<Sigma_mol_pix_{rstr}>"] /
            const.k_B).to('K cm-3')
        t_phys[f"A<W_cloud_star_pix_{rstr}>"] = (
            3/2 * np.pi * const.G * rho_star * R_cloud *
            t_phys[f"A<Sigma_mol_pix_{rstr}>"] /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_star_pix_{rstr}>"] = (
            3/2 * np.pi * const.G * rho_star * R_cloud *
            t_phys[f"F<Sigma_mol_pix_{rstr}>"] /
            const.k_B).to('K cm-3')
        t_phys[f"A<P_DE_pix_{rstr}>"] = (
            t_phys[f"A<W_cloud_self_pix_{rstr}>"].quantity +
            t_phys[f"A<W_cloud_mol_pix_{rstr}>"].quantity +
            t_phys[f"A<W_cloud_star_pix_{rstr}>"].quantity +
            t_phys["W_atom"].quantity)
        t_phys[f"F<P_DE_pix_{rstr}>"] = (
            t_phys[f"F<W_cloud_self_pix_{rstr}>"].quantity +
            t_phys[f"F<W_cloud_mol_pix_{rstr}>"].quantity +
            t_phys[f"F<W_cloud_star_pix_{rstr}>"].quantity +
            t_phys["W_atom"].quantity)
    for res in CO_high_res:
        rstr = f"{res.to('pc').value:.0f}pc"
        t_phys[f"U<W_cloud_self_CPROPS_{rstr}>"] = (
            3 * const.G / 8 / np.pi *
            mt[f"U<F^2/R^4_CPROPS_{rstr}>"].quantity *
            alphaCO21**2 / R_factor**4 /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_self_CPROPS_{rstr}>"] = (
            3 * const.G / 8 / np.pi *
            mt[f"F<F^2/R^4_CPROPS_{rstr}>"].quantity *
            alphaCO21**2 / R_factor**4 /
            const.k_B).to('K cm-3')
        t_phys[f"U<W_cloud_mol_CPROPS_{rstr}>"] = (
            np.pi * const.G / 2 * Sigma_mol *
            t_phys[f"U<Sigma_mol_CPROPS_{rstr}>"].quantity /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_mol_CPROPS_{rstr}>"] = (
            np.pi * const.G / 2 * Sigma_mol *
            t_phys[f"F<Sigma_mol_CPROPS_{rstr}>"].quantity /
            const.k_B).to('K cm-3')
        t_phys[f"U<W_cloud_star_CPROPS_{rstr}>"] = (
            3/2 * const.G * rho_star *
            mt[f"U<F/R_CPROPS_{rstr}>"].quantity *
            alphaCO21 / R_factor * (gal_dist / u.rad) /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_star_CPROPS_{rstr}>"] = (
            3/2 * const.G * rho_star *
            mt[f"F<F/R_CPROPS_{rstr}>"].quantity *
            alphaCO21 / R_factor * (gal_dist / u.rad) /
            const.k_B).to('K cm-3')
        t_phys[f"U<P_DE_CPROPS_{rstr}>"] = (
            t_phys[f"U<W_cloud_self_CPROPS_{rstr}>"].quantity +
            t_phys[f"U<W_cloud_mol_CPROPS_{rstr}>"].quantity +
            t_phys[f"U<W_cloud_star_CPROPS_{rstr}>"].quantity +
            t_phys["W_atom"].quantity)
        t_phys[f"F<P_DE_CPROPS_{rstr}>"] = (
            t_phys[f"F<W_cloud_self_CPROPS_{rstr}>"].quantity +
            t_phys[f"F<W_cloud_mol_CPROPS_{rstr}>"].quantity +
            t_phys[f"F<W_cloud_star_CPROPS_{rstr}>"].quantity +
            t_phys["W_atom"].quantity)

    if append_raw_data:
        for key in mt.colnames:
            if key not in t_phys.colnames:
                t_phys[key] = mt[key].quantity

    if writefile:
        t_phys.write(writefile, overwrite=True)
        return writefile
    else:
        return t_phys


######################################################################
######################################################################


if __name__ == '__main__':

    # ----------------------------------------------------------------
    # parameters that specify how to construct the mega-tables
    # ----------------------------------------------------------------

    # aperture (linear) size
    aperture_size = 1 * u.kpc

    # aperture shape
    aperture_shape = 'hexagon'

    # (linear) resolutions of the PHANGS-ALMA data
    CO_high_res = np.array([60, 90, 120, 150]) * u.pc

    # list of morphological regions in environmental masks
    regions = ('disk', 'bulge', 'bars', 'rings', 'lenses', 'sp_arms')

    # ----------------------------------------------------------------
    # pipeline main body starts from here
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

    # read PHANGS sample table
    catalog = Table.read(get_data_path('sample_table'))
    # only keep targets with the 'ALMA' tag
    catalog = catalog[catalog['HAS_ALMA'] == 1]
    # loop through sample table
    for row in catalog:

        # galaxy parameters
        name = row['NAME'].strip()
        dist = row['DIST'] * u.Mpc
        ra = row['ORIENT_RA'] * u.deg
        dec = row['ORIENT_DEC'] * u.deg
        incl = row['ORIENT_INCL'] * u.deg
        posang = row['ORIENT_POSANG'] * u.deg
        logMstar = row['MSTAR_LOGMSTAR']
        Rstar = row['SIZE_S4G_RSTAR'] * u.arcsec
        Reff = row['SIZE_W1_R50'] * u.arcsec

        # skip targets with bad geometrical information
        if not ((incl >= 0*u.deg) and (incl < 90*u.deg) and
                np.isfinite(posang)):
            continue
        # skip targets with bad distance
        if not (dist > 0):
            continue

        print(f"Processing data for {name}")

        # ------------------------------------------------------------
        # generate raw measurement table
        # ------------------------------------------------------------

        mtfile = (
            workdir /
            f"{name}_{aperture_shape}_"
            f"{aperture_size.to('kpc').value:.0f}kpc.ecsv")
        if not mtfile.is_file():
            print(f"Constructing raw measurement table for {name}")
            gen_raw_measurement_table(
                name, aperture_shape, aperture_size.to('kpc').value,
                CO_res_pc=CO_high_res.to('pc').value,
                env_regions=regions,
                gal_ra_deg=ra.value, gal_dec_deg=dec.value,
                gal_incl_deg=incl.value, gal_posang_deg=posang.value,
                gal_dist_Mpc=dist.value,
                verbose=True, writefile=mtfile)

        # ------------------------------------------------------------
        # convert raw measurements to physical properties
        # ------------------------------------------------------------

        physfile = (
            workdir /
            f"{name}_{aperture_shape}_"
            f"{aperture_size.to('kpc').value:.0f}kpc_phys.ecsv")
        if (mtfile.is_file() and not physfile.is_file()):
            print(f"Constructing physical property table for {name}")
            gen_phys_props_table(
                mtfile, CO_res_pc=CO_high_res.to('pc').value,
                gal_name=name,
                gal_ra_deg=ra.value, gal_dec_deg=dec.value,
                gal_incl_deg=incl.value, gal_posang_deg=posang.value,
                gal_dist_Mpc=dist.value, gal_logMstar=logMstar,
                gal_Rstar_arcsec=Rstar.value,
                gal_Reff_arcsec=Reff.value,
                append_raw_data=False,
                note='PHANGS sample table v1p4 (but distances=v1p3)',
                version=0.9, writefile=physfile)
        print("")

        # ------------------------------------------------------------

        print(f"Finished processing data for {name}!")
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
