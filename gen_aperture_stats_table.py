import os
import sys
from pathlib import Path
import warnings
import numpy as np
from astropy import units as u, constants as const
from astropy.table import Table
from astropy.io import fits
from reproject import reproject_interp
from AlmaTools.XCO import predict_metallicity, predict_alphaCO10
from mega_table.table import VoronoiTessTable
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
        return PHANGSdir / 'sample_v1p3.ecsv'

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
        basedir = PHANGSdir / 'HI'
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


def add_resampled_image_to_table(
    t, infile, colname='new_col', unit=u.Unit(''), **kwargs):

    if not infile.is_file():
        t[colname] = np.full(len(t), np.nan) * unit
        return

    t.resample_image(infile, colname=colname, unit='header', **kwargs)

    return


# --------------------------------------------------------------------


def add_env_frac_to_table(
    t, envfile, wtfile, colname='new_col', **kwargs):

    if not (envfile.is_file() and wtfile.is_file()):
        t[colname] = np.full(len(t), np.nan)
        return

    with fits.open(wtfile) as hdul:
        wtmap = hdul[0].data.copy()
        wtmap[~np.isfinite(wtmap) | (wtmap < 0)] = 0
        wthdr = hdul[0].header.copy()
    with fits.open(envfile) as hdul:
        envmap, footprint = reproject_interp(
            hdul[0], wthdr, order=0)
    envmap[~footprint.astype('?')] = 0
    envbimap = (envmap > 0).astype('float')
    t.calc_image_stats(
        envbimap, header=wthdr, stat_func=nanaverage, weight=wtmap,
        colname=colname, **kwargs)

    return


# --------------------------------------------------------------------


def add_CO_stats_to_table(
    t, bm0file, sm0file, sewfile, res, **kwargs):

    rstr = f"{res.to('pc').value:.0f}pc"
    cols = [
        # sums
        (f"Area_CO21_total_{rstr}", 'arcsec2'),
        (f"Area_CO21_broad_{rstr}", 'arcsec2'),
        (f"Area_CO21_strict_{rstr}", 'arcsec2'),
        (f"Flux_CO21_broad_{rstr}", 'K km s-1 arcsec2'),
        (f"Flux_CO21_strict_{rstr}", 'K km s-1 arcsec2'),
        # area-weighted averages
        (f"A<I_CO21_{rstr}>", 'K km s-1'),
        (f"A<I^2_CO21_{rstr}>", 'K2 km2 s-2'),
        (f"A<sigv_CO21_{rstr}>", 'km s-1'),
        (f"A<sigv^2_CO21_{rstr}>", 'km2 s-2'),
        (f"A<I*sigv^2_CO21_{rstr}>", 'K km3 s-3'),
        (f"A<sigv^2/I_CO21_{rstr}>", 'km s-1 K-1'),
        # CO flux-weighted averages
        (f"F<I_CO21_{rstr}>", 'K km s-1'),
        (f"F<I^2_CO21_{rstr}>", 'K2 km2 s-2'),
        (f"F<sigv_CO21_{rstr}>", 'km s-1'),
        (f"F<sigv^2_CO21_{rstr}>", 'km2 s-2'),
        (f"F<I*sigv^2_CO21_{rstr}>", 'K km3 s-3'),
        (f"F<sigv^2/I_CO21_{rstr}>", 'km s-1 K-1'),
        ]

    # if no data file found: add placeholder (empty) columns
    if not (bm0file.is_file() and sm0file.is_file() and
            sewfile.is_file()):
        for col, unit in cols:
            t[col] = np.full(len(t), np.nan) * u.Unit(unit)
        return

    # read files
    with fits.open(bm0file) as hdul:
        bm0_map = hdul[0].data.copy()
        hdr = hdul[0].header.copy()
    with fits.open(sm0file) as hdul:
        sm0_map = hdul[0].data.copy()
        if sm0_map.shape != bm0_map.shape:
            raise ValueError("Input maps have inconsistent shape")
    with fits.open(sewfile) as hdul:
        sew_map = hdul[0].data.copy()
        if sew_map.shape != bm0_map.shape:
            raise ValueError("Input maps have inconsistent shape")
    bm0_map[~np.isfinite(bm0_map)] = 0
    nan_map = np.ones_like(bm0_map).astype('float')
    nan_map[~np.isfinite(sm0_map) | (sm0_map <= 0) |
           ~np.isfinite(sew_map) | (sew_map <= 0)] = np.nan
    sm0_map[np.isnan(nan_map)] = 0
    sew_map[np.isnan(nan_map)] = 0

    # pixel size (in arcsec^2)
    pixsz = np.abs(
        hdr['CDELT1'] * hdr['CDELT2'] * u.deg**2
        ).to('arcsec2').value

    # maps corresponding to each column
    maps = [
        # sums
        np.ones_like(bm0_map).astype('float')*pixsz,
        (bm0_map != 0).astype('float')*pixsz,
        (sm0_map > 0).astype('float')*pixsz,
        bm0_map*pixsz,
        sm0_map*pixsz,
        # area-weighted averages (among regions w/ CO detection)
        sm0_map * nan_map,
        sm0_map**2 * nan_map,
        sew_map * nan_map,
        sew_map**2 * nan_map,
        sm0_map*sew_map**2 * nan_map,
        sew_map**2/sm0_map * nan_map,
        # CO flux-weighted averages
        sm0_map,
        sm0_map**2,
        sew_map,
        sew_map**2,
        sm0_map*sew_map**2,
        sew_map**2/sm0_map,
    ]

    # calculate statistics and add into table
    for (col, unit), map in zip(cols, maps):
        if col[:2] == 'A<':
            # area-weighted average
            t.calc_image_stats(
                map, header=hdr, stat_func=nanaverage, weight=None,
                colname=col, unit=unit, **kwargs)
        elif col[:2] == 'F<':
            # CO flux-weighted average
            t.calc_image_stats(
                map, header=hdr, stat_func=nanaverage, weight=sm0_map,
                colname=col, unit=unit, **kwargs)
        else:
            # sum
            t.calc_image_stats(
                map, header=hdr, stat_func=np.nansum,
                colname=col, unit=unit, **kwargs)

    return


# --------------------------------------------------------------------


def add_cprops_stats_to_table(
    t, cpropsfile, res, **kwargs):

    rstr = f"{res.to('pc').value:.0f}pc"
    cols = [
        # sums
        (f"Nobj_CPROPS_{rstr}", ''),
        (f"Flux_CPROPS_total_{rstr}", 'K km s-1 arcsec2'),
        # uniformly weighted averages
        (f"U<F_CPROPS_{rstr}>", 'K km s-1 arcsec2'),
        (f"U<R_CPROPS_{rstr}>", 'arcsec'),
        (f"U<F/R^2_CPROPS_{rstr}>", 'K km s-1'),
        (f"U<F^2/R^4_CPROPS_{rstr}>", 'K2 km2 s-2'),
        (f"U<F/R_CPROPS_{rstr}>", 'K km s-1 arcsec'),
        (f"U<sigv_CPROPS_{rstr}>", 'km s-1'),
        (f"U<sigv^2_CPROPS_{rstr}>", 'km2 s-2'),
        (f"U<F*sigv^2/R^3_CPROPS_{rstr}>", 'K km3 s-3 arcsec-1'),
        (f"U<R*sigv^2/F_CPROPS_{rstr}>", 'km s-1 K-1 arcsec-1'),
        # CO flux-weighted averages
        (f"F<F_CPROPS_{rstr}>", 'K km s-1 arcsec2'),
        (f"F<R_CPROPS_{rstr}>", 'arcsec'),
        (f"F<F/R^2_CPROPS_{rstr}>", 'K km s-1'),
        (f"F<F^2/R^4_CPROPS_{rstr}>", 'K2 km2 s-2'),
        (f"F<F/R_CPROPS_{rstr}>", 'K km s-1 arcsec'),
        (f"F<sigv_CPROPS_{rstr}>", 'km s-1'),
        (f"F<sigv^2_CPROPS_{rstr}>", 'km2 s-2'),
        (f"F<F*sigv^2/R^3_CPROPS_{rstr}>", 'K km3 s-3 arcsec-1'),
        (f"F<R*sigv^2/F_CPROPS_{rstr}>", 'km s-1 K-1 arcsec-1'),
    ]

    # if no CPROPS file found: add placeholder (empty) columns
    if not cpropsfile.is_file():
        for col, unit in cols:
            t[col] = np.full(len(t), np.nan) * u.Unit(unit)
        return

    # read CPROPS file
    try:
        t_cat = Table.read(cpropsfile)
    except ValueError as e:
        print(e)
        return
    ra_cat = t_cat['XCTR_DEG'].quantity.value
    dec_cat = t_cat['YCTR_DEG'].quantity.value
    flux_cat = (
        t_cat['FLUX_KKMS_PC2'] / t_cat['DISTANCE_PC']**2 *
        u.Unit('K km s-1 sr')).to('K km s-1 arcsec2').value
    sigv_cat = t_cat['SIGV_KMS'].quantity.value
    rad_cat = (  # expressed in Solomon+87 convention
        t_cat['RAD_PC'] / t_cat['DISTANCE_PC'] *
        u.rad).to('arcsec').value
    wt_arr = flux_cat.copy()
    wt_arr[~np.isfinite(rad_cat)] = 0
    flux_cat[~np.isfinite(rad_cat)] = np.nan
    sigv_cat[~np.isfinite(rad_cat)] = np.nan
    rad_cat[~np.isfinite(rad_cat)] = np.nan

    # entries corresponding to each column
    entries = [
        # sums
        np.isfinite(flux_cat).astype('int'),
        flux_cat,
        # uniformly weighted averages
        flux_cat,
        rad_cat,
        flux_cat/rad_cat**2,
        flux_cat**2/rad_cat**4,
        flux_cat/rad_cat,
        sigv_cat,
        sigv_cat**2,
        flux_cat*sigv_cat**2/rad_cat**3,
        rad_cat*sigv_cat**2/flux_cat,
        # CO flux-weighted averages
        flux_cat,
        rad_cat,
        flux_cat/rad_cat**2,
        flux_cat**2/rad_cat**4,
        flux_cat/rad_cat,
        sigv_cat,
        sigv_cat**2,
        flux_cat*sigv_cat**2/rad_cat**3,
        rad_cat*sigv_cat**2/flux_cat,
    ]

    # calculate statistics and add into table
    for (col, unit), entry in zip(cols, entries):
        if col[:2] == 'U<':
            # area-weighted average
            t.calc_catalog_stats(
                entry, ra_cat, dec_cat,
                stat_func=nanaverage, weight=None,
                colname=col, unit=unit, **kwargs)
        elif col[:2] == 'F<':
            # CO flux-weighted average
            t.calc_catalog_stats(
                entry, ra_cat, dec_cat,
                stat_func=nanaverage, weight=wt_arr,
                colname=col, unit=unit, **kwargs)
        else:
            # sum
            t.calc_catalog_stats(
                entry, ra_cat, dec_cat, stat_func=np.nansum,
                colname=col, unit=unit, **kwargs)

    return


# --------------------------------------------------------------------


def get_R21():
    return 0.7  # CO(2-1)/CO(1-0) ratio


def get_alpha21cm(include_He=True):
    if include_He:  # include the extra 35% mass of Helium
        alpha21cm =  1.97e-2 * u.Msun/u.pc**2/(u.K*u.km/u.s)
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


def gen_phys_props_table(vtt, params, append_raw_data=False):

    # initiate new table
    t_phys = Table()
    for key in params:
        t_phys.meta[key] = params[key]

    # galaxy global parameters
    gal_cosi = np.cos(np.deg2rad(params['INCL']))
    gal_dist = params['DIST'] * u.Mpc
    gal_Mstar = 10**params['LOGMSTAR'] * u.Msun
    gal_Rstar = params['RSTAR'] * u.kpc
    gal_Reff = params['REFF_W1'] * u.kpc

    # coordinates
    t_phys['RA'] = vtt['RA'].quantity.to('deg')
    t_phys['DEC'] = vtt['DEC'].quantity.to('deg')
    t_phys['r_gal'] = r_gal = (
        vtt['rang_gal'].quantity.to('rad').value * gal_dist).to('kpc')
    t_phys['phi_gal'] = vtt['phi_gal'].quantity.to('deg')

    # metallicity
    t_phys['log(O/H)_PP04'] = predict_metallicity(
        gal_Mstar, calibrator='O3N2(PP04)', MZR='Sanchez+19',
        Rgal=r_gal, Re=gal_Reff, gradient='Sanchez+14')
    t_phys['Zprime'] = Zprime = 10**(t_phys['log(O/H)_PP04'] - 8.69)

    # SFR surface density
    for key in ('Sigma_SFR_NUVW3', 'Sigma_SFR_W3ONLY'):
        t_phys[key] = vtt[key].quantity.to('Msun kpc-2 yr-1')
        if (np.isfinite(t_phys[key]) != 0 and
            'Sigma_SFR' not in t_phys.colnames):
            t_phys['Sigma_SFR'] = t_phys[key].quantity
    if 'Sigma_SFR' not in t_phys.colnames:
        t_phys['Sigma_SFR'] = np.nan * u.Unit('Msun kpc-2 yr-1')
    Sigma_SFR = t_phys['Sigma_SFR'].quantity

    # stellar mass surface densities
    alpha3p6um = get_alpha3p6um(ref='MS14')
    for key in ('I_3p6um_ICA', 'I_3p6um_raw'):
        newkey = 'Sigma_star'+key[-4:]
        t_phys[newkey] = (
            vtt[key].quantity * cosi * alpha3p6um).to('Msun/pc^2')
        if (np.isfinite(t_phys[newkey]) != 0 and
            'Sigma_star' not in t_phys.colnames):
            t_phys['Sigma_star'] = t_phys[newkey].quantity
    if 'Sigma_star' not in t_phys.colnames:
        t_phys['Sigma_star'] = np.nan * u.Unit('Msun pc-2')
    Sigma_star = t_phys['Sigma_star'].quantity

    # stellar mass volume density near disk mid-plane
    t_phys['rho_star'] = rho_star = (
        t_phys['Sigma_star'] / 4 /
        get_h_star(gal_Rstar, diskshape='flat')
    ).to('Msun pc-3')
    t_phys['rho_star_flared'] = rho_star_flared = (
        t_phys['Sigma_star'] / 4 /
        get_h_star(gal_Rstar, diskshape='flared', Rgal=r_gal)
    ).to('Msun pc-3')

    # HI mass surface density
    alpha21cm = get_alpha21cm(include_He=True)
    t_phys['Sigma_atom'] = (
        vtt['I_21cm'].quantity * cosi * alpha21cm).to('Msun pc-2'))
    if np.isfinite(t_phys['Sigma_atom']) == 0:
        t_phys['Sigma_atom'] = (
            vtt['I_21cm_raw'].quantity * cosi * alpha21cm
        ).to('Msun pc-2'))
    Sigma_atom = t_phys['Sigma_atom'].quantity

    # H2 surface density (low resolution)
    R21 = get_R21()
    rstr_fid = f"{res_pcs[-1].to('pc').value:.0f}pc"  # 150 pc
    ICO10kpc = vtt['I_CO21'].quantity * cosi / R21
    ICO10GMC = vtt[f"F<I_CO21_{rstr_fid}>"].quantity / R21
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
        vtt['I_CO21'].quantity * cosi * alphaCO21).to('Msun pc-2')
    Sigma_mol = t_phys['Sigma_mol'].quantity

    # CO contribution by environments
    for reg in regions:
        t_phys[f"frac_{reg}"] = vtt[f"frac_{reg}"].quantity
    # if no environment masks are available:
    # manually flag all data points (for now)  # <--------------------
    if ((np.isfinite(t_phys['frac_bulge']).sum() == 0) and
        (np.isfinite(t_phys['frac_bars']).sum() == 0)):
        t_phys['frac_bulge'] = 1.

    # CO map statistics
    for res_pc in res_pcs:
        R_cloud = res_pc / 2
        rstr = f"{res_pc.to('pc').value:.0f}pc"
        t_phys[f"fracA_CO21_{rstr}"] = (
            vtt[f"Area_CO21_strict_{rstr}"].quantity /
            vtt[f"Area_CO21_total_{rstr}"].quantity).to('')
        t_phys[f"fracF_CO21_{rstr}"] = (
            vtt[f"Flux_CO21_strict_{rstr}"].quantity /
            vtt[f"Flux_CO21_broad_{rstr}"].quantity).to('')
        t_phys[f"clumping_CO21_{rstr}"] = (
            vtt[f"F<I_CO21_{rstr}>"].quantity /
            vtt[f"A<I_CO21_{rstr}>"].quantity).to('')
        t_phys[f"A<Sigma_mol_pix_{rstr}>"] = (
            vtt[f"A<I_CO21_{rstr}>"].quantity *
            alphaCO21).to('Msun pc-2')
        t_phys[f"F<Sigma_mol_pix_{rstr}>"] = (
            vtt[f"F<I_CO21_{rstr}>"].quantity *
            alphaCO21).to('Msun pc-2')
        t_phys[f"A<vdisp_mol_pix_{rstr}>"] = (
            vtt[f"A<sigv_CO21_{rstr}>"].quantity).to('km s-1')
        t_phys[f"F<vdisp_mol_pix_{rstr}>"] = (
            vtt[f"F<sigv_CO21_{rstr}>"].quantity).to('km s-1')
        t_phys[f"A<P_turb_pix_{rstr}>"] = (  # Eq.4 in Sun+20
            (3/2) * vtt[f"A<I*sigv^2_CO21_{rstr}>"].quantity /
            (2*R_cloud) * alphaCO21 / const.k_B).to('K cm-3')
        t_phys[f"F<P_turb_pix_{rstr}>"] = (  # Eq.4 in Sun+20
            (3/2) * vtt[f"F<I*sigv^2_CO21_{rstr}>"].quantity /
            (2*R_cloud) * alphaCO21 / const.k_B).to('K cm-3')
        t_phys[f"A<alpha_vir_pix_{rstr}>"] = (  # Eq.13 in Sun+18
            5 * np.log(2) / (10/9 * np.pi * const.G) *
            vtt[f"A<sigv^2/I_CO21_{rstr}>"].quantity / R_cloud /
            alphaCO21).to('')
        t_phys[f"F<alpha_vir_pix_{rstr}>"] = (  # Eq.13 in Sun+18
            5 * np.log(2) / (10/9 * np.pi * const.G) *
            vtt[f"F<sigv^2/I_CO21_{rstr}>"].quantity / R_cloud /
            alphaCO21).to('')

    # CPROPS cloud statistics
    R_factor = np.sqrt(5) / 1.91  # Rosolowsky&Leroy06: Section 3.1
    for res_pc in res_pcs:
        rstr = f"{res_pc.to('pc').value:.0f}pc"
        t_phys[f"Nobj_CPROPS_{rstr}"] = (
            vtt[f"Nobj_CPROPS_{rstr}"].quantity).to('')
        t_phys[f"fracF_CPROPS_{rstr}"] = (
            vtt[f"Flux_CPROPS_total_{rstr}"].quantity /
            vtt[f"Flux_CO21_broad_{rstr}"].quantity).to('')
        t_phys[f"U<M_mol_CPROPS_{rstr}>"] = (  # F [=] K*km/s arcsec2
            vtt[f"U<F_CPROPS_{rstr}>"].quantity * alphaCO21 *
            gal_dist**2 / u.sr).to('Msun')
        t_phys[f"F<M_mol_CPROPS_{rstr}>"] = (  # F [=] K*km/s arcsec2
            vtt[f"F<F_CPROPS_{rstr}>"].quantity * alphaCO21 *
            gal_dist**2 / u.sr).to('Msun')
        t_phys[f"U<R_CPROPS_{rstr}>"] = (  # R [=] arcsec
            vtt[f"U<R_CPROPS_{rstr}>"].quantity *
            gal_dist / u.rad).to('pc')
        t_phys[f"F<R_CPROPS_{rstr}>"] = (  # R [=] arcsec
            vtt[f"F<R_CPROPS_{rstr}>"].quantity *
            gal_dist / u.rad).to('pc')
        t_phys[f"U<Sigma_mol_CPROPS_{rstr}>"] = (
            vtt[f"U<F/R^2_CPROPS_{rstr}>"].quantity * alphaCO21 /
            (np.pi*R_factor**2)).to('Msun pc-2')
        t_phys[f"F<Sigma_mol_CPROPS_{rstr}>"] = (
            vtt[f"F<F/R^2_CPROPS_{rstr}>"].quantity * alphaCO21 /
            (np.pi*R_factor**2)).to('Msun pc-2')
        t_phys[f"U<vdisp_mol_CPROPS_{rstr}>"] = (
            vtt[f"U<sigv_CPROPS_{rstr}>"].quantity).to('km s-1')
        t_phys[f"F<vdisp_mol_CPROPS_{rstr}>"] = (
            vtt[f"F<sigv_CPROPS_{rstr}>"].quantity).to('km s-1')
        t_phys[f"U<P_turb_CPROPS_{rstr}>"] = (
            3 / (4 * np.pi) *
            vtt[f"U<F*sigv^2/R^3_CPROPS_{rstr}>"].quantity *
            alphaCO21 / R_factor**3 / (gal_dist / u.rad) /
            const.k_B).to('K cm-3')
        t_phys[f"F<P_turb_CPROPS_{rstr}>"] = (
            3 / (4 * np.pi) *
            vtt[f"F<F*sigv^2/R^3_CPROPS_{rstr}>"].quantity *
            alphaCO21 / R_factor**3 / (gal_dist / u.rad) /
            const.k_B).to('K cm-3')
        t_phys[f"U<alpha_vir_CPROPS_{rstr}>"] = (  # Eq.6 in Sun+18
            5 / const.G *
            vtt[f"U<R*sigv^2/F_CPROPS_{rstr}>"].quantity /
            alphaCO21 * R_factor / (gal_dist / u.rad)).to('')
        t_phys[f"F<alpha_vir_CPROPS_{rstr}>"] = (  # Eq.6 in Sun+18
            5 / const.G *
            vtt[f"F<R*sigv^2/F_CPROPS_{rstr}>"].quantity /
            alphaCO21 * R_factor / (gal_dist / u.rad)).to('')

    # dynamical equilibrium pressure estimates
    Sigma_gas = Sigma_mol + Sigma_atom
    res_fid = res_pcs[-2].to('pc')  # 120 pc
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
         Sigma_gas * vdisp_z * np.sqrt(2*const.G*rho_star_flared)) /
        const.k_B).to('K cm-3')
    t_phys["P_DE_classic_fixedsiggas"] = (
        (np.pi * const.G / 2 * Sigma_gas**2 +
         Sigma_gas * vdisp_atom_z * np.sqrt(2*const.G*rho_star)) /
        const.k_B).to('K cm-3')
    t_phys["W_atom"] = (
        (np.pi * const.G / 2 * Sigma_atom**2 +
         np.pi * const.G * Sigma_atom * Sigma_mol +
         Sigma_atom * vdisp_atom_z * np.sqrt(2*const.G*rho_star)) /
        const.k_B).to('K cm-3')
    t_phys["P_DE_smooth"] = (
        (np.pi * const.G / 2 * Sigma_mol**2 +
         np.pi * const.G * Sigma_mol * rho_star * res_fid/2) /
        const.k_B).to('K cm-3') + t_phys['W_atom'].quantity
    for res_pc in res_pcs:
        R_cloud = res_pc / 2
        rstr = f"{res_pc.to('pc').value:.0f}pc"
        t_phys[f"A<W_cloud_self_pix_{rstr}>"] = (
            3/8 * np.pi * const.G *
            vtt[f"A<I^2_CO21_{rstr}>"].quantity * alphaCO21**2 /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_self_pix_{rstr}>"] = (
            3/8 * np.pi * const.G *
            vtt[f"F<I^2_CO21_{rstr}>"].quantity * alphaCO21**2 /
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
    for res_pc in res_pcs:
        rstr = f"{res_pc.to('pc').value:.0f}pc"
        t_phys[f"U<W_cloud_self_CPROPS_{rstr}>"] = (
            3 * const.G / 8 / np.pi *
            vtt[f"U<F^2/R^4_CPROPS_{rstr}>"].quantity *
            alphaCO21**2 / R_factor**4 /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_self_CPROPS_{rstr}>"] = (
            3 * const.G / 8 / np.pi *
            vtt[f"F<F^2/R^4_CPROPS_{rstr}>"].quantity *
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
            vtt[f"U<F/R_CPROPS_{rstr}>"].quantity *
            alphaCO21 / R_factor * (gal_dist / u.rad) /
            const.k_B).to('K cm-3')
        t_phys[f"F<W_cloud_star_CPROPS_{rstr}>"] = (
            3/2 * const.G * rho_star *
            vtt[f"F<F/R_CPROPS_{rstr}>"].quantity *
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
        for key in vtt.colnames:
            if key not in t_phys.colnames:
                t_phys[key] = vtt[key].quantity

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

    # averaging aperture (linear) size
    apersz = 1 * u.kpc

    # averaging aperture shape
    aperture_shape = 'hexagon'

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
        ctr_radec = np.array([row['RA_DEG'], row['DEC_DEG']]) * u.deg
        dist = row['DIST'] * u.Mpc
        incl = row['INCL'] * u.deg
        posang = row['POSANG'] * u.deg

        # skip targets with bad geometrical information
        if not ((incl >= 0*u.deg) and (incl < 90*u.deg) and
                np.isfinite(posang)):
            continue

        vttfile = (workdir /
                   f"{name}_{aperture_shape}_"
                   f"{apersz.to('kpc').value:.0f}kpc.ecsv")
        # skip targets with aperture statistics table already on disk
        if vttfile.is_file():
            print(f"Table file already on disk - skipping {name}")
            continue

        print(f"Processing data for {name}")

        # initialize a VoronoiTessTable
        print("  Initializing data table")
        infile = get_data_path('ALMA:CO:tpeak', name, apersz)
        if not infile.is_file():
            print(f"No CO low resolution data found for {name}")
            print("")
            continue
        apersz_deg = (apersz/dist*u.rad).to('deg').value
        with fits.open(infile) as hdul:
            vtt = VoronoiTessTable(
                hdul[0].header, cell_shape=aperture_shape,
                ref_radec=ctr_radec.value,
                seed_spacing=apersz_deg)

        # add galactic radii and projected angles in table
        print("  Calculating r_gal and phi_gal")
        radii, projang = deproject(
            center_coord=ctr_radec, incl=incl.value, pa=posang.value,
            ra=vtt['RA'], dec=vtt['DEC'])
        vtt['rang_gal'] = (radii * u.deg).to('arcsec')
        vtt['phi_gal'] = projang * u.deg
        # sort rows by galactic radii
        vtt[:] = vtt[np.argsort(vtt['r_gal'])]

        # add low resolution CO data in table
        print("  Resampling low resolution CO data")
        infile = get_data_path('ALMA:CO:mom0:strict', name, apersz)
        vtt.resample_image(
            infile, colname='I_CO21',
            unit='header', #unit=u.Unit('K km s-1'),
            fill_outside=np.nan)
        if np.isfinite(vtt['I_CO21']).sum() == 0:
            print(f"No CO detection in any aperture -- skip {name}")
            print("")
            continue

        # add HI data in table
        print("  Resampling HI data")
        infile = get_data_path('HI:mom0', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='I_21cm',
            unit=u.Unit('K km s-1'),
            fill_outside=np.nan)
        infile = get_data_path('HI:mom0', name)
        add_resampled_image_to_table(
            vtt, infile, colname='I_21cm_native',
            unit=u.Unit('K km s-1'),
            fill_outside=np.nan)

        # add S4G data in table
        print("  Resampling S4G data")
        infile = get_data_path('S4G:ICA3p6um', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='I_3p6um_ICA',
            unit=u.Unit('MJy sr-1'),
            fill_outside=np.nan)
        infile = get_data_path('S4G:3p6um', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='I_3p6um_raw',
            unit=u.Unit('MJy sr-1'),
            fill_outside=np.nan)

        # add Z0MGS data in table
        print("  Resampling Z0MGS data")
        infile = get_data_path('Z0MGS:SFR:NUVW3', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='Sigma_SFR_NUVW3',
            unit=u.Unit('Msun kpc-2 yr-1'),
            fill_outside=np.nan)
        infile = get_data_path('Z0MGS:SFR:W3ONLY', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='Sigma_SFR_W3ONLY',
            unit=u.Unit('Msun kpc-2 yr-1'),
            fill_outside=np.nan)

        # add environmental fraction (CO flux-weighted) in table
        print("  Calculating (CO flux-weighted) "
              "environmental fraction")
        res = lin_res[-1]
        wtfile = get_data_path('ALMA:CO:mom0:strict', name, res)
        for reg in regions:
            print(f"    > fraction of {reg}")
            envfile = get_data_path('S4G:env_mask:'+reg, name)
            add_env_frac_to_table(
                vtt, envfile, wtfile, colname='frac_'+reg)

        # add statistics of high resolution CO data in table
        print("  Calculating statistics of high resolution CO data")
        for res in lin_res:
            print(f"    @ {res.to('pc').value:.0f}pc resolution")
            bm0file = get_data_path('ALMA:CO:mom0:broad', name, res)
            sm0file = get_data_path('ALMA:CO:mom0:strict', name, res)
            sewfile = get_data_path('ALMA:CO:ew:strict', name, res)
            add_CO_stats_to_table(
                vtt, bm0file, sm0file, sewfile, res)

        # add statistics of CPROPS clouds in table
        print("  Calculating statistics of CPROPS clouds")
        for res in lin_res:
            print(f"    @ {res.to('pc').value:.0f}pc resolution")
            cpropsfile = get_data_path('ALMA:CPROPS', name, res)
            add_cprops_stats_to_table(
                vtt, cpropsfile, res)

        # mask rows where low resolution 'I_CO21' is NaN
        # This has to be the last step!!
        vtt.clean(discard_NaN='I_CO21')

        # write table to disk
        print("  Writing table to disk")
        vtt.write(vttfile)
        del vtt

        print(f"Finished processing data for {name}!")
        print("")

    # ----------------------------------------------------------------

    for row in catalog:
        # convert raw observables to physical properties
        print("  Constructing physical property table")
        name = row['NAME'].strip()
        vttfile = (workdir /
                   f"{name}_{aperture_shape}_"
                   f"{apersz.to('kpc').value:.0f}kpc.ecsv")
        vtt = Table.read(vttfile)
        t_phys = gen_phys_props_table(vtt, row)
        print(" Writing physical property table to disk")
        t_phys.write(vttfile.replace('.', '_phys.'))
        del vtt, t_phys

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
