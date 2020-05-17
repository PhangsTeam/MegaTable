import os
import sys
import warnings
from pathlib import Path
import numpy as np
from astropy import units as u, constants as const
from astropy.table import Table
from AlmaTools.XCO import predict_metallicity, predict_alphaCO10
from mega_table.table import RadialMegaTable
from mega_table.mixin import PhangsAlmaMixin, EnvMaskMixin
from mega_table.utils import (
    get_R21, get_alpha21cm, get_alpha3p6um, get_h_star, nanaverage)


# --------------------------------------------------------------------


class MyRadialMegaTable(
        PhangsAlmaMixin, EnvMaskMixin, RadialMegaTable):

    """
    Enhanced RadialMegaTable.
    """


# --------------------------------------------------------------------


def get_data_path(datatype, galname=None, lin_res=None):
    """
    Get the path to any required data on disk.
    """
    datatypes = datatype.split(':')

    # PHANGS data parent directory
    PHANGSdir = Path(os.getenv('PHANGSWORKDIR'))

    if datatypes[0] == 'sample_table':
        return (PHANGSdir / 'mega-tables' /
                'sample_table_for_Sun+20.fits')

    elif datatypes[0] == 'ALMA':
        # PHANGS-ALMA data
        basedir = PHANGSdir / 'ALMA'
        if datatypes[1] == 'CO':
            # PHANGS-ALMA CO map
            basedir /= 'v3p4-processed'
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
        gal_name, gal_dist_Mpc=None,
        gal_ra_deg=None, gal_dec_deg=None,
        gal_incl_deg=None, gal_posang_deg=None,
        rgal_bin_kpc=None, rgal_max_kpc=None,
        CO_res_pc=[], env_regions=[],
        verbose=True, writefile=''):

    rgal_bin_arcsec = np.rad2deg(
        rgal_bin_kpc / gal_dist_Mpc / 1e3) * 3600
    if rgal_max_kpc is None:
        rgal_max_arcsec = None
    else:
        rgal_max_arcsec = np.rad2deg(
            rgal_max_kpc / gal_dist_Mpc / 1e3) * 3600

    CO_high_res = np.array(CO_res_pc) * u.pc

    # initialize table
    if verbose:
        print("  Initializing data table")
    rt = MyRadialMegaTable(
        gal_ra_deg, gal_dec_deg, rgal_bin_arcsec,
        rgal_max_arcsec=rgal_max_arcsec,
        gal_incl_deg=gal_incl_deg,
        gal_posang_deg=gal_posang_deg)

    # add z0MGS data in table
    if verbose:
        print("  Incorporating z0MGS data")
    rt.calc_image_stats(
        get_data_path('z0MGS:SFR:FUVW4', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_FUVW4',
        unit=u.Unit('Msun kpc-2 yr-1'))
    rt.calc_image_stats(
        get_data_path('z0MGS:SFR:NUVW4', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_NUVW4',
        unit=u.Unit('Msun kpc-2 yr-1'))
    rt.calc_image_stats(
        get_data_path('z0MGS:SFR:W4ONLY', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_W4ONLY',
        unit=u.Unit('Msun kpc-2 yr-1'))
    rt.calc_image_stats(
        get_data_path('z0MGS:SFR:FUVW3', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_FUVW3',
        unit=u.Unit('Msun kpc-2 yr-1'))
    rt.calc_image_stats(
        get_data_path('z0MGS:SFR:NUVW3', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_NUVW3',
        unit=u.Unit('Msun kpc-2 yr-1'))
    rt.calc_image_stats(
        get_data_path('z0MGS:SFR:W3ONLY', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='Sigma_SFR_W3ONLY',
        unit=u.Unit('Msun kpc-2 yr-1'))

    # add S4G data in table
    if verbose:
        print("  Incorporating S4G data")
    rt.calc_image_stats(
        get_data_path('S4G:ICA3p6um', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='I_3p6um_ICA', unit=u.Unit('MJy sr-1'))
    rt.calc_image_stats(
        get_data_path('S4G:3p6um', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='I_3p6um_raw', unit=u.Unit('MJy sr-1'))

    # add HI data in table
    if verbose:
        print("  Incorporating HI data")
    rt.calc_image_stats(
        get_data_path('HI:mom0', gal_name),
        suppress_error=True, stat_func=nanaverage,
        colname='I_21cm', unit=u.Unit('K km s-1'))
    # mask all values below 20 K km s-1 (for now)
    thres = 20 * u.Unit('K km s-1')
    rt.table['I_21cm'][rt['I_21cm'].quantity < thres] = 0

    # add environmental fraction (CO flux-weighted) in table
    if verbose:
        print("  Calculating (CO flux-weighted) "
              "environmental fraction")
    res = CO_high_res[-1]
    for reg in env_regions:
        if verbose:
            print(f"    > fraction of {reg}")
        rt.calc_env_frac(
            get_data_path('S4G:env_mask:'+reg, gal_name),
            get_data_path('ALMA:CO:mom0:strict', gal_name, res),
            suppress_error=True, colname='frac_'+reg)

    # add statistics of high resolution CO data in table
    if verbose:
        print("  Calculating statistics of high resolution CO data")
    for res in CO_high_res:
        if verbose:
            print(f"    @ {res.to('pc').value:.0f}pc resolution")
        # search for "parts" if the full mosaic is not available
        # at the requested linear resolution
        for part_str in ['', '_1', '_2', '_3']:
            gal_str = gal_name + part_str
            if get_data_path(
                    'ALMA:CO:mom0:broad', gal_str, res).is_file():
                break
        rt.calc_CO_stats(
            get_data_path('ALMA:CO:mom0:broad', gal_str, res),
            get_data_path('ALMA:CO:mom0:strict', gal_str, res),
            get_data_path('ALMA:CO:ew:strict', gal_str, res),
            res, suppress_error=True)

    # add statistics of CPROPS clouds in table
    if verbose:
        print("  Calculating statistics of CPROPS clouds")
    for res in CO_high_res:
        if verbose:
            print(f"    @ {res.to('pc').value:.0f}pc resolution")
        rt.calc_cprops_stats(
            get_data_path('ALMA:CPROPS', gal_name, res),
            res, suppress_error=True)

    # record metadata
    rt.meta['GALAXY'] = str(gal_name)
    rt.meta['DIST_MPC'] = gal_dist_Mpc
    rt.meta['RA_DEG'] = gal_ra_deg
    rt.meta['DEC_DEG'] = gal_dec_deg
    rt.meta['INCL_DEG'] = gal_incl_deg
    rt.meta['PA_DEG'] = gal_posang_deg

    # write table to disk
    if writefile:
        if verbose:
            print("  Writing table to disk")
        rt.write(rtfile, add_timestamp=True, overwrite=True)
        return writefile
    else:
        return rt


# --------------------------------------------------------------------


def gen_phys_props_table(
        rtfile, gal_name=None, gal_logMstar=None,
        gal_Reff_arcsec=None, gal_Rstar_arcsec=None,
        CO_res_pc=[], append_raw_data=False, note='', version=0,
        writefile=''):

    from itertools import product

    CO_high_res = np.array(CO_res_pc) * u.pc

    # read mega-table
    rt = RadialMegaTable.read(rtfile)
    assert rt.meta['GALAXY'] == str(gal_name)

    # galaxy global parameters
    gal_cosi = np.cos(np.deg2rad(rt.meta['INCL_DEG']))
    gal_dist = rt.meta['DIST_MPC'] * u.Mpc
    gal_Mstar = 10**gal_logMstar * u.Msun
    gal_Reff = (np.deg2rad(gal_Reff_arcsec/3600)*gal_dist).to('kpc')
    gal_Rstar = (np.deg2rad(gal_Rstar_arcsec/3600)*gal_dist).to('kpc')

    # initiate new table
    pt = RadialMegaTable(
        rt.meta['RA_DEG'], rt.meta['DEC_DEG'],
        rt.meta['RBIN_AS'],
        rgal_max_arcsec=rt.meta['RMAX_AS'],
        gal_incl_deg=rt.meta['INCL_DEG'],
        gal_posang_deg=rt.meta['PA_DEG'])
    for key in rt.meta:
        pt.meta[key] = rt.meta[key]

    # coordinates
    pt['REGION'] = rt['REGION']
    pt['r_gal_min'] = r_gal_min = (
        rt['r_gal_angl_min'].quantity.to('rad').value *
        gal_dist).to('kpc')
    pt['r_gal_max'] = r_gal_max = (
        rt['r_gal_angl_max'].quantity.to('rad').value *
        gal_dist).to('kpc')
    r_gal = (r_gal_min + r_gal_max) / 2

    # CO contribution by environments
    for reg in regions:
        pt[f"frac_{reg}"] = rt[f"frac_{reg}"].quantity
    # manually flag the aperture on the galactic center as "bulge"
    # if no environment masks are available for this target
    if ((np.isfinite(pt['frac_bulge']).sum() == 0) and
        (np.isfinite(pt['frac_bars']).sum() == 0) and
        (np.isfinite(pt['frac_disk']).sum() == 0)):
        pt['frac_bulge'][np.nonzero(r_gal_min == 0)[0]] = 1.

    # metallicity
    pt['log(O/H)_O3N2_PP04_predicted'] = predict_metallicity(
        gal_Mstar, calibrator='O3N2(PP04)', MZR='Sanchez+19',
        Rgal=r_gal, Re=gal_Reff, gradient='Sanchez+14')
    pt['Zprime'] = Zprime = 10**(
        pt['log(O/H)_O3N2_PP04_predicted'] - 8.69)

    # SFR surface density
    pt['Sigma_SFR'] = np.nan * u.Unit('Msun kpc-2 yr-1')
    for key in (
            'Sigma_SFR_FUVW4', 'Sigma_SFR_NUVW4', 'Sigma_SFR_W4ONLY',
            'Sigma_SFR_FUVW3', 'Sigma_SFR_NUVW3', 'Sigma_SFR_W3ONLY',
            ):
        pt[key] = rt[key].quantity.to('Msun kpc-2 yr-1')
        if (np.isfinite(pt[key]).sum() != 0 and
            np.isfinite(pt['Sigma_SFR']).sum() == 0):
            pt['Sigma_SFR'] = pt[key].quantity

    # stellar mass surface densities
    alpha3p6um = get_alpha3p6um(ref='MS14')
    pt['Sigma_star'] = np.nan * u.Unit('Msun pc-2')
    for key in ('I_3p6um_ICA', 'I_3p6um_raw'):
        newkey = 'Sigma_star'+key[-4:]
        pt[newkey] = (
            rt[key].quantity * gal_cosi * alpha3p6um
        ).to('Msun/pc^2')
        if (np.isfinite(pt[newkey]).sum() != 0 and
            np.isfinite(pt['Sigma_star']).sum() == 0):
            pt['Sigma_star'] = pt[newkey].quantity
    Sigma_star = pt['Sigma_star'].quantity

    # HI mass surface density
    alpha21cm = get_alpha21cm(include_He=True)
    pt['Sigma_atom'] = (
        rt['I_21cm'].quantity * gal_cosi * alpha21cm
    ).to('Msun pc-2')
    Sigma_atom = pt['Sigma_atom'].quantity

    # H2 surface density (low resolution)
    R21 = get_R21()
    rstr_fid = f"{CO_high_res[-1].to('pc').value:.0f}pc"  # 150 pc
    pt['I_CO21'] = I_CO21 = (
        rt[f"Flux_CO21_broad_{rstr_fid}"].quantity /
        rt[f"Area_CO21_total_{rstr_fid}"].quantity).to('K km s-1')
    ICO10kpc = I_CO21 * gal_cosi / R21
    ICO10GMC = rt[f"F<I_CO21_{rstr_fid}>"].quantity / R21
    pt['alphaCO10_MW'] = predict_alphaCO10(
        prescription='constant')
    pt['alphaCO10_PHANGS'] = predict_alphaCO10(
        prescription='PHANGS', PHANGS_Zprime=Zprime)
    pt['alphaCO10_N12'] = predict_alphaCO10(
        prescription='Narayanan+12',
        N12_Zprime=Zprime, N12_WCO10GMC=ICO10GMC)
    pt['alphaCO10_B13'] = predict_alphaCO10(
        prescription='Bolatto+13',
        iterative=True, suppress_error=True,
        B13_Zprime=Zprime, B13_Sigmakpc=Sigma_atom+Sigma_star,
        B13_WCO10kpc=ICO10kpc, B13_WCO10GMC=ICO10GMC)
    alphaCO21 = pt['alphaCO10_PHANGS'].quantity / R21
    pt['Sigma_mol'] = (
        I_CO21 * alphaCO21 * gal_cosi).to('Msun pc-2')
    Sigma_mol = pt['Sigma_mol'].quantity

    # CO map statistics
    for res, wstr in product(CO_high_res, ('A', 'F')):
        R_cloud = res / 2
        rstr = f"{res.to('pc').value:.0f}pc"
        pt[f"fracA_CO21_{rstr}"] = (
            rt[f"Area_CO21_strict_{rstr}"].quantity /
            rt[f"Area_CO21_total_{rstr}"].quantity).to('')
        pt[f"fracF_CO21_{rstr}"] = (
            rt[f"Flux_CO21_strict_{rstr}"].quantity /
            rt[f"Flux_CO21_broad_{rstr}"].quantity).to('')
        pt[f"clumping_CO21_{rstr}"] = (
            rt[f"F<I_CO21_{rstr}>"].quantity /
            rt[f"A<I_CO21_{rstr}>"].quantity).to('')
        pt[f"{wstr}<Sigma_mol_pix_{rstr}>"] = (
            rt[f"{wstr}<I_CO21_{rstr}>"].quantity *
            alphaCO21).to('Msun pc-2')
        pt[f"{wstr}<vdisp_mol_pix_{rstr}>"] = (
            rt[f"{wstr}<sigv_CO21_{rstr}>"].quantity).to('km s-1')
        pt[f"{wstr}<P_turb_pix_{rstr}>"] = (
            # Sun+20 Eq.4
            3/2 * rt[f"{wstr}<I*sigv^2_CO21_{rstr}>"].quantity /
            (2*R_cloud) * alphaCO21 / const.k_B).to('K cm-3')
        pt[f"{wstr}<alpha_vir_pix_{rstr}>"] = (
            # Sun+18 Eq.13
            5 * np.log(2) / (10/9 * np.pi * const.G) *
            rt[f"{wstr}<sigv^2/I_CO21_{rstr}>"].quantity /
            R_cloud / alphaCO21).to('')

    # CPROPS cloud statistics
    R_factor = np.sqrt(5) / 1.91  # Rosolowsky&Leroy06 Sec.3.1
    for res, wstr in product(CO_high_res, ('U', 'F')):
        los_depth = res
        rstr = f"{res.to('pc').value:.0f}pc"
        pt[f"Nobj_CPROPS_{rstr}"] = (
            rt[f"Nobj_CPROPS_{rstr}"].quantity).to('')
        pt[f"fracF_CPROPS_{rstr}"] = (
            rt[f"Flux_CPROPS_total_{rstr}"].quantity /
            rt[f"Flux_CO21_broad_{rstr}"].quantity).to('')
        pt[f"{wstr}<M_mol_CPROPS_{rstr}>"] = (
            # Note that F [=] K*km/s arcsec2
            rt[f"{wstr}<F_CPROPS_{rstr}>"].quantity *
            alphaCO21 * gal_dist**2 / u.sr).to('Msun')
        pt[f"{wstr}<R_CPROPS_{rstr}>"] = (
            # Note that R [=] arcsec
            rt[f"{wstr}<R_CPROPS_{rstr}>"].quantity *
            gal_dist / u.rad).to('pc')
        pt[f"{wstr}<Sigma_mol_CPROPS_{rstr}>"] = (
            rt[f"{wstr}<F/R^2_CPROPS_{rstr}>"].quantity *
            alphaCO21 / (np.pi*R_factor**2)).to('Msun pc-2')
        pt[f"{wstr}<vdisp_mol_CPROPS_{rstr}>"] = (
            rt[f"{wstr}<sigv_CPROPS_{rstr}>"].quantity).to('km s-1')
        pt[f"{wstr}<P_turb_CPROPS_sphere_{rstr}>"] = (
            # Sun+20 Eq.28
            3 / (4*np.pi) *
            rt[f"{wstr}<F*sigv^2/R^3_CPROPS_{rstr}>"].quantity *
            alphaCO21 / R_factor**3 / (gal_dist / u.rad) /
            const.k_B).to('K cm-3')
        pt[f"{wstr}<P_turb_CPROPS_fixlos_{rstr}>"] = (
            # Sun+20 Sec.6.3
            3 / (2*np.pi) *
            rt[f'{wstr}<F*sigv^2/R^2_CPROPS_{rstr}>'].quantity *
            alphaCO21 / R_factor**2 / los_depth /
            const.k_B).to('K cm-3')
        pt[f"{wstr}<alpha_vir_CPROPS_sphere_{rstr}>"] = (
            # Sun+18 Eq.6
            5 / const.G *
            rt[f"{wstr}<R*sigv^2/F_CPROPS_{rstr}>"].quantity /
            alphaCO21 * R_factor / (gal_dist / u.rad)).to('')

    # dynamical equilibrium pressure estimates
    Sigma_gas = Sigma_mol + Sigma_atom
    rstr_fid = f"{CO_high_res[-1].to('pc').value:.0f}pc"  # 150 pc
    vdisp_mol_z = pt[f"F<vdisp_mol_pix_{rstr_fid}>"].quantity
    vdisp_atom_z = 10 * u.Unit('km s-1')
    vdisp_z = (
        (vdisp_mol_z * Sigma_mol + vdisp_atom_z * Sigma_atom) /
        Sigma_gas).to('km s-1')
    rho_star = (
        pt['Sigma_star'] / 4 /
        get_h_star(gal_Rstar, diskshape='flat')
    ).to('Msun pc-3')
    pt["P_DE_classic"] = (
        # Sun+20 Eq.12
        (np.pi * const.G / 2 * Sigma_gas**2 +
         Sigma_gas * vdisp_z * np.sqrt(2*const.G*rho_star)) /
        const.k_B).to('K cm-3')

    if append_raw_data:
        for key in rt[:].colnames:
            if key not in pt[:].colnames:
                pt[key] = rt[key].quantity

    # record metadata
    pt.meta['LOGMSTAR'] = gal_logMstar
    pt.meta['REFF_AS'] = gal_Reff_arcsec
    pt.meta['RDISK_AS'] = gal_Rstar_arcsec
    pt.meta['TBLNOTE'] = str(note)
    pt.meta['VERSION'] = float(version)

    if writefile:
        pt.write(writefile, add_timestamp=True, overwrite=True)
        return writefile
    else:
        return pt


######################################################################
######################################################################


if __name__ == '__main__':

    # ----------------------------------------------------------------
    # parameters that specify how to construct the mega-tables
    # ----------------------------------------------------------------

    # ring (deprojected) width
    rgal_bin = 0.5 * u.kpc

    # maximal (depojected) galactic radius
    rgal_max = 25 * u.kpc

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
    logging = True
    if logging:
        # output log to a file
        orig_stdout = sys.stdout
        log = open(workdir/(str(Path(__file__).stem)+'.log'), 'w')
        sys.stdout = log

    # read PHANGS sample table
    catalog = Table.read(get_data_path('sample_table'))
    # only keep targets with the 'HAS_ALMA' tag
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
        Reff = row['SIZE_W1_R50'] * u.arcsec
        Rstar = row['SIZE_S4G_RSTAR'] * u.arcsec

        # skip targets with bad geometrical information
        if not ((incl >= 0*u.deg) and (incl < 90*u.deg) and
                np.isfinite(posang)):
            print(f"Bad orientation measurement - skipping {name}")
            print("")
            continue
        # skip targets with bad distance
        if not (dist > 0):
            print(f"Bad distance measurement - skipping {name}")
            print("")
            continue

        print(f"Processing data for {name}")

        # ------------------------------------------------------------
        # generate raw measurement table
        # ------------------------------------------------------------

        rtfile = (
            workdir /
            f"{name}_radial_profile_"
            f"{rgal_bin.to('pc').value:.0f}pc.fits")
        if rtfile.is_file():
            print(f"Table file already on disk - skipping {name}")
            print("")
            continue
        else:
            print(f"Constructing raw measurement table for {name}")
            gen_raw_measurement_table(
                name, gal_dist_Mpc=dist.value,
                gal_ra_deg=ra.value, gal_dec_deg=dec.value,
                gal_incl_deg=incl.value, gal_posang_deg=posang.value,
                rgal_bin_kpc=rgal_bin.to('kpc').value,
                rgal_max_kpc=rgal_max.to('kpc').value,
                CO_res_pc=CO_high_res.to('pc').value,
                env_regions=regions,
                verbose=True, writefile=rtfile)

        # ------------------------------------------------------------
        # convert raw measurements to physical properties
        # ------------------------------------------------------------

        ptfile = (
            workdir /
            f"{name}_radial_profile_"
            f"{rgal_bin.to('pc').value:.0f}pc_phys.fits")
        if (rtfile.is_file() and not ptfile.is_file()):
            print(f"Constructing physical property table for {name}")
            gen_phys_props_table(
                rtfile, gal_name=name,
                gal_logMstar=logMstar,
                gal_Reff_arcsec=Reff.value,
                gal_Rstar_arcsec=Rstar.value,
                CO_res_pc=CO_high_res.to('pc').value,
                append_raw_data=False,
                note=(
                    'PHANGS sample table v1p4 (except dist=v1p3); '
                    'PHANGS-ALMA v3p4; '
                    'CPROPS catalog v3; '
                    'PHANGS-VLA v1p0'),
                version=1.0, writefile=ptfile)

        # ------------------------------------------------------------

        print(f"Finished processing data for {name}!")
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
