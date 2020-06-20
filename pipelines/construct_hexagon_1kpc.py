import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
from astropy import units as u, constants as const
from astropy.table import Table, QTable
from astropy.io import fits
from AlmaTools.XCO import predict_alphaCO10
from mega_table.table import TessellMegaTable
from mega_table.mixin import PhangsAlmaMixin, EnvMaskMixin
from mega_table.utils import (
    get_alpha3p6um, get_h_star, nanaverage, deproject)

warnings.filterwarnings('ignore')

logging = False

# --------------------------------------------------------------------


class MyTessellMegaTable(
        PhangsAlmaMixin, EnvMaskMixin, TessellMegaTable):

    """
    Enhanced TessellMegaTable.
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
        return PHANGSdir / 'sample' / 'sample_Sun+20b.ecsv'

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
        elif datatypes[1] == 'alphaCO':
            # PHANGS alphaCO map
            basedir = basedir / 'alphaCO'  # / 'v0p1'
            fname_seq = [galname] + datatypes[2:]

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

    elif datatypes[0] == 'Halpha':
        # narrow band Halpha data
        basedir = PHANGSdir / 'Halpha'
        fname_seq = [galname] + datatypes[1:]
        if lin_res is not None:
            fname_seq += [f"{lin_res.to('pc').value:.0f}pc"]

    else:
        raise ValueError("Unrecognized dataset")

    return basedir / ('_'.join(fname_seq) + '.fits')


# --------------------------------------------------------------------


def gen_raw_measurement_table(
        gal_name, gal_dist_Mpc=None,
        gal_ra_deg=None, gal_dec_deg=None,
        gal_incl_deg=None, gal_posang_deg=None,
        aperture_shape=None, aperture_size_kpc=None,
        config=None, verbose=True, writefile=''):

    aperture_size_arcsec = np.rad2deg(
        aperture_size_kpc / gal_dist_Mpc / 1e3) * 3600

    # initialize table
    if verbose:
        print("  Initializing data table")
    infile = get_data_path(
        'ALMA:CO:tpeak', gal_name, aperture_size_kpc*u.kpc)
    if not infile.is_file():
        if verbose:
            print(f"No CO low resolution data found for {gal_name}")
            print("")
        return
    with fits.open(infile) as hdul:
        rt = MyTessellMegaTable(
            hdul[0].header,
            aperture_shape=aperture_shape,
            aperture_size_arcsec=aperture_size_arcsec,
            gal_ra_deg=gal_ra_deg,
            gal_dec_deg=gal_dec_deg)

    # add columns according to config
    for row in config:

        if row['res_pc'] == 0:
            res = None
        else:
            res = row['res_pc'] * u.pc

        if verbose:
            print(f"  Calculating {row['colname']}")

        if row['method'] == 'deproject':
            # galactocentric radii and projected angles
            radii, projang = deproject(
                center_ra=gal_ra_deg, center_dec=gal_dec_deg,
                incl=gal_incl_deg, pa=gal_posang_deg,
                ra=rt['RA'], dec=rt['DEC'])
            colname_r, colname_phi = row['colname'].split('&')
            unit_r, unit_phi = row['unit'].split('&')
            rt[colname_r] = (radii * u.deg).to(unit_r)
            rt[colname_phi] = (projang * u.deg).to(unit_phi)
            # sort rows by galactocentric radii
            rt[:] = rt[np.argsort(rt[colname_r])]

        elif row['method'] == 'env_frac':
            # environmental fraction
            envsource, wtsource = row['source'].split('&')
            rt.calc_env_frac(
                get_data_path(envsource, gal_name),
                get_data_path(wtsource, gal_name, res),
                colname=row['colname'], suppress_error=True)

        elif row['method'] == 'CO_stats':
            # pixel statistics from high resolution CO image
            bm0source, sm0source, sewsource = row['source'].split('&')
            # search for "parts" if the full mosaic is not available
            # at the requested linear resolution
            for part_str in ['', '_1', '_2', '_3']:
                gal_str = gal_name + part_str
                if get_data_path(bm0source, gal_str, res).is_file():
                    break
            rt.calc_CO_stats(
                get_data_path(bm0source, gal_str, res),
                get_data_path(sm0source, gal_str, res),
                get_data_path(sewsource, gal_str, res),
                res, suppress_error=True)

        elif row['method'] == 'CPROPS_stats':
            # cloud statistics from CPROPS catalog
            rt.calc_cprops_stats(
                get_data_path(row['source'], gal_name, res),
                res, suppress_error=True)

        elif row['method'] == 'resample':
            rt.resample_image(
                get_data_path(row['source'], gal_name, res),
                colname=row['colname'], unit=u.Unit(row['unit']),
                suppress_error=True, fill_outside=np.nan)

        elif row['method'] == 'area_mean':
            rt.calc_image_stats(
                get_data_path(row['source'], gal_name, res),
                colname=row['colname'], unit=u.Unit(row['unit']),
                suppress_error=True, stat_func=nanaverage)

        else:
            raise ValueError(
                f"Inavlid method {row['method']} - check config file")

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
        rt.write(writefile, add_timestamp=True, overwrite=True)
        return writefile
    else:
        return rt


# --------------------------------------------------------------------


def gen_phys_props_table(
        rtfile, gal_name=None,
        # gal_logMstar=None, gal_Reff_arcsec=None,
        gal_Rstar_arcsec=None,
        config=None, note='', version=0, writefile='',
        **params):

    # read raw measurement table
    rt = TessellMegaTable.read(rtfile, ignore_inconsistency=True)
    assert rt.meta['GALAXY'] == str(gal_name)

    # galaxy global parameters
    gal_cosi = np.cos(np.deg2rad(rt.meta['INCL_DEG']))
    gal_dist = rt.meta['DIST_MPC'] * u.Mpc
    # gal_Mstar = 10**gal_logMstar * u.Msun
    # gal_Reff = (np.deg2rad(gal_Reff_arcsec/3600)*gal_dist).to('kpc')
    gal_Rstar = (np.deg2rad(gal_Rstar_arcsec/3600)*gal_dist).to('kpc')

    # initiate new table
    pt = TessellMegaTable(
        fits.Header(rt.meta),
        aperture_shape=rt.meta['APERTYPE'],
        aperture_size_arcsec=rt.meta['APER_AS'],
        gal_ra_deg=rt.meta['RA_DEG'],
        gal_dec_deg=rt.meta['DEC_DEG'])
    pt.table = QTable()
    for key in rt.meta:
        pt.meta[key] = rt.meta[key]

    if 'coord' in config['group']:
        # coordinates
        pt['RA'] = rt['RA']
        pt['DEC'] = rt['DEC']
        pt['r_gal'] = rt['r_gal_angl'].to('rad').value * gal_dist
        pt['phi_gal'] = rt['phi_gal']
    else:
        pt['r_gal'] = np.nan * u.Unit('kpc')

    if 'env_frac' in config['group']:
        # CO contribution by environments
        for key in config[config['group'] == 'env_frac']['colname']:
            pt[key] = rt[key]

    if 'metal' in config['group']:
        # metallicity
        pt['log(O/H)_MZR+GRD'] = rt['log(O/H)_MZR+GRD']
        pt['Zprime'] = rt['Zprime']
        with fits.open(get_data_path(
                'ALMA:alphaCO:Zprime', gal_name)) as hdul:
            pt['Zprime'].description = (
                f"({hdul[0].header['METALREF']})")
    else:
        pt['Zprime'] = np.nan

    if 'SFR' in config['group']:
        # SFR surface density
        pt['Sigma_SFR'] = np.nan * u.Unit('Msun yr-1 kpc-2')
        for key in config[config['group'] == 'SFR']['colname']:
            if key == 'Sigma_SFR':
                continue
            if np.isfinite(rt[key+'_kpc']).sum() != 0:
                pt[key] = rt[key+'_kpc']
            elif key+'_nat' in rt.colnames:
                pt[key] = rt[key+'_nat']
                pt[key].description = '[coarser resolution]'
            else:
                pt[key] = np.nan * u.Unit('Msun yr-1 kpc-2')
            if (np.isfinite(pt[key]).sum() != 0 and
                    np.isfinite(pt['Sigma_SFR']).sum() == 0):
                pt['Sigma_SFR'] = pt[key]
                pt['Sigma_SFR'].description = (
                    f"({key.replace('Sigma_SFR_', '')})")
    else:
        pt['Sigma_SFR'] = np.nan * u.Unit('Msun yr-1 kpc-2')

    if 'star' in config['group']:
        # stellar mass surface density
        pt['Sigma_star'] = np.nan * u.Unit('Msun pc-2')
        for key in config[config['group'] == 'star']['colname']:
            if key == 'Sigma_star':
                continue
            pt[key] = (
                rt[key.replace('Sigma_star', 'I')+'_kpc'] *
                gal_cosi * get_alpha3p6um(ref='MS14'))
            if (np.isfinite(pt[key]).sum() != 0 and
                np.isfinite(pt['Sigma_star']).sum() == 0):
                pt['Sigma_star'] = pt[key]
                pt['Sigma_star'].description = (
                    f"({key.replace('Sigma_star_', '')})")
    else:
        pt['Sigma_star'] = np.nan * u.Unit('Msun pc-2')

    if 'HI' in config['group']:
        # atomic gas mass surface density
        if np.isfinite(rt['I_21cm_kpc']).sum() != 0:
            pt['Sigma_atom'] = (
                rt['I_21cm_kpc'] * gal_cosi *
                params['alpha21cm'] *
                u.Unit(params['alpha21cm_unit']))
        else:
            pt['Sigma_atom'] = (
                rt['I_21cm_nat'] * gal_cosi *
                params['alpha21cm'] *
                u.Unit(params['alpha21cm_unit']))
            pt['Sigma_atom'].description = '[coarser resolution]'
    else:
        pt['Sigma_atom'] = np.nan * u.Unit('Msun pc-2')

    if 'alphaCO' in config['group']:
        # CO-to-H2 conversion factor
        pt['alphaCO10'] = np.nan * u.Unit('Msun pc-2 K-1 km-1 s')
        pt['alphaCO10_PHANGS'] = rt['alphaCO10_PHANGS']
        pt['alphaCO10_MW'] = predict_alphaCO10(
            prescription='Galactic')
        if 'CO_stats' in config['group']:
            res_fid = np.max(
                config[config['group'] == 'CO_stats']['res_pc'])
            ICO10GMC = rt[f"F<I_CO21_{res_fid}pc>"] / params['CO_R21']
            pt['alphaCO10_N12'] = predict_alphaCO10(
                prescription='Narayanan+12',
                N12_Zprime=pt['Zprime'],
                N12_WCO10GMC=ICO10GMC)
            del res_fid, ICO10GMC
        else:
            pt['alphaCO10_N12'] = (
                np.nan * u.Unit('Msun pc-2 K-1 km-1 s'))
        if 'CO_stats' in config['group'] and 'H2' in config['group']:
            res_fid = np.max(
                config[config['group'] == 'CO_stats']['res_pc'])
            ICO10GMC = rt[f"F<I_CO21_{res_fid}pc>"] / params['CO_R21']
            ICO10kpc = rt['I_CO21_kpc'] * gal_cosi / params['CO_R21']
            pt['alphaCO10_B13'] = predict_alphaCO10(
                prescription='Bolatto+13',
                iterative=True, suppress_error=True,
                B13_Zprime=pt['Zprime'],
                B13_Sigmakpc=pt['Sigma_star']+pt['Sigma_atom'],
                B13_WCO10kpc=ICO10kpc,
                B13_WCO10GMC=ICO10GMC)
            del res_fid, ICO10GMC, ICO10kpc
        else:
            pt['alphaCO10_B13'] = (
                np.nan * u.Unit('Msun pc-2 K-1 km-1 s'))
        if np.isfinite(pt['alphaCO10_PHANGS']).sum() != 0:
            pt['alphaCO10'] = pt['alphaCO10_PHANGS']
            pt['alphaCO10'].description = "(PHANGS)"
        else:
            pt['alphaCO10'] = pt['alphaCO10_MW']
            pt['alphaCO10'].description = "(MW)"
        pt['alphaCO21'] = pt['alphaCO10'] / params['CO_R21']
        pt['alphaCO21'].description = pt['alphaCO10'].description
    else:
        pt['alphaCO10'] = pt['alphaCO21'] = (
            np.nan * u.Unit('Msun pc-2 K-1 km-1 s'))

    if 'H2' in config['group']:
        # molecular gas mass surface density
        pt['I_CO21'] = rt['I_CO21_kpc']
        pt['Sigma_mol'] = pt['I_CO21'] * pt['alphaCO21'] * gal_cosi
    else:
        pt['Sigma_mol'] = np.nan * u.Unit('Msun pc-2')

    if 'CO_stats' in config['group']:
        # CO pixel statistics
        for res_pc in (
                config[config['group'] == 'CO_stats']['res_pc']):
            R_cloud = res_pc * u.pc / 2
            rstr = f"{res_pc}pc"
            pt[f"fracA_CO21_{rstr}"] = (
                rt[f"Area_CO21_strict_{rstr}"] /
                rt[f"Area_CO21_total_{rstr}"])
            pt[f"fracF_CO21_{rstr}"] = (
                rt[f"Flux_CO21_strict_{rstr}"] /
                rt[f"Flux_CO21_broad_{rstr}"])
            pt[f"clumping_CO21_{rstr}"] = (
                rt[f"F<I_CO21_{rstr}>"] /
                rt[f"A<I_CO21_{rstr}>"])
            for wstr in ['A', 'F']:
                pt[f"{wstr}<Sigma_mol_pix_{rstr}>"] = (
                    rt[f"{wstr}<I_CO21_{rstr}>"] * pt['alphaCO21'])
                pt[f"{wstr}<vdisp_mol_pix_{rstr}>"] = (
                    rt[f"{wstr}<sigv_CO21_{rstr}>"])
                pt[f"{wstr}<P_turb_pix_{rstr}>"] = (
                    # Sun+20 Eq.4
                    3/2 * rt[f"{wstr}<I*sigv^2_CO21_{rstr}>"] /
                    (2*R_cloud) * pt['alphaCO21'] / const.k_B)
                pt[f"{wstr}<alpha_vir_pix_{rstr}>"] = (
                    # Sun+18 Eq.13
                    5 * np.log(2) / (10/9 * np.pi * const.G) *
                    rt[f"{wstr}<sigv^2/I_CO21_{rstr}>"] /
                    R_cloud / pt['alphaCO21'])

    if 'CPROPS_stats' in config['group']:
        # CPROPS cloud statistics
        R_factor = np.sqrt(5) / 1.91  # Rosolowsky&Leroy06 Sec.3.1
        for res_pc in (
                config[config['group'] == 'CPROPS_stats']['res_pc']):
            los_depth = 100 * u.pc
            rstr = f"{res_pc}pc"
            pt[f"Nobj_CPROPS_{rstr}"] = rt[f"Nobj_CPROPS_{rstr}"]
            pt[f"fracF_CPROPS_{rstr}"] = (
                rt[f"Flux_CPROPS_total_{rstr}"] /
                rt[f"Flux_CO21_broad_{rstr}"])
            for wstr in ['U', 'F']:
                pt[f"{wstr}<M_mol_CPROPS_{rstr}>"] = (
                    # Note that F [=] K*km/s arcsec2
                    rt[f"{wstr}<F_CPROPS_{rstr}>"] *
                    pt['alphaCO21'] * gal_dist**2 / u.sr)
                pt[f"{wstr}<R_CPROPS_{rstr}>"] = (
                    # Note that R [=] arcsec
                    rt[f"{wstr}<R_CPROPS_{rstr}>"] *
                    R_factor * gal_dist / u.rad)
                pt[f"{wstr}<Sigma_mol_CPROPS_{rstr}>"] = (
                    rt[f"{wstr}<F/R^2_CPROPS_{rstr}>"] *
                    pt['alphaCO21'] / (np.pi*R_factor**2))
                pt[f"{wstr}<vdisp_mol_CPROPS_{rstr}>"] = (
                    rt[f"{wstr}<sigv_CPROPS_{rstr}>"])
                pt[f"{wstr}<P_turb_CPROPS_sph_{rstr}>"] = (
                    # Sun+20 Eq.28
                    3 / (4*np.pi) * pt['alphaCO21'] *
                    rt[f"{wstr}<F*sigv^2/R^3_CPROPS_{rstr}>"] /
                    R_factor**3 / (gal_dist / u.rad) / const.k_B)
                pt[f"{wstr}<P_turb_CPROPS_cyl_{rstr}>"] = (
                    # Sun+20 Sec.6.3
                    3 / (2*np.pi) * pt['alphaCO21'] *
                    rt[f'{wstr}<F*sigv^2/R^2_CPROPS_{rstr}>'] /
                    R_factor**2 / los_depth / const.k_B)
                pt[f"{wstr}<alpha_vir_CPROPS_sph_{rstr}>"] = (
                    # Sun+18 Eq.6
                    5 / const.G *
                    rt[f"{wstr}<R*sigv^2/F_CPROPS_{rstr}>"] /
                    pt['alphaCO21'] * R_factor / (gal_dist / u.rad))

    if 'P_DE' in config['group']:
        # dynamical equilibrium pressure
        pt["rho_star_midplane"] = (
            pt['Sigma_star'] / 4 /
            get_h_star(gal_Rstar, diskshape='flat'))
        Sigma_gas = pt['Sigma_mol'] + pt['Sigma_atom']
        if 'CO_stats' in config['group']:
            res_fid = np.max(
                config[config['group'] == 'CO_stats']['res_pc'])
            vdisp_z = (
                pt[f"F<vdisp_mol_pix_{res_fid}pc>"] *
                pt['Sigma_mol'] / Sigma_gas +
                10 * u.Unit('km s-1') *
                pt['Sigma_atom'] / Sigma_gas)
            del res_fid
        else:
            vdisp_z = np.nan
        vdisp_z_L08 = 11 * u.Unit('km s-1')
        pt["P_DE_L08"] = (
            (np.pi * const.G / 2 * Sigma_gas**2 +
             Sigma_gas * vdisp_z_L08 *
             np.sqrt(2 * const.G * pt["rho_star_midplane"])) /
            const.k_B)
        pt["P_DE_S20"] = (
            # Sun+20 Eq.12
            (np.pi * const.G / 2 * Sigma_gas**2 +
             Sigma_gas * vdisp_z *
             np.sqrt(2 * const.G * pt["rho_star_midplane"])) /
            const.k_B)

    # clean and format table
    new_table = Table(pt[list(config['colname'])])
    for row in config:
        postfix = ''
        if hasattr(pt[row['colname']], 'description'):
            if pt[row['colname']].description is not None:
                postfix = ' ' + pt[row['colname']].description
        new_table[row['colname']] = pt[row['colname']].to(row['unit'])
        new_table[row['colname']].info.format = str(row['format'])
        new_table[row['colname']].info.description = (
            str(row['description']) + postfix)
    pt.table = new_table

    # record metadata
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

    # aperture (linear) size
    aperture_size = 1 * u.kpc

    # aperture shape
    aperture_shape = 'hexagon'

    # ----------------------------------------------------------------

    # working directory
    workdir = Path(__file__).parent

    # warnings & logging settings
    if logging:
        # output log to a file
        orig_stdout = sys.stdout
        log = open(workdir/(str(Path(__file__).stem)+'.log'), 'w')
        sys.stdout = log

    # read configuration files
    config_raw = Table.read(
        workdir /
        f"config_{aperture_shape}_"
        f"{aperture_size.to('kpc').value:.0f}kpc_raw.csv")
    config_phys = Table.read(
        workdir /
        f"config_{aperture_shape}_"
        f"{aperture_size.to('kpc').value:.0f}kpc_phys.csv")
    with open(workdir / "config_params.json") as f:
        config_params = json.load(f)

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
        Rstar = row['SIZE_LSTAR_S4G'] * u.arcsec
        # logMstar = row['MSTAR_LOGMSTAR']
        # Reff = row['SIZE_W1_R50'] * u.arcsec
        # ==> changed to below to better replicate
        # the methods used in Sanchez+19 and Sanchez+14
        logMstar = np.log10(row['MSTAR_MAP']) + 0.21
        Reff = row['SIZE_LSTAR_MASS'] * 1.67 * u.arcsec

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
            f"{name}_{aperture_shape}_stats_"
            f"{aperture_size.to('kpc').value:.0f}kpc_raw.fits")
        if not rtfile.is_file():
            print(f"Constructing raw measurement table for {name}")
            gen_raw_measurement_table(
                name, gal_dist_Mpc=dist.value,
                gal_ra_deg=ra.value, gal_dec_deg=dec.value,
                gal_incl_deg=incl.value, gal_posang_deg=posang.value,
                aperture_shape=aperture_shape,
                aperture_size_kpc=aperture_size.to('kpc').value,
                config=config_raw, verbose=True, writefile=rtfile)

        # ------------------------------------------------------------
        # convert raw measurements to physical properties
        # ------------------------------------------------------------

        ptfile = (
            workdir /
            f"{name}_{aperture_shape}_stats_"
            f"{aperture_size.to('kpc').value:.0f}kpc_phys.fits")
        if (rtfile.is_file() and not ptfile.is_file()):
            print(f"Constructing physical property table for {name}")
            gen_phys_props_table(
                rtfile, gal_name=name,
                # gal_logMstar=logMstar,
                # gal_Reff_arcsec=Reff.value,
                gal_Rstar_arcsec=Rstar.value,
                config=config_phys,
                note=(
                    'PHANGS-ALMA v3.4; '
                    'PHANGS-CPROPS v3; '
                    'PHANGS-alphaCO v0.1; '
                    'PHANGS-VLA v1.0; '
                    'PHANGS-Halpha v0.1&0.3; '
                    'sample table v1.4 (dist=v1.2)'),
                version=1.1, writefile=ptfile, **config_params)

        # ------------------------------------------------------------

        ptfile_new = (
            workdir /
            f"{name}_{aperture_shape}_stats_"
            f"{aperture_size.to('kpc').value:.0f}kpc_phys.ecsv")
        if ptfile.is_file() and not ptfile_new.is_file():
            print("Converting FITS table to ECSV format")
            pt = Table.read(ptfile)
            pt.write(ptfile_new, delimiter=',', overwrite=True)

        # ------------------------------------------------------------

        print(f"Finished processing data for {name}!")
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
