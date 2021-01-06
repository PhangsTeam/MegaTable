import sys
import json
import warnings
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from mega_table.utils import nanaverage, deproject
from mega_table.table import TessellMegaTable
from mega_table_PHANGS import PhangsMegaTable, get_data_path

warnings.filterwarnings('ignore')

logging = False

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


def gen_tessell_mega_table(
        config, gal_params={}, phys_params={},
        aperture_shape=None, aperture_size_kpc=None,
        verbose=True, note='', version=0.0, writefile=''):

    aperture_size_arcsec = np.rad2deg(
        aperture_size_kpc / gal_params['dist_Mpc'] / 1e3) * 3600

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
    infile = get_data_path(
        'ALMA:CO:tpeak', gal_name, aperture_size_kpc*u.kpc)
    if not infile.is_file():
        if verbose:
            print(f"No CO low resolution data found for {gal_name}")
            print("")
        return
    with fits.open(infile) as hdul:
        t = PhangsTessellMegaTable(
            hdul[0].header,
            aperture_shape=aperture_shape,
            aperture_size_arcsec=aperture_size_arcsec,
            gal_ra_deg=gal_params['ra_deg'],
            gal_dec_deg=gal_params['dec_deg'])

    if 'coord' in config['group']:
        # galactic radius and projected angle
        if verbose:
            print("  Calculating galactic radius and projected angle")
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

    if 'rotcurve' in config['group']:
        # rotation curve-related quantities
        if verbose:
            print("  Calculating rotation curve-related quantities")
        for row in config[config['group'] == 'rotcurve']:
            if row['colname'] in ('V_circ', 'q_shear'):
                modelfile = get_data_path(
                    row['source'], gal_name)
                if not modelfile.is_file():
                    t[row['colname']] = np.nan * u.Unit(row['unit'])
                    continue
                if 'r_gal' not in t.colnames:
                    raise ValueError("No coordinate info found")
                r_gal_angle = (
                    t['r_gal'] / gal_dist * u.rad).to('arcsec')
                t.add_rotcurve(
                    modelfile=modelfile, r_gal_angle=r_gal_angle,
                    colname=row['colname'], unit=row['unit'])

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
            IRfile = get_data_path(
                row['source'], gal_name, row['res_pc']*u.pc)
            if not IRfile.is_file():
                t[row['colname']] = np.nan * u.Unit(row['unit'])
                continue
            t.add_Sigma_star(
                IRfile, MtoL=t['MtoL_3p4um'],
                band=band, Lsun_IR=Lsun, cosi=gal_cosi,
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
                t.add_Sigma_atom(
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
                if 'r_gal' not in t.colnames:
                    raise ValueError("No coordinate info found")
                t.add_metallicity(
                    Mstar=gal_params['Mstar_Msun']*u.Msun,
                    r_gal=t['r_gal'], Re=gal_Reff,
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
                if 'alphaCO21' not in t.colnames:
                    raise ValueError("No alphaCO column found")
                COm0file = get_data_path(
                    row['source'], gal_name, row['res_pc']*u.pc)
                if not COm0file.is_file():
                    t[row['colname']] = np.nan * u.Unit(row['unit'])
                    continue
                t.add_Sigma_mol(
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
            SFRfile = get_data_path(
                row['source'], gal_name, row['res_pc']*u.pc)
            if not SFRfile.is_file():
                t[row['colname']] = np.nan * u.Unit(row['unit'])
                continue
            t.add_Sigma_SFR(
                SFRfile=SFRfile, cosi=gal_cosi,
                colname=row['colname'], unit=u.Unit(row['unit']))
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
    else:
        orig_stdout = log = None

    # read configuration file
    config = Table.read(
        workdir /
        f"config_{aperture_shape}_"
        f"{aperture_size.to('kpc').value:.0f}kpc.csv")

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
            f"{gal_params['name']}_{aperture_shape}_stats_"
            f"{aperture_size.to('kpc').value:.0f}kpc.fits")
        if not mtfile.is_file():
            print(f"Constructing mega-table for {gal_params['name']}")
            gen_tessell_mega_table(
                config, gal_params=gal_params, phys_params=phys_params,
                aperture_shape=aperture_shape,
                aperture_size_kpc=aperture_size.to('kpc').value,
                note=(
                    'PHANGS-ALMA v3.4; '
                    'CPROPS catalogs v3.4 (ST1.2); '
                    'PHANGS-VLA v1.0; '
                    'PHANGS-Halpha v0.1&0.3; '
                    'sample table v1.4 (dist=v1.2)'),
                version=1.3, writefile=mtfile)

        # ------------------------------------------------------------

        # mtfile_new = (
        #     workdir /
        #     f"{gal_params['name']}_{aperture_shape}_stats_"
        #     f"{aperture_size.to('kpc').value:.0f}kpc.ecsv")
        # if mtfile.is_file() and not mtfile_new.is_file():
        #     print("Converting FITS table to ECSV format")
        #     t = Table.read(mtfile)
        #     t.write(mtfile_new, delimiter=',', overwrite=True)

        # ------------------------------------------------------------

        print(f"Finished processing data for {gal_params['name']}!")
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
