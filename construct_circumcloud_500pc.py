import sys
import json
import warnings
from pathlib import Path
from itertools import product
import numpy as np
from astropy import units as u
from astropy.table import Table
from mega_table_PHANGS import (
    PhangsApertureMegaTable, get_data_path, add_columns_to_mega_table)

warnings.filterwarnings('ignore')

logging = False

# --------------------------------------------------------------------


def gen_aperture_mega_table(
        config, gal_params={}, phys_params={},
        aperture_ra_deg=None, aperture_dec_deg=None,
        aperture_size_pc=None, aperture_names=None,
        verbose=True, note='', version=0.0, writefile=''):

    aperture_size_arcsec = np.rad2deg(
        aperture_size_pc / gal_params['dist_Mpc'] / 1e6) * 3600

    # initialize table
    if verbose:
        print("  Initializing mega table")
    t = PhangsApertureMegaTable(
        aperture_ra_deg, aperture_dec_deg, aperture_size_arcsec,
        aperture_names=aperture_names)

    # add measurements to table
    add_columns_to_mega_table(
        t, config, gal_params=gal_params, phys_params=phys_params,
        verbose=verbose)

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
    t.meta['GALAXY'] = str(gal_params['name'])
    t.meta['DIST_MPC'] = gal_params['dist_Mpc']
    t.meta['RA_DEG'] = gal_params['ra_deg']
    t.meta['DEC_DEG'] = gal_params['dec_deg']
    t.meta['INCL_DEG'] = gal_params['incl_deg']
    t.meta['PA_DEG'] = gal_params['posang_deg']
    t.meta['LOGMSTAR'] = np.log10(gal_params['Mstar_Msun'])
    t.meta['RDISKKPC'] = (
        (gal_params['Rstar_arcsec'] * u.arcsec).to('rad').value *
        gal_params['dist_Mpc'] * u.Mpc).to('kpc').value
    t.meta['CO_R21'] = phys_params['CO_R21']
    t.meta['H_MOL_PC'] = phys_params['CO_full_height']
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
    aperture_size = 500 * u.pc

    # Seed catalog type
    catalog_types = ['originoise', 'homogenoise:hi', 'homogenoise:lo']

    # Seed catalog resolution
    catalog_res_pc = [60, 90, 120, 150]

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
        f"config_circumcloud_"
        f"{aperture_size.to('pc').value:.0f}pc.csv")

    # read physical parameter file
    with open(workdir / "config_params.json") as f:
        phys_params = json.load(f)

    # read PHANGS sample table
    t_sample = Table.read(get_data_path('sample'))

    # loop through sample table
    for row in t_sample:

        # galaxy parameters
        gal_params = {
            'name': row['name'].strip().upper(),
            'dist_Mpc': row['dist'],
            'ra_deg': row['orient_ra'],
            'dec_deg': row['orient_dec'],
            'incl_deg': row['orient_incl'],
            'posang_deg': row['orient_posang'],
            'Mstar_Msun': row['props_mstar'],
            'Rstar_arcsec': row['size_scalelength'],
        }

        # skip targets with bad distance
        if not gal_params['dist_Mpc'] > 0:
            print(
                f"Bad distance measurement - skipping "
                f"{gal_params['name']}")
            print("")
            continue
        # skip targets with edge-on viewing angle
        if not (0 <= gal_params['incl_deg'] <= 75.1 and
                np.isfinite(gal_params['posang_deg'])):
            print(
                f"Edge-on viewing angle - skipping "
                f"{gal_params['name']}")
            print("")
            continue

        print(f"Processing data for {gal_params['name']}")

        for ctype, res_pc in product(catalog_types, catalog_res_pc):

            # CPROPS catalog file
            cprops_file = get_data_path(
                f"ALMA:CPROPS:{ctype}",
                gal_params['name'], res_pc*u.pc)
            if not cprops_file.is_file():
                continue
            else:
                noise_level = cprops_file.stem.split('_')[-2]

            mtfile = (
                workdir /
                f"{gal_params['name']}_circumcloud_stats_"
                f"{aperture_size.to('pc').value:.0f}pc_"
                f"{noise_level}_{res_pc}pc.fits")
            if not mtfile.is_file():
                print(
                    f"Constructing mega-table ({ctype} @ {res_pc}pc)")
                t_cprops = Table.read(cprops_file)
                gen_aperture_mega_table(
                    config,
                    gal_params=gal_params, phys_params=phys_params,
                    aperture_ra_deg=t_cprops['XCTR_DEG'],
                    aperture_dec_deg=t_cprops['YCTR_DEG'],
                    aperture_size_pc=aperture_size.to('pc').value,
                    aperture_names=[
                        f"CLOUD#{i}" for i in t_cprops['CLOUDNUM']],
                    note=(
                        'PHANGS-ALMA v4; '
                        'PHANGS-VLA v1.1; '
                        'PHANGS-Halpha v0.1&0.3; '
                        'sample table v1.6'),
                    version=2.0, writefile=mtfile)

        # ------------------------------------------------------------

        print(f"Finished processing data for {gal_params['name']}!")
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
