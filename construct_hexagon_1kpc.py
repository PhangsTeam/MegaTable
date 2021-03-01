import sys
import json
import warnings
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.table import Table
from mega_table_PHANGS import (
    PhangsTessellMegaTable, get_data_path, add_columns_to_mega_table)

warnings.filterwarnings('ignore')

logging = False

# --------------------------------------------------------------------


def gen_tessell_mega_table(
        config, gal_params={}, phys_params={},
        tile_shape=None, tile_size_kpc=None, fov_radius_R25=None,
        verbose=True, note='', version=0.0, writefile=''):

    tile_size_arcsec = np.rad2deg(
        tile_size_kpc / gal_params['dist_Mpc'] / 1e3) * 3600
    fov_radius_arcsec = fov_radius_R25 * gal_params['R25_arcsec']

    # initialize table
    if verbose:
        print("  Initializing mega table")
    t = PhangsTessellMegaTable(
        gal_params['ra_deg'], gal_params['dec_deg'],
        fov_radius_arcsec, tile_size_arcsec,
        tile_shape=tile_shape)

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
    t.meta['R25_KPC'] = (
        (gal_params['R25_arcsec'] * u.arcsec).to('rad').value *
        gal_params['dist_Mpc'] * u.Mpc).to('kpc').value
    t.meta['RDISKKPC'] = (
        (gal_params['Rstar_arcsec'] * u.arcsec).to('rad').value *
        gal_params['dist_Mpc'] * u.Mpc).to('kpc').value
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

    # tile (linear) size
    tile_size = 1 * u.kpc

    # tile shape
    tile_shape = 'hexagon'

    # FoV radius (relative to R25)
    fov_radius_R25 = 1.5

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
        f"config_{tile_shape}_"
        f"{tile_size.to('kpc').value:.0f}kpc.csv")

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
            'R25_arcsec': row['size_r25'],
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

        mtfile = (
            workdir /
            f"{gal_params['name']}_{tile_shape}_stats_"
            f"{tile_size.to('kpc').value:.0f}kpc.fits")
        if not mtfile.is_file():
            print(f"Constructing mega-table for {gal_params['name']}")
            gen_tessell_mega_table(
                config, gal_params=gal_params, phys_params=phys_params,
                tile_shape=tile_shape,
                tile_size_kpc=tile_size.to('kpc').value,
                fov_radius_R25=fov_radius_R25,
                note=(
                    'PHANGS-ALMA v3.4; '
                    'CPROPS catalogs v3.4; '
                    'PHANGS-VLA v1.0; '
                    'PHANGS-Halpha v0.1&0.3; '
                    'sample table v1.6 (dist=v1.2)'),
                version=1.4, writefile=mtfile)

        # ------------------------------------------------------------

        print(f"Finished processing data for {gal_params['name']}!")
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
