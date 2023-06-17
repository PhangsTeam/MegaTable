import sys
import json
import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table, join

from mega_table.table import TessellMegaTable, RadialMegaTable

###############################################################################

# location of all relevant config files
config_dir = Path('/data/kant/0/sun.1608/PHANGS/mega-tables/code')

# location to save the output data tables
work_dir = Path('/data/kant/0/sun.1608/PHANGS/mega-tables')

# logging setting
logging = False

###############################################################################


def combine_tables(t, *extra_ts, writefile=None, verbose=True):

    for extra_t in extra_ts:
        # remove timestamp to avoid metadata conflict
        extra_t.meta.pop('TIMESTMP')
        # discard repeated columns
        colmask = ~np.isin(
            extra_t.colnames, t.colnames, assume_unique=True)
        newcols = ['ID'] + list(np.array(extra_t.colnames)[colmask])
        # join tables
        t.table = join(
            t.table, extra_t.table[newcols], keys=['ID'], join_type='left')

    if writefile is not None:
        if verbose:
            print("  Write table")
        if Path(writefile).suffix == '.ecsv':
            t.write(
                writefile, add_timestamp=True,
                delimiter=',', overwrite=True)
        else:
            t.write(writefile, add_timestamp=True, overwrite=True)
        return writefile
    else:
        return t


# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # warning and logging settings
    warnings.simplefilter('ignore', RuntimeWarning)
    if logging:
        # output log to a file
        orig_stdout = sys.stdout
        log = open(work_dir / (str(Path(__file__).stem) + '.log'), 'w')
        sys.stdout = log
    else:
        orig_stdout = log = None

    with open(config_dir / "config_data_path.json") as f:
        data_paths = json.load(f)
    with open(config_dir / "config_tables.json") as f:
        table_configs = json.load(f)

    # read sample table
    t_sample = Table.read(data_paths['PHANGS_sample_table'])

    # sub-select sample
    t_sample = t_sample[t_sample['data_has_megatable']]

    # loop through all galaxies
    for row in t_sample:

        # extract galaxy name
        gal_name = row['name'].upper()

        print("\n############################################################")
        print(f"# {gal_name}")
        print("############################################################\n")

        # TessellMegaTable
        tile_shape = table_configs['tessell_tile_shape']
        tile_size_str = (
            str(table_configs['tessell_tile_size']).replace('.', 'p') +
            table_configs['tessell_tile_size_unit'])

        tessell_combined_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content='combined',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        tessell_base_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content='base',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        tessell_alma_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content='phangsalma',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        tessell_muse_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content='phangsmuse',
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        if (not tessell_combined_table_file.is_file() and
                tessell_base_table_file.is_file() and
                tessell_alma_table_file.is_file() and
                tessell_muse_table_file.is_file()):
            print("Combining all tessellation statistics tables...")
            t_base = TessellMegaTable.read(tessell_base_table_file)
            t_alma = TessellMegaTable.read(tessell_alma_table_file)
            t_muse = TessellMegaTable.read(tessell_muse_table_file)
            combine_tables(
                t_base, t_alma, t_muse,
                writefile=tessell_combined_table_file)
            # remove PHANGS-ALMA table if empty
            if np.isfinite(t_alma['I_CO21']).sum() == 0:
                print("Remove empty PHANGS-ALMA table")
                tessell_alma_table_file.unlink()
            # remove PHANGS-MUSE table if empty
            if np.isfinite(t_muse['I_Halpha_DAP']).sum() == 0:
                print("Remove empty PHANGS-MUSE table")
                tessell_muse_table_file.unlink()
            print("Done\n")

        # RadialMegaTable
        annulus_width_str = (
            str(table_configs['radial_annulus_width']).replace('.', 'p') +
            table_configs['radial_annulus_width_unit'])

        radial_combined_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content='combined',
                annulus_width_str=annulus_width_str))
        radial_base_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content='base',
                annulus_width_str=annulus_width_str))
        radial_alma_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content='phangsalma',
                annulus_width_str=annulus_width_str))
        radial_muse_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content='phangsmuse',
                annulus_width_str=annulus_width_str))
        if (not radial_combined_table_file.is_file() and
                radial_base_table_file.is_file() and
                radial_alma_table_file.is_file() and
                radial_muse_table_file.is_file()):
            print("Combining all radial statistics tables...")
            t_base = RadialMegaTable.read(radial_base_table_file)
            t_alma = RadialMegaTable.read(radial_alma_table_file)
            t_muse = RadialMegaTable.read(radial_muse_table_file)
            combine_tables(
                t_base, t_alma, t_muse,
                writefile=radial_combined_table_file)
            # remove PHANGS-ALMA table if empty
            if np.isfinite(t_alma['I_CO21']).sum() == 0:
                print("Remove empty PHANGS-ALMA table")
                radial_alma_table_file.unlink()
            # remove PHANGS-MUSE table if empty
            if np.isfinite(t_muse['I_Halpha_DAP']).sum() == 0:
                print("Remove empty PHANGS-MUSE table")
                radial_muse_table_file.unlink()
            print("Done\n")

    # logging settings
    if logging:
        sys.stdout = orig_stdout
        log.close()
