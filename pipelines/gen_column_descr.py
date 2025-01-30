import json
import warnings
from pathlib import Path
from astropy.table import Table

###############################################################################

# location of all relevant config files
config_dir = Path('/data/bell-kant/sun.1608/PHANGS/mega-tables/code')

# location to save the output data tables
work_dir = Path('/data/bell-kant/sun.1608/PHANGS/mega-tables')

# example galaxy to extract column information from
gal_name = 'NGC0628'

###############################################################################


def gen_column_descr_file(tablefile, writefile=None, **kwargs):

    t = Table.read(tablefile)
    t_info = Table()
    t_info['name'] = t.colnames
    t_info['unit'] = ' ' * 100
    t_info['description'] = ' ' * 200
    for row in t_info:
        if t[row['name']].unit is not None:
            row['unit'] = t[row['name']].unit.to_string()
        row['description'] = t[row['name']].info.description

    if writefile is not None:
        t_info.write(
            writefile, overwrite=True, format='ascii.rst', **kwargs)
        return writefile
    else:
        return t_info
    

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # warning and logging settings
    warnings.simplefilter('ignore', RuntimeWarning)

    with open(config_dir / "config_tables.json") as f:
        table_configs = json.load(f)
    ver_str = 'v' + str(table_configs['table_version']).replace('.', 'p')

    # TessellMegaTable
    tile_shape = table_configs['tessell_tile_shape']
    tile_size_str = (
        str(table_configs['tessell_tile_size']).replace('.', 'p') +
        table_configs['tessell_tile_size_unit'])
    for content in ['base', 'phangsalma', 'phangsmuse', 'gauss']:
        tessell_table_file = (
            work_dir / table_configs['tessell_table_name'].format(
                galaxy=gal_name, content=content,
                tile_shape=tile_shape, tile_size_str=tile_size_str))
        tessell_info_file = (
            work_dir / f"{ver_str}_column_descr_{content}_{tile_shape}.rst")
        gen_column_descr_file(
            tessell_table_file, writefile=tessell_info_file)

    # RadialMegaTable
    annulus_width_str = (
        str(table_configs['radial_annulus_width']).replace('.', 'p') +
        table_configs['radial_annulus_width_unit'])
    for content in ['base', 'phangsalma', 'phangsmuse']:
        radial_table_file = (
            work_dir / table_configs['radial_table_name'].format(
                galaxy=gal_name, content=content,
                annulus_width_str=annulus_width_str))
        radial_info_file = (
            work_dir / f"{ver_str}_column_descr_{content}_annulus.rst")
        gen_column_descr_file(
            radial_table_file, writefile=radial_info_file)