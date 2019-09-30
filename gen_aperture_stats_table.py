import os
import sys
from pathlib import Path
import warnings
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from .utils import VoronoiTessTable

# --------------------------------------------------------------------


def get_data_path(datatype, galname=None, lin_res=None):
    """
    Get the path to any required data on disk.
    """
    datatypes = datatype.split(':')

    # PHANGS data parent directory
    PHANGSdir = Path(os.getenv('PHANGSWORKDIR'))

    if datatypes[0] == 'ALMA':
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
            fname_seq = [galname, 'mask'] + datatype[2:]
        else:
            fname_seq = [galname] + datatype[1:]
            if lin_res is not None:
                fname_seq += [f"{lin_res.to('pc').value:.0f}pc"]

    return basedir / ('_'.join(fname_seq) + '.fits')


# --------------------------------------------------------------------


def add_env_frac_to_table():
    pass


# --------------------------------------------------------------------


def add_CO_stat_to_table():
    pass


# --------------------------------------------------------------------


def add_CPROPS_stat_to_table():
    pass


# --------------------------------------------------------------------


if __name__ == '__main__':

    # ----------------------------------------------------------------

    # (linear) resolutions of the PHANGS-ALMA data
    lin_res = np.array([60, 90, 120, 150]) * u.pc

    # list of morphological regions in environmental masks
    regions = ('disk', 'bulge', 'bars', 'rings', 'lenses', 'sp_arms')

    # working directory
    WORKDIR = PHANGSDIR / 'mega-tables'
    catfile = WORKDIR / 'catalog.ecsv'
    logfile = WORKDIR / Path(__file__).stem+'.log'
    logging = False

    # (linear) size of the averaging apertures
    apersz = 1 * u.kpc

    # ----------------------------------------------------------------

    # ignore warnings
    warnings.filterwarnings('ignore')

    if logging:
        # output log to a file
        orig_stdout = sys.stdout
        log = open(logfile, 'w')
        sys.stdout = log

    catalog = Table.read(catfile)

    for row in catalog:

        # galaxy parameters
        name = row['NAME'].strip()
        dist = row['DIST'] * u.Mpc
        ctr_radec = np.array([row['RA_DEG'], row['DEC_DEG']]) * u.deg

        # skip "bad" targets
        if name in (
            # Excluding edge-on objects:
            'NGC0253', 'NGC0891', 'NGC3623', 'NGC4565', 'NGC4945',
            # # Excluding objects with P.A. = NaN
            # 'NGC0278', 'NGC3239', 'NGC3344', 'NGC3599',
            ):
            continue
        if (row['CO_SURVEY'] == '-'):
            continue

        print(f"Processing data for {name}")

        # initialize a VoronoiTessTable
        infile = get_data_path(
            'ALMA:CO:tpeak', name, lin_res=apersz)
        with fits.open(infile) as hdul:
            vtt = VoronoiTessTable(
                hdul[0].header, cell_shape='hexagon',
                ref_radec=ctr_radec.value,
                seed_spacing=(apersz/dist*u.rad).to('deg').value)

        # add low resolution CO data in table
        infile = get_data_path(
            'ALMA:CO:mom0:strict', name, lin_res=apersz)
        vtt.resample_image(
            infile, colname='I_'+line,
            unit='header', #unit=u.Unit('K cm-3'),
            fill_outside=np.nan)

        # add HI data in table
        infile = get_data_path(
            'HI:mom0', name, lin_res=apersz)
        if not infile.is_file():
            vtt['I_21cm'] = np.full(len(vtt), np.nan)
        else:
            vtt.resample_image(
                infile, colname='I_21cm',
                unit='header', #unit=u.Unit('K cm-3'),
                fill_outside=np.nan)
        infile = get_data_path('HI:mom0', name)
        vtt.resample_image(
            infile, colname='I_21cm_native',
            unit='header', #unit=u.Unit('K cm-3'),
            fill_outside=np.nan)

        # add Z0MGS data in table
        infile = get_data_path(
            'Z0MGS:SFR:NUVW3', name, lin_res=apersz)
        if not infile.is_file():
            vtt['SigSFR_NUVW3'] = np.full(len(vtt), np.nan)
        else:
            vtt.resample_image(
                infile, colname='SigSFR_NUVW3',
                unit='header', #unit=u.Unit('Msun kpc-2 yr-1'),
                fill_outside=np.nan)
        infile = get_data_path(
            'Z0MGS:SFR:W3ONLY', name, lin_res=apersz)
        vtt.resample_image(
            infile, colname='SigSFR_W3ONLY',
            unit='header', #unit=u.Unit('Msun kpc-2 yr-1'),
            fill_outside=np.nan)

        # add S4G data in table
        infile = get_data_path(
            'S4G:stellar', name, lin_res=apersz)
        vtt.resample_image(
            infile, colname='I_stellar',
            unit='header', #unit=u.Unit('mJy sr-1'),
            fill_outside=np.nan)

    if logging:
        sys.stdout = orig_stdout
        log.close()
