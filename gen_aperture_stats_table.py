import os
import sys
from pathlib import Path
import warnings
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from AlmaTools.utils import deproject
from mega_table.utils import VoronoiTessTable

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
            fname_seq = [galname, 'mask'] + datatypes[2:]
        else:
            fname_seq = [galname, 'S4G'] + datatypes[1:]
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
    PHANGSdir = Path(os.getenv('PHANGSWORKDIR'))
    workdir = PHANGSdir / 'mega-tables'
    catfile = PHANGSdir / 'sample_v1p3.ecsv'
    logfile = workdir / (str(Path(__file__).stem)+'.log')
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
    catalog = catalog[catalog['ALMA'] == 1]

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
                   f"{name}_hex_{apersz.to('kpc').value:.0f}kpc.ecsv")
        # skip targets with aperture statistics table already on disk
        if vttfile.is_file():
            continue

        print(f"Processing data for {name}")

        # initialize a VoronoiTessTable
        infile = get_data_path(
            'ALMA:CO:tpeak', name, lin_res=apersz)
        if not infile.is_file():
            print(f"No CO low resolution data found")
            continue
        apersz_deg = (apersz/dist*u.rad).to('deg').value
        with fits.open(infile) as hdul:
            vtt = VoronoiTessTable(
                hdul[0].header, cell_shape='hexagon',
                ref_radec=ctr_radec.value,
                seed_spacing=apersz_deg)

        # add galactic radii and projected angles in table
        radii, projang = deproject(
            center_coord=ctr_radec, incl=incl.value, pa=posang.value,
            ra=vtt['RA'], dec=vtt['DEC'])
        vtt['r_gal'] = (radii * u.deg).to('arcsec')
        vtt['phi_gal'] = projang * u.deg
        # sort rows by galactic radii
        vtt[:] = vtt[np.argsort(vtt['r_gal'])]

        # add low resolution CO data in table
        infile = get_data_path(
            'ALMA:CO:mom0:strict', name, lin_res=apersz)
        vtt.resample_image(
            infile, colname='I_CO21',
            unit='header', #unit=u.Unit('K km s-1'),
            fill_outside=np.nan)
        # mask rows where low resolution 'I_CO21' is NaN
        vtt.clean(discard_NaN='I_CO21')
        if len(vtt) == 0:
            continue

        # add HI data in table
        infile = get_data_path(
            'HI:mom0', name, lin_res=apersz)
        if not infile.is_file():
            vtt['I_21cm'] = np.full(len(vtt), np.nan)
        else:
            vtt.resample_image(
                infile, colname='I_21cm',
                unit='header', #unit=u.Unit('K km s-1'),
                fill_outside=np.nan)
        infile = get_data_path('HI:mom0', name)
        if not infile.is_file():
            vtt['I_21cm_natres'] = np.full(len(vtt), np.nan)
        else:
            vtt.resample_image(
                infile, colname='I_21cm_natres',
                unit='header', #unit=u.Unit('K km s-1'),
                fill_outside=np.nan)

        # add S4G data in table
        infile = get_data_path(
            'S4G:ICA3p6um', name, lin_res=apersz)
        if not infile.is_file():
            vtt['I_3p6um_ICA'] = np.full(len(vtt), np.nan)
        else:
            vtt.resample_image(
                infile, colname='I_3p6um_ICA',
                unit='header', #unit=u.Unit('Msun kpc-2 yr-1'),
                fill_outside=np.nan)
        infile = get_data_path(
            'S4G:3p6um', name, lin_res=apersz)
        if not infile.is_file():
            vtt['I_3p6um_raw'] = np.full(len(vtt), np.nan)
        else:
            vtt.resample_image(
                infile, colname='I_3p6um_raw',
                unit='header', #unit=u.Unit('MJy sr-1'),
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
        if not infile.is_file():
            vtt['SigSFR_W3ONLY'] = np.full(len(vtt), np.nan)
        else:
            vtt.resample_image(
                infile, colname='SigSFR_W3ONLY',
                unit='header', #unit=u.Unit('Msun kpc-2 yr-1'),
                fill_outside=np.nan)

        # write aperture statistics table to disk
        vtt.write(vttfile)

    if logging:
        sys.stdout = orig_stdout
        log.close()
