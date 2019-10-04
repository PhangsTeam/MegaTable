import os
import sys
from pathlib import Path
import warnings
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from reproject import reproject_interp
from mega_table.table import VoronoiTessTable
from mega_table.utils import deproject

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
    vtt, infile, colname='new_col', **kwargs):

    if not infile.is_file():
        vtt[colname] = np.full(len(vtt), np.nan)
        return

    vtt.resample_image(infile, colname=colname, **kwargs)

    return


# --------------------------------------------------------------------


def add_env_frac_to_table(
    vtt, envfile, wtfile, colname='new_col', **kwargs):

    if not (envfile.is_file() and wtfile.is_file()):
        vtt[colname] = np.full(len(vtt), np.nan)
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
    vtt.calc_image_stats(
        envbimap, header=wthdr, weight=wtmap,
        colname=colname, **kwargs)

    return


# --------------------------------------------------------------------


def add_CO_stat_to_table(
    vtt, bm0file, sm0file, sewfile, res, **kwargs):

    rstr = f"{res.to('pc').value:.0f}pc"
    cols = [
        # sums
        (f"Area_CO21_total_{rstr}", 'arcsec2'),
        (f"Area_CO21_broad_{rstr}", 'arcsec2'),
        (f"Area_CO21_strict_{rstr}", 'arcsec2'),
        (f"Flux_CO21_broad_{rstr}", 'K km s-1 arcsec2'),
        (f"Flux_CO21_strict_{rstr}", 'K km s-1 arcsec2'),
        # area weighted averages
        (f"A<I_CO21_{rstr}>", 'K km s-1'),
        (f"A<I^2_CO21_{rstr}>", 'K2 km2 s-2'),
        (f"A<vdisp_CO21_{rstr}>", 'km s-1'),
        (f"A<vdisp^2_CO21_{rstr}>", 'km2 s-2'),
        (f"A<I*vdisp^2_CO21_{rstr}>", 'K km3 s-3'),
        (f"A<vdisp^2/I_CO21_{rstr}>", 'km s-1 K-1'),
        # CO flux weighted averages
        (f"F<I_CO21_{rstr}>", 'K km s-1'),
        (f"F<I^2_CO21_{rstr}>", 'K2 km2 s-2'),
        (f"F<vdisp_CO21_{rstr}>", 'km s-1'),
        (f"F<vdisp^2_CO21_{rstr}>", 'km2 s-2'),
        (f"F<I*vdisp^2_CO21_{rstr}>", 'K km3 s-3'),
        (f"F<vdisp^2/I_CO21_{rstr}>", 'km s-1 K-1'),
        ]

    # no data file found: add placeholder (empty) columns to table
    if not (bm0file.is_file() and sm0file.is_file() and
            sewfile.is_file()):
        for col, unit in cols:
            vtt[col] = np.full(len(vtt), np.nan)
        return

    # read files
    with fits.open(bm0file) as hdul:
        bm0map = hdul[0].data.copy()
        hdr = hdul[0].header.copy()
    with fits.open(sm0file) as hdul:
        sm0map = hdul[0].data.copy()
        if sm0map.shape != bm0map.shape:
            raise ValueError("Input maps have inconsistent shape")
    with fits.open(sewfile) as hdul:
        sewmap = hdul[0].data.copy()
        if sewmap.shape != bm0map.shape:
            raise ValueError("Input maps have inconsistent shape")
    bm0map[~np.isfinite(bm0map)] = 0
    nanmap = np.ones_like(bm0map).astype('float')
    nanmap[~np.isfinite(sm0map) | (sm0map <= 0) |
           ~np.isfinite(sewmap) | (sewmap <= 0)] = np.nan
    sm0map[np.isnan(nanmap)] = 0
    sewmap[np.isnan(nanmap)] = 0

    # pixel size (in arcsec^2)
    pixsz = np.abs(
        hdr['CDELT1'] * hdr['CDELT2'] * u.deg**2
        ).to('arcsec2').value

    # maps corresponding to each column
    maps = [
        # sums
        np.ones_like(bm0map).astype('float')*pixsz,
        (bm0map != 0).astype('float')*pixsz,
        (sm0map > 0).astype('float')*pixsz,
        bm0map*pixsz,
        sm0map*pixsz,
        # area weighted averages (among regions w/ CO detection)
        sm0map * nanmap,
        sm0map**2 * nanmap,
        sewmap * nanmap,
        sewmap**2 * nanmap,
        sm0map*sewmap**2 * nanmap,
        sewmap**2/sm0map * nanmap,
        # CO flux weighted averages
        sm0map,
        sm0map**2,
        sewmap,
        sewmap**2,
        sm0map*sewmap**2,
        sewmap**2/sm0map,
    ]

    # calculate statistics and add into table
    for (col, unit), map in zip(cols, maps):
        if col[:2] == 'A<':
            # area-weighted average
            vtt.calc_image_stats(
                map, header=hdr, weight=None,
                colname=col, unit=unit, **kwargs)
        elif col[:2] == 'F<':
            # CO flux-weighted average
            vtt.calc_image_stats(
                map, header=hdr, weight=sm0map,
                colname=col, unit=unit, **kwargs)
        else:
            # sum
            vtt.calc_image_stats(
                map, header=hdr, stat_func=np.nansum,
                colname=col, unit=unit, **kwargs)

    return


# --------------------------------------------------------------------


def add_CPROPS_stat_to_table():
    pass


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

    # loop through the sample table
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
        vtt['r_gal'] = (radii * u.deg).to('arcsec')
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
            unit='header', #unit=u.Unit('K km s-1'),
            fill_outside=np.nan)
        infile = get_data_path('HI:mom0', name)
        add_resampled_image_to_table(
            vtt, infile, colname='I_21cm_native',
            unit='header', #unit=u.Unit('K km s-1'),
            fill_outside=np.nan)

        # add S4G data in table
        print("  Resampling S4G data")
        infile = get_data_path('S4G:ICA3p6um', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='I_3p6um_ICA',
            unit='header', #unit=u.Unit('MJy sr-1'),
            fill_outside=np.nan)
        infile = get_data_path('S4G:3p6um', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='I_3p6um_raw',
            unit='header', #unit=u.Unit('MJy sr-1'),
            fill_outside=np.nan)

        # add Z0MGS data in table
        print("  Resampling Z0MGS data")
        infile = get_data_path('Z0MGS:SFR:NUVW3', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='SigSFR_NUVW3',
            unit='header', #unit=u.Unit('Msun kpc-2 yr-1'),
            fill_outside=np.nan)
        infile = get_data_path('Z0MGS:SFR:W3ONLY', name, apersz)
        add_resampled_image_to_table(
            vtt, infile, colname='SigSFR_W3ONLY',
            unit='header', #unit=u.Unit('Msun kpc-2 yr-1'),
            fill_outside=np.nan)

        # add environmental fraction (CO flux weighted) in table
        print("  Calculating (CO flux weighted) "
              "environmental fraction")
        res = lin_res[-1]
        wtfile = get_data_path('ALMA:CO:mom0:strict', name, res)
        for reg in regions:
            envfile = get_data_path('S4G:env_mask:'+reg, name)
            add_env_frac_to_table(
                vtt, envfile, wtfile, colname='frac_'+reg)

        # add statistics of high resolution CO data in table
        print("  Calculating statistics of high resolution CO data")
        for res in lin_res:
            bm0file = get_data_path('ALMA:CO:mom0:broad', name, res)
            sm0file = get_data_path('ALMA:CO:mom0:strict', name, res)
            sewfile = get_data_path('ALMA:CO:ew:strict', name, res)
            add_CO_stat_to_table(
                vtt, bm0file, sm0file, sewfile, res)

        # mask rows where low resolution 'I_CO21' is NaN
        # This step has to be the last step!!
        vtt.clean(discard_NaN='I_CO21')

        # write table to disk
        print("  Writing table to disk")
        vtt.write(vttfile)

        print(f"Finished processing data for {name}!")
        print("")

    # ----------------------------------------------------------------

    if logging:
        # shift back to original log output location
        sys.stdout = orig_stdout
        log.close()
