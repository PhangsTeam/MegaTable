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
from mega_table.utils import deproject, nanaverage

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
    t, infile, colname='new_col', **kwargs):

    if not infile.is_file():
        t[colname] = np.full(len(t), np.nan)
        return

    t.resample_image(infile, colname=colname, **kwargs)

    return


# --------------------------------------------------------------------


def add_env_frac_to_table(
    t, envfile, wtfile, colname='new_col', **kwargs):

    if not (envfile.is_file() and wtfile.is_file()):
        t[colname] = np.full(len(t), np.nan)
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
    t.calc_image_stats(
        envbimap, header=wthdr, stat_func=nanaverage, weight=wtmap,
        colname=colname, **kwargs)

    return


# --------------------------------------------------------------------


def add_CO_stats_to_table(
    t, bm0file, sm0file, sewfile, res, **kwargs):

    rstr = f"{res.to('pc').value:.0f}pc"
    cols = [
        # sums
        (f"Area_CO21_total_{rstr}", 'arcsec2'),
        (f"Area_CO21_broad_{rstr}", 'arcsec2'),
        (f"Area_CO21_strict_{rstr}", 'arcsec2'),
        (f"Flux_CO21_broad_{rstr}", 'K km s-1 arcsec2'),
        (f"Flux_CO21_strict_{rstr}", 'K km s-1 arcsec2'),
        # area-weighted averages
        (f"A<I_CO21_{rstr}>", 'K km s-1'),
        (f"A<I^2_CO21_{rstr}>", 'K2 km2 s-2'),
        (f"A<sigv_CO21_{rstr}>", 'km s-1'),
        (f"A<sigv^2_CO21_{rstr}>", 'km2 s-2'),
        (f"A<I*sigv^2_CO21_{rstr}>", 'K km3 s-3'),
        (f"A<sigv^2/I_CO21_{rstr}>", 'km s-1 K-1'),
        # CO flux-weighted averages
        (f"F<I_CO21_{rstr}>", 'K km s-1'),
        (f"F<I^2_CO21_{rstr}>", 'K2 km2 s-2'),
        (f"F<sigv_CO21_{rstr}>", 'km s-1'),
        (f"F<sigv^2_CO21_{rstr}>", 'km2 s-2'),
        (f"F<I*sigv^2_CO21_{rstr}>", 'K km3 s-3'),
        (f"F<sigv^2/I_CO21_{rstr}>", 'km s-1 K-1'),
        ]

    # if no data file found: add placeholder (empty) columns
    if not (bm0file.is_file() and sm0file.is_file() and
            sewfile.is_file()):
        for col, unit in cols:
            t[col] = np.full(len(t), np.nan)
        return

    # read files
    with fits.open(bm0file) as hdul:
        bm0_map = hdul[0].data.copy()
        hdr = hdul[0].header.copy()
    with fits.open(sm0file) as hdul:
        sm0_map = hdul[0].data.copy()
        if sm0_map.shape != bm0_map.shape:
            raise ValueError("Input maps have inconsistent shape")
    with fits.open(sewfile) as hdul:
        sew_map = hdul[0].data.copy()
        if sew_map.shape != bm0_map.shape:
            raise ValueError("Input maps have inconsistent shape")
    bm0_map[~np.isfinite(bm0_map)] = 0
    nan_map = np.ones_like(bm0_map).astype('float')
    nan_map[~np.isfinite(sm0_map) | (sm0_map <= 0) |
           ~np.isfinite(sew_map) | (sew_map <= 0)] = np.nan
    sm0_map[np.isnan(nan_map)] = 0
    sew_map[np.isnan(nan_map)] = 0

    # pixel size (in arcsec^2)
    pixsz = np.abs(
        hdr['CDELT1'] * hdr['CDELT2'] * u.deg**2
        ).to('arcsec2').value

    # maps corresponding to each column
    maps = [
        # sums
        np.ones_like(bm0_map).astype('float')*pixsz,
        (bm0_map != 0).astype('float')*pixsz,
        (sm0_map > 0).astype('float')*pixsz,
        bm0_map*pixsz,
        sm0_map*pixsz,
        # area-weighted averages (among regions w/ CO detection)
        sm0_map * nan_map,
        sm0_map**2 * nan_map,
        sew_map * nan_map,
        sew_map**2 * nan_map,
        sm0_map*sew_map**2 * nan_map,
        sew_map**2/sm0_map * nan_map,
        # CO flux-weighted averages
        sm0_map,
        sm0_map**2,
        sew_map,
        sew_map**2,
        sm0_map*sew_map**2,
        sew_map**2/sm0_map,
    ]

    # calculate statistics and add into table
    for (col, unit), map in zip(cols, maps):
        if col[:2] == 'A<':
            # area-weighted average
            t.calc_image_stats(
                map, header=hdr, stat_func=nanaverage, weight=None,
                colname=col, unit=unit, **kwargs)
        elif col[:2] == 'F<':
            # CO flux-weighted average
            t.calc_image_stats(
                map, header=hdr, stat_func=nanaverage, weight=sm0_map,
                colname=col, unit=unit, **kwargs)
        else:
            # sum
            t.calc_image_stats(
                map, header=hdr, stat_func=np.nansum,
                colname=col, unit=unit, **kwargs)

    return


# --------------------------------------------------------------------


def add_cprops_stats_to_table(
    t, cpropsfile, res, **kwargs):

    rstr = f"{res.to('pc').value:.0f}pc"
    cols = [
        # sums
        (f"Ncloud_CPROPS_total_{rstr}", ''),
        (f"Flux_CPROPS_total_{rstr}", 'K km s-1 arcsec2'),
        # uniformly weighted averages
        (f"U<F_CPROPS_{rstr}>", 'K km s-1 arcsec2'),
        (f"U<R_CPROPS_{rstr}>", 'arcsec'),
        (f"U<F/R^2_CPROPS_{rstr}>", 'K km s-1'),
        (f"U<F^2/R^4_CPROPS_{rstr}>", 'K2 km2 s-2'),
        (f"U<sigv_CPROPS_{rstr}>", 'km s-1'),
        (f"U<sigv^2_CPROPS_{rstr}>", 'km2 s-2'),
        (f"U<F*sigv^2/R^3_CPROPS_{rstr}>", 'K km3 s-3 arcsec-1'),
        (f"U<R*sigv^2/F_CPROPS_{rstr}>", 'km s-1 K-1 arcsec-1'),
        # CO flux-weighted averages
        (f"F<F_CPROPS_{rstr}>", 'K km s-1 arcsec2'),
        (f"F<R_CPROPS_{rstr}>", 'arcsec'),
        (f"F<I_CPROPS_{rstr}>", 'K km s-1'),
        (f"F<I^2_CPROPS_{rstr}>", 'K2 km2 s-2'),
        (f"F<sigv_CPROPS_{rstr}>", 'km s-1'),
        (f"F<sigv^2_CPROPS_{rstr}>", 'km2 s-2'),
        (f"F<F*sigv^2/R^3_CPROPS_{rstr}>", 'K km3 s-3 arcsec-1'),
        (f"F<R*sigv^2/F_CPROPS_{rstr}>", 'km s-1 K-1 arcsec-1'),
    ]

    # if no CPROPS file found: add placeholder (empty) columns
    if not cpropsfile.is_file():
        for col, unit in cols:
            t[col] = np.full(len(t), np.nan)
        return

    # read CPROPS file
    try:
        t_cat = Table.read(catfile)
    except ValueError as e:
        print(e)
        return
    ra_cat = t_cat['XCTR_DEG'].quantity.value
    dec_cat = t_cat['YCTR_DEG'].quantity.value
    flux_cat = (
        t_cat['FLUX_KKMS_PC2'] / t_cat['DISTANCE_PC']**2 *
        u.Unit('K km s-1 sr')).to('K km s-1 arcsec2').value
    sigv_cat = t_cat['SIGV_KMS'].quantity.value
    radius_cat = (  # expressed in Solomon+87 convention (w/ 1.91)
        t_cat['RAD_PC'] / t_cat['DISTANCE_PC'] *
        u.rad).to('arcsec').value
    # radius_cat = (  # expressed in terms of Gaussian HWHM
    #     t_cat['RAD_PC'] / t_cat['DISTANCE_PC'] / t['RMSTORAD'] *
    #     np.sqrt(2*np.log(2)) * u.rad).to('arcsec').value
    wt_arr = flux_cat.copy()
    wt_arr[~np.isfinite(radius_cat)] = 0
    flux_cat[~np.isfinite(radius_cat)] = np.nan
    sigv_cat[~np.isfinite(radius_cat)] = np.nan
    rad_cat[~np.isfinite(radius_cat)] = np.nan

    # entries corresponding to each column
    entries = [
        # sums
        np.isfinite(flux_cat).astype('int'),
        flux_cat,
        # uniformly weighted averages
        flux_cat,
        rad_cat,
        flux_cat/rad_cat**2,
        flux_cat**2/rad_cat**4,
        sigv_cat,
        sigv_cat**2,
        flux_cat*sigv_cat**2/rad_cat**3,
        rad_cat*sigv_cat**2/flux_cat,
        # CO flux-weighted averages
        flux_cat,
        rad_cat,
        flux_cat/rad_cat**2,
        flux_cat**2/rad_cat**4,
        sigv_cat,
        sigv_cat**2,
        flux_cat*sigv_cat**2/rad_cat**3,
        rad_cat*sigv_cat**2/flux_cat,
    ]

    # calculate statistics and add into table
    for (col, unit), entry in zip(cols, entries):
        if col[:2] == 'U<':
            # area-weighted average
            t.calc_catalog_stats(
                entry, ra_cat, dec_cat,
                stat_func=nanaverage, weight=None,
                colname=col, unit=unit, **kwargs)
        elif col[:2] == 'F<':
            # CO flux-weighted average
            t.calc_catalog_stats(
                entry, ra_cat, dec_cat,
                stat_func=nanaverage, weight=wt_arr,
                colname=col, unit=unit, **kwargs)
        else:
            # sum
            t.calc_catalog_stats(
                entry, ra_cat, dec_cat, stat_func=np.nansum,
                colname=col, unit=unit, **kwargs)

    return


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

        # add environmental fraction (CO flux-weighted) in table
        print("  Calculating (CO flux-weighted) "
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
            print(f"    @ {res.to('pc').value:.0f}pc")
            bm0file = get_data_path('ALMA:CO:mom0:broad', name, res)
            sm0file = get_data_path('ALMA:CO:mom0:strict', name, res)
            sewfile = get_data_path('ALMA:CO:ew:strict', name, res)
            add_CO_stats_to_table(
                vtt, bm0file, sm0file, sewfile, res)

        # add statistics of CPROPS clouds in table
        print("  Calculating statistics of CPROPS clouds")
        for res in lin_res:
            print(f"    @ {res.to('pc').value:.0f}pc")
            cpropsfile = get_data_path('ALMA:CPROPS', name, res)
            add_cprops_stats_to_table(
                vtt, cpropsfile, res)

        # mask rows where low resolution 'I_CO21' is NaN
        # This has to be the last step!!
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
