"""Tests for mega_table public table classes.

Inherited behavior from core (BaseTable.format, find_coords_in_regions,
calc_catalog_stats, calc_image_stats) is exercised entirely through the
user-facing classes rather than their internal base classes.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from mega_table import ApertureMegaTable, RadialMegaTable, TessellMegaTable
from conftest import make_simple_header


# ======================================================================
# RadialMegaTable
# ======================================================================


class TestRadialMegaTable:

    def test_default_nring(self) -> None:
        mt = RadialMegaTable(180.0, 0.0, 5.0)
        assert len(mt) == 20

    def test_explicit_max_radius(self) -> None:
        # nring = ceil(25 / 5) = 5
        mt = RadialMegaTable(180.0, 0.0, 5.0, rgal_max_arcsec=25.0)
        assert len(mt) == 5

    def test_ring_boundary_columns(self) -> None:
        mt = RadialMegaTable(180.0, 0.0, 5.0, rgal_max_arcsec=15.0)
        # nring = ceil(15/5) = 3
        assert len(mt) == 3
        assert mt['r_gal_angl_min'][0].to('arcsec').value == pytest.approx(0.0)
        assert mt['r_gal_angl_max'][0].to('arcsec').value == pytest.approx(5.0)
        assert mt['r_gal_angl_min'][2].to('arcsec').value == pytest.approx(10.0)
        assert mt['r_gal_angl_max'][2].to('arcsec').value == pytest.approx(15.0)

    def test_metadata_keys(self) -> None:
        mt = RadialMegaTable(
            180.0, 5.0, 10.0,
            gal_incl_deg=30.0, gal_posang_deg=45.0)
        assert mt.meta['TBLTYPE'] == 'RadialMegaTable'
        assert mt.meta['RA_DEG'] == pytest.approx(180.0)
        assert mt.meta['DEC_DEG'] == pytest.approx(5.0)
        assert mt.meta['INCL_DEG'] == pytest.approx(30.0)
        assert mt.meta['PA_DEG'] == pytest.approx(45.0)
        assert mt.meta['RBIN_DEG'] == pytest.approx(10.0 / 3600)

    def test_format_converts_units(self) -> None:
        mt = RadialMegaTable(180.0, 0.0, 5.0, rgal_max_arcsec=10.0)
        mt.format(
            colnames=['r_gal_angl_min', 'r_gal_angl_max'],
            units=['arcmin', 'arcmin'])
        assert mt['r_gal_angl_min'].unit == u.arcmin
        assert mt['r_gal_angl_max'].unit == u.arcmin

    def test_format_ignore_missing_fills_nan(self) -> None:
        mt = RadialMegaTable(180.0, 0.0, 5.0, rgal_max_arcsec=10.0)
        mt.format(
            colnames=['r_gal_angl_min', 'nonexistent_col'],
            ignore_missing=True)
        assert 'nonexistent_col' in mt.colnames
        assert np.all(np.isnan(mt['nonexistent_col']))

    def test_format_sets_description_and_format_spec(self) -> None:
        mt = RadialMegaTable(180.0, 0.0, 5.0, rgal_max_arcsec=10.0)
        mt.format(
            colnames=['r_gal_angl_min'],
            formats=['.3f'],
            descriptions=['inner ring radius'])
        assert mt['r_gal_angl_min'].info.format == '.3f'
        assert mt['r_gal_angl_min'].info.description == 'inner ring radius'

    def test_calc_image_stats_uniform(
            self, simple_header: fits.Header) -> None:
        data = np.full((20, 20), 3.0)
        mt = RadialMegaTable(180.0, 0.0, 5.0, rgal_max_arcsec=30.0)
        mt.calc_image_stats(
            data, header=simple_header,
            stat_func=np.nanmean, colname='T_mb', unit='K')
        values = mt['T_mb'].value
        non_nan = values[~np.isnan(values)]
        assert len(non_nan) > 0
        np.testing.assert_allclose(non_nan, 3.0)

    def test_calc_image_stats_missing_file_suppress(self) -> None:
        mt = RadialMegaTable(180.0, 0.0, 5.0, rgal_max_arcsec=15.0)
        mt.calc_image_stats(
            '/nonexistent/map.fits',
            stat_func=np.nanmean, colname='T_mb', unit='K',
            suppress_error=True)
        assert np.all(np.isnan(mt['T_mb'].value))
        assert mt['T_mb'].unit == u.K

    def test_write_read_roundtrip(self, tmp_path) -> None:
        mt = RadialMegaTable(180.0, 0.0, 5.0, rgal_max_arcsec=20.0)
        path = str(tmp_path / 'radial.ecsv')
        mt.write(path, overwrite=True)

        mt2 = RadialMegaTable.read(path)
        assert len(mt2) == len(mt)
        np.testing.assert_allclose(
            mt2['r_gal_angl_min'].to('arcsec').value,
            mt['r_gal_angl_min'].to('arcsec').value)
        np.testing.assert_allclose(
            mt2['r_gal_angl_max'].to('arcsec').value,
            mt['r_gal_angl_max'].to('arcsec').value)
        assert mt2.meta['TBLTYPE'] == 'RadialMegaTable'

    def test_read_wrong_type_raises(self, tmp_path) -> None:
        # Writing a TessellMegaTable and trying to read it as
        # RadialMegaTable should raise ValueError.
        tmt = TessellMegaTable(180.0, 0.0, 60.0, 15.0)
        path = str(tmp_path / 'tessell.ecsv')
        tmt.write(path, overwrite=True)
        with pytest.raises(ValueError, match="RadialMegaTable"):
            RadialMegaTable.read(path)


# ======================================================================
# ApertureMegaTable
# ======================================================================


class TestApertureMegaTable:

    def test_init_and_length(self) -> None:
        ra = np.array([180.0, 181.0])
        dec = np.array([0.0, 0.0])
        mt = ApertureMegaTable(ra, dec, aperture_size_arcsec=10.0)
        assert len(mt) == 2
        assert 'RA' in mt.colnames
        assert 'DEC' in mt.colnames
        assert mt.meta['APER_DEG'] == pytest.approx(10.0 / 3600)

    def test_custom_aperture_names(self) -> None:
        mt = ApertureMegaTable(
            np.array([180.0]), np.array([0.0]),
            aperture_size_arcsec=10.0,
            aperture_names=['NGC0000'])
        assert mt['REGION'][0] == 'NGC0000'

    def test_default_aperture_names(self) -> None:
        mt = ApertureMegaTable(
            np.array([180.0, 181.0]), np.array([0.0, 0.0]),
            aperture_size_arcsec=10.0)
        assert mt['REGION'][0] == 'Aper#1'
        assert mt['REGION'][1] == 'Aper#2'

    def test_find_coords_in_regions_membership(self) -> None:
        # Two apertures 1 degree apart; flag array should be (n_coords, 2).
        # Aperture radius is 100 arcsec ≈ 0.028 deg,
        # so neither aperture contains a point at the other's center.
        mt = ApertureMegaTable(
            np.array([180.0, 181.0]), np.array([0.0, 0.0]),
            aperture_size_arcsec=100.0)
        ra = np.array([180.0, 181.0, 185.0])
        dec = np.array([0.0, 0.0, 0.0])
        flags = mt.find_coords_in_regions(ra, dec)
        assert flags.shape == (3, 2)
        assert flags[0, 0]          # (180, 0) is inside aperture 0
        assert not flags[0, 1]      # (180, 0) is outside aperture 1
        assert flags[1, 1]          # (181, 0) is inside aperture 1
        assert not flags[1, 0]      # (181, 0) is outside aperture 0
        assert not flags[2, 0]      # (185, 0) is in neither aperture
        assert not flags[2, 1]

    def test_calc_catalog_stats_basic(self) -> None:
        # Two apertures; catalog has two objects in aperture 0 and one
        # object in aperture 1.
        mt = ApertureMegaTable(
            np.array([180.0, 181.0]), np.array([0.0, 0.0]),
            aperture_size_arcsec=100.0)
        entries = np.array([10.0, 20.0, 30.0])
        ra_cat = np.array([180.0, 180.0, 181.0])
        dec_cat = np.array([0.0, 0.0, 0.0])
        mt.calc_catalog_stats(
            entries, ra_cat, dec_cat,
            stat_func=np.mean, colname='val', unit='pc')
        assert mt['val'][0].value == pytest.approx(15.0)  # mean of [10, 20]
        assert mt['val'][1].value == pytest.approx(30.0)  # mean of [30]
        assert mt['val'].unit == u.pc

    def test_calc_catalog_stats_empty_region_is_nan(self) -> None:
        mt = ApertureMegaTable(
            np.array([180.0]), np.array([0.0]),
            aperture_size_arcsec=100.0)
        # Catalog objects are far outside the aperture.
        mt.calc_catalog_stats(
            np.array([1.0, 2.0]),
            np.array([185.0, 185.0]),
            np.array([0.0, 0.0]),
            stat_func=np.mean, colname='val', unit='')
        assert np.isnan(mt['val'][0])

    def test_calc_catalog_stats_invalid_stat_func_raises(self) -> None:
        mt = ApertureMegaTable(
            np.array([180.0]), np.array([0.0]),
            aperture_size_arcsec=100.0)
        with pytest.raises(ValueError, match="stat_func"):
            mt.calc_catalog_stats(
                np.array([1.0]), np.array([180.0]), np.array([0.0]),
                stat_func=None, colname='val')

    def test_calc_image_stats_uniform(
            self, simple_header: fits.Header) -> None:
        data = np.full((20, 20), 7.0)
        mt = ApertureMegaTable(
            np.array([180.0]), np.array([0.0]),
            aperture_size_arcsec=36.0)  # ~10 pixels radius
        mt.calc_image_stats(
            data, header=simple_header,
            stat_func=np.nanmean, colname='flux', unit='K')
        assert mt['flux'][0].value == pytest.approx(7.0)
        assert mt['flux'].unit == u.K

    def test_calc_image_stats_missing_file_suppress(self) -> None:
        mt = ApertureMegaTable(
            np.array([180.0]), np.array([0.0]),
            aperture_size_arcsec=10.0)
        mt.calc_image_stats(
            '/nonexistent/map.fits',
            stat_func=np.nanmean, colname='flux', unit='K',
            suppress_error=True)
        assert np.isnan(mt['flux'][0])
        assert mt['flux'].unit == u.K

    def test_resample_image_constant(
            self, simple_header: fits.Header) -> None:
        # All pixels are 5.0, so no matter which pixel is the nearest,
        # the resampled value must equal 5.0.
        data = np.full((20, 20), 5.0)
        mt = ApertureMegaTable(
            np.array([180.0]), np.array([0.0]),
            aperture_size_arcsec=10.0)
        mt.resample_image(
            data, header=simple_header, colname='flux', unit='K')
        assert mt['flux'][0].value == pytest.approx(5.0)
        assert mt['flux'].unit == u.K

    def test_resample_image_missing_file_suppress(self) -> None:
        mt = ApertureMegaTable(
            np.array([180.0]), np.array([0.0]),
            aperture_size_arcsec=10.0)
        mt.resample_image(
            '/nonexistent/map.fits', colname='flux', unit='K',
            suppress_error=True)
        assert np.isnan(mt['flux'][0])

    def test_write_read_roundtrip(self, tmp_path) -> None:
        ra = np.array([180.0, 181.0])
        dec = np.array([0.0, 1.0])
        mt = ApertureMegaTable(
            ra, dec, aperture_size_arcsec=15.0,
            aperture_names=['A', 'B'])
        path = str(tmp_path / 'aperture.ecsv')
        mt.write(path, overwrite=True)

        mt2 = ApertureMegaTable.read(path)
        assert len(mt2) == len(mt)
        np.testing.assert_allclose(
            mt2['RA'].to('deg').value,
            mt['RA'].to('deg').value)
        assert mt2.meta['TBLTYPE'] == 'ApertureMegaTable'


# ======================================================================
# TessellMegaTable
# ======================================================================


class TestTessellMegaTable:

    def test_hexagon_tiles(self) -> None:
        mt = TessellMegaTable(180.0, 0.0, 60.0, 15.0, tile_shape='hexagon')
        assert len(mt) > 0
        assert 'RA' in mt.colnames
        assert 'DEC' in mt.colnames

    def test_square_tiles(self) -> None:
        mt = TessellMegaTable(180.0, 0.0, 60.0, 15.0, tile_shape='square')
        assert len(mt) > 0

    def test_invalid_tile_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tile shape"):
            TessellMegaTable(180.0, 0.0, 60.0, 15.0, tile_shape='triangle')

    def test_metadata_keys(self) -> None:
        mt = TessellMegaTable(180.0, 0.0, 60.0, 15.0, tile_shape='hexagon')
        assert mt.meta['TBLTYPE'] == 'TessellMegaTable'
        assert mt.meta['RA_DEG'] == pytest.approx(180.0)
        assert mt.meta['DEC_DEG'] == pytest.approx(0.0)
        assert mt.meta['FOV_DEG'] == pytest.approx(60.0 / 3600)
        assert mt.meta['TILE_DEG'] == pytest.approx(15.0 / 3600)
        assert mt.meta['TILE_DEF'] == 'hexagon'

    def test_tiles_within_fov(self) -> None:
        fov_arcsec = 60.0
        tile_arcsec = 15.0
        mt = TessellMegaTable(180.0, 0.0, fov_arcsec, tile_arcsec)
        # All tile centres should lie within fov + one tile-spacing from centre.
        cos_dec = np.cos(np.deg2rad(0.0))
        sep = np.sqrt(
            ((mt['RA'].value - 180.0) * cos_dec) ** 2 +
            (mt['DEC'].value - 0.0) ** 2)
        threshold = (fov_arcsec + tile_arcsec / np.sqrt(3)) / 3600
        assert np.all(sep <= threshold + 1e-9)

    def test_calc_image_stats_uniform(
            self, simple_header: fits.Header) -> None:
        # VoronoiTessTable.find_coords_in_regions returns an index array
        # (not boolean), so this exercises a different code path from the
        # GeneralRegionTable-based classes.
        data = np.full((20, 20), 4.0)
        mt = TessellMegaTable(180.0, 0.0, 60.0, 15.0)
        mt.calc_image_stats(
            data, header=simple_header,
            stat_func=np.nanmean, colname='flux', unit='K')
        values = mt['flux'].value
        non_nan = values[~np.isnan(values)]
        assert len(non_nan) > 0
        np.testing.assert_allclose(non_nan, 4.0)

    def test_resample_image_constant(
            self, simple_header: fits.Header) -> None:
        data = np.full((20, 20), 9.0)
        mt = TessellMegaTable(180.0, 0.0, 60.0, 15.0)
        mt.resample_image(
            data, header=simple_header, colname='flux', unit='K')
        np.testing.assert_allclose(mt['flux'].value, 9.0)
        assert mt['flux'].unit == u.K

    def test_write_read_roundtrip(self, tmp_path) -> None:
        mt = TessellMegaTable(180.0, 0.0, 60.0, 15.0, tile_shape='square')
        path = str(tmp_path / 'tessell.ecsv')
        mt.write(path, overwrite=True)

        mt2 = TessellMegaTable.read(path)
        assert len(mt2) == len(mt)
        np.testing.assert_allclose(
            mt2['RA'].to('deg').value,
            mt['RA'].to('deg').value)
        assert mt2.meta['TBLTYPE'] == 'TessellMegaTable'
