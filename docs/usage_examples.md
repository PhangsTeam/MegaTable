# Usage examples

## Importing the package

```python
# for radial bin-based statistics
from mega_table import RadialMegaTable

# for square/hexagonal cell statistics
from mega_table import TessellMegaTable

# for fixed-size aperture statistics
from mega_table import ApertureMegaTable
```

## Reading and inspecting mega-tables

Read an existing radial mega-table:

```python
from mega_table import RadialMegaTable

t = RadialMegaTable.read('path/to/table.ecsv')
```

Inspect the generated table structure and metadata:

```python
print(t.info)
print(t.meta)
```

## Creating new mega-tables from scratch

Create a new mega-table for radial bin statistics:

```python
t_rad = RadialMegaTable(
    gal_ra_deg=120.0,  # galaxy center RA (in degrees)
    gal_dec_deg=-30.0,  # galaxy center Dec (in degrees)
    gal_incl_deg=60.0,  # galaxy inclination angle (in degrees)
    gal_posang_deg=45.0,  # galaxy position angle (in degrees)
    rgal_bin_arcsec=10.0,  # radial bin width (in arcsec)
    rgal_max_arcsec=1000.0,  # outermost bin radius (in arcsec)
)
```

Create a new mega-table for hexagonal cell statistics:

```python
t_hex = TessellMegaTable(
    center_ra_deg=120.0,  # galaxy center RA (in degrees)
    center_dec_deg=-30.0,  # galaxy center Dec (in degrees)
    fov_radius_arcsec=1000.0,  # FoV covered by all the cells (in arcsec)
    tile_size_arcsec=10.0,  # size of each cell (in arcsec)
    tile_shape='hexagon',  # shape of the cell (hexagon or square)
)
```
 
## Manipulating mega-tables

Given a FITS image, calculate a user-specified statistics (mean, median, etc) from the pixel values in each radial bin, and write the results to a new column:

```python
t.calc_image_stats(
    '/path/to/image.fits',  # path to the FITS image file
    stat_func=np.nanmean,  # stats to calculate in each radial bin
    colname='new_column_from_image',  # new column name to store the result
    unit='header',  # inherit the physical unit in the FITS header
)

# inspect results
print(t['new_column_from_image'])
```

Given a source catalog (with RA, Dec, and some properties of interest), calculate a user-specified statistics from the sources falling in each radial bin, and write the results to a new column:

```python
from astropy.table import Table

catalog = Table.read("/path/to/catalog.fits")

t.calc_catalog_stats(
    catalog['prop'],  # the property of interest
    catalog['RA'],  # RA coordinates
    catalog['Dec'],  # Dec coordinates
    stat_func=np.nanmean,  # stats to calculate in each radial bin
    colname='new_column_from_catalog',  # new column name to store the result
) 

# inspect results
print(t['new_column_from_catalog'])
```

## Writing mega-tables to disk

We recommend the astropy [ECSV](https://docs.astropy.org/en/stable/io/ascii/ecsv.html) format to ensure a full write/read round trip:

```python
t.write('path/to/new_table.ecsv')
```
