# MegaTable

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6584841.svg)](https://doi.org/10.5281/zenodo.6584841)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

This repository contains the source code for generating rich multiwavelength data tables (a.k.a. the "mega-tables") for [the PHANGS team](https://sites.google.com/view/phangs/home). The current state of this repository matches the **version 4.2** internal release of the PHANGS mega-table products. The latest published version of the PHANGS mega-table products is **version 4.0** (in the [PHANGS CADC archive](https://www.canfar.net/storage/vault/list/phangs/RELEASES/Sun_etal_2022)).

The structure and content of these mega-tables are described in the following papers:

+ [Sun, Leroy, Rosolowsky, et al. (2022), *"Molecular Cloud Populations in the Context of Their Host Galaxy Environments: A Multiwavelength Perspective"*](https://ui.adsabs.harvard.edu/abs/2022AJ....164...43S)
+ [Sun, Leroy, Ostriker, et al. (2023), *"Star Formation Laws and Efficiencies across 80 Nearby Galaxies"*](https://ui.adsabs.harvard.edu/abs/2023ApJ...945L..19S)

Below is a figure from [Sun et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164...43S) showing *part of* the data aggregation workflow:

![Figure 1 in Sun et al. (2022)](https://content.cld.iop.org/journals/1538-3881/164/2/43/revision1/ajac74bdf1_lr.jpg "Figure 1 in Sun et al. (2022)")

## Code description

**[Important note] If you want to use the PHANGS mega-table products but do *not* plan to ingest more data into them or make your own mega-tables, you likely do not need the code in this repository. The PHANGS mega-table products (available in the CADC archive, see link above) can be read in and analyzed with `astropy.table.Table`.**

This repository offers a python module named `mega_table`, which provides the core infrastructure for mega-table creation, manipulation, and input/output. Most of the tools in this module are offered through three python classes:
+ `mega_table.table.RadialMegaTable`: Assemble measurements in radial bins with a given width
+ `mega_table.table.TessellMegaTable`: Assemble measurements according to a given tessellation pattern
+ `mega_table.table.ApertureMegaTable`: Assemble measurements in arbitrarily placed, fixed size apertures

In addition, the `pipelines` directory includes the actual production code for the PHANGS mega-table products. We publish all code here for reproducibility and hope that this serves as a template for creating mega-tables possibly for different targets with different sets of data. Key files in this directory include:
+ `config_data_path.json`: This file specifies the path to the underlying datasets on disk
+ `config_tables.json`: This file provides input parameters for table creation (e.g., table naming convention, bin width, FoV extent)
+ `format_*.csv`: These files controls the column-by-column content of the output tables (e.g., column names, physical units, descriptions)
+ `make_*.py`: These are the python scripts that create mega-tables from scratch and ingest multiwavelength measurements into them

## Dependencies

The core `mega_table` module depends on the following packages:
+ [`numpy`](https://numpy.org/)
+ [`scipy`](https://scipy.org/)
+ [`astropy`](https://www.astropy.org/)

If you would like to modify and run the scripts in the `pipelines` directory (e.g., to make your own mega-tables for some other galaxies), then you may also need the following packages depending on which part of the script is relevant:
+ [`reproject`](https://reproject.readthedocs.io/en/stable/index.html)
+ [`CO_conversion_factor`](https://github.com/astrojysun/COConversionFactor)

## Contact

If you need help using the code in this repository for science applications, please reach out to [Jiayi Sun](https://github.com/astrojysun). For bug reports and improvement suggestions, please open an Issue on Github.
