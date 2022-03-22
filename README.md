# MegaTable

This repository contains the source code for generating rich multiwavelength data tables (a.k.a. the "mega-tables") for [the PHANGS team](https://sites.google.com/view/phangs/home). The table construction and data aggregation schemes are described in the following papers:

+ Sun, Leroy, Rosolowsky, et al. (submitted), *"Molecular Cloud Populations in the Context of Their Host Galaxy Environments: A Multiwavelength Perspective"*
+ [Sun, Leroy, Ostriker, et al. (2020), *"Dynamical Equilibrium in the Molecular ISM in 28 Nearby Star-forming Galaxies"*](http://adsabs.harvard.edu/abs/2020ApJ...892..148S)

The current version of the repository corresponds to PHANGS mega-tables **version 3.0**, which is the first public release version.

## Code Architecture

This repository consists of a python module named `mega_table` and a suite of python scripts and configuration files in the `pipelines` subdirectory.

The `mega_table` module provides the core infrastructure for table creation, manipulation, and input/output. This module relies heavily on the [`astropy.table`](https://docs.astropy.org/en/stable/table/index.html) subpackage. Most of the tools in this module are offered through three python classes:
+ `mega_table.table.RadialMegaTable`: Assemble measurements in radial bins with a given width
+ `mega_table.table.TessellMegaTable`: Assemble measurements according to a given tessellation pattern
+ `mega_table.table.ApertureMegaTable`: Assemble measurements in arbitrarily placed, fixed size apertures

The `pipelines` subdirectory includes the actual production code for the PHANGS mega-tables:
+ `config_data_path.json`: This file specifies the path to the underlying datasets on disk
+ `config_tables.json`: This file provides input parameters for table creation (e.g., table naming convention, bin width, FoV extent)
+ `format_*.csv`: These files controls the column-by-column content of the output tables (e.g., column names, physical units, descriptions)
+ `make_*.py`: These are the python scripts that define and incorporate the multiwavelength measurements in the PHANGS mega-tables

## Contact

If you need help using the code in this repository for science applications, please reach out to [Jiayi Sun](https://github.com/astrojysun). For bug reports and improvement suggestions, please open an Issue on Github.
