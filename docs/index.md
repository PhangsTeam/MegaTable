# mega_table

## Overview

``mega_table`` is a Python package for building and manipulating multiwavelength data tables (a.k.a. "mega-tables") for galaxy observations. The package provides table abstractions for radial bins, tessellations, apertures, and general region-based statistics.

This package has been used for generating mega-tables for [the PHANGS team](https://sites.google.com/view/phangs/home). The structure and content of these mega-tables are described in the papers by [Sun et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164...43S) and [Sun et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJ...945L..19S). The published mega-tables products are available in a [CADC archive](https://www.canfar.net/storage/vault/list/phangs/RELEASES/Sun_etal_2022), whereas the table creation pipeline can be found on [GitHub](https://github.com/PhangsTeam/MegaTable/tree/master/pipelines).

!!! note "Important Note"
    **If you just want to use the PHANGS mega-table products but do not plan to manipulate them or to make new mega-tables from scratch, you likely do not need the `mega_table` package. The PHANGS mega-table products can be read in and analyzed with [`astropy.table`](https://docs.astropy.org/en/stable/table/index.html).**

Below is a figure from [Sun et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164...43S) showing part of the PHANGS data aggregation workflow:

![Figure 1 in Sun et al. (2022)](https://content.cld.iop.org/journals/1538-3881/164/2/43/revision1/ajac74bdf1_lr.jpg "Figure 1 in Sun et al. (2022)")


## Installation

Install from a local clone:

```bash
pip install .
```

Install from GitHub:

```bash
pip install "git+https://github.com/PhangsTeam/MegaTable.git"
```


## Dependencies

`mega_table` depends on the following packages:

+ [`numpy`](https://numpy.org/)
+ [`scipy`](https://scipy.org/)
+ [`astropy`](https://www.astropy.org/)

If you would like to modify and run [the PHANGS mega-table pipeline scripts](https://github.com/PhangsTeam/MegaTable/tree/master/pipelines) (e.g., to make your own mega-tables from scratch), then you may also need the following packages depending on which part of the script is relevant:

+ [`reproject`](https://reproject.readthedocs.io/en/stable/index.html)
+ [`CO_conversion_factor`](https://github.com/astrojysun/COConversionFactor)


## Navigation links

- [Usage examples](usage_examples.md)
- [API reference](api/index.md)
