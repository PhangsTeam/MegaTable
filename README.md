# MegaTable

[![Docs](https://img.shields.io/badge/docs-MkDocs-blue)](docs/index.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6584841.svg)](https://doi.org/10.5281/zenodo.6584841)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

``mega_table`` is a Python package for building and manipulating multiwavelength data tables (a.k.a. "mega-tables") for galaxy observations. The package provides table abstractions for radial bins, tessellations, apertures, and general region-based statistics.

This package has been used for generating mega-tables for [the PHANGS team](https://sites.google.com/view/phangs/home). The structure and content of these mega-tables are described in the papers by [Sun et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164...43S) and [Sun et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJ...945L..19S).

The current state of this repository matches the **version 4.3** internal release of the PHANGS mega-table products. The latest published version of the PHANGS mega-table products is **version 4.0** (in the [PHANGS CADC archive](https://www.canfar.net/storage/vault/list/phangs/RELEASES/Sun_etal_2022)).

## Installation

Install from a local clone:

```bash
pip install .
```

Install from GitHub:

```bash
pip install "git+https://github.com/astrojysun/MegaTable.git"
```

## Documentation

Read the documentation [here](https://astrojysun.github.io/MegaTable/).

## Contact

If you need help using the code in this repository for science applications, please reach out to [Jiayi Sun](https://github.com/astrojysun). For bug reports and improvement suggestions, please open an Issue on Github.
