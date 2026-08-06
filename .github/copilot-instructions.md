# Copilot Instructions for MegaTable

## Commands

- This repository defines project metadata in the `pyproject.toml` file.
- The committed runnable entry points are the pipeline scripts under `pipelines/`:
  - `python pipelines/make_base.py`
  - `python pipelines/make_phangsalma.py`
  - `python pipelines/make_phangsmuse.py`
  - `python pipelines/make_gauss.py`
  - `python pipelines/make_combined.py`
  - `python pipelines/gen_column_descr.py`
- Do not invent package, lint, or test commands that are not present in the repo. If you add tests or tooling later, document the exact invocation here instead of assuming a standard Python layout.

## High-level architecture

- `mega_table/` is the reusable library layer. It provides the table abstractions, WCS/image reduction helpers, and region/tessellation geometry used by all production scripts.
- `mega_table/core.py` defines the core hierarchy:
  - `BaseTable` wraps `astropy.table.QTable`.
  - `StatsTable` adds image and catalog aggregation helpers such as `calc_image_stats` and `calc_catalog_stats`.
  - `GeneralRegionTable` handles arbitrary region membership.
  - `VoronoiTessTable` handles seed-based tessellations and resampling.
- `mega_table/table.py` builds the public table types on top of those primitives:
  - `RadialMegaTable` for annuli
  - `TessellMegaTable` for hexagonal/square tessellations
  - `ApertureMegaTable` for fixed apertures
  - `StripeMegaTable` for minor-axis stripes
- `mega_table/utils.py` is the shared utility layer for WCS/FITS reduction, unit comparison, NaN-aware statistics, beam/pixel calculations, and galaxy deprojection. Reuse these helpers before adding new FITS/WCS logic elsewhere.
- `pipelines/` is PHANGS-specific production code built on the reusable classes:
  - `make_base.py` constructs the base radial and tessellated tables, samples raw maps, computes derived quantities, formats columns, and stamps metadata.
  - `make_phangsalma.py` adds ALMA/GMC measurements and derived molecular-gas quantities.
  - `make_phangsmuse.py` adds MUSE DAP and nebulae statistics.
  - `make_gauss.py` builds Gaussian-kernel sampled products on tessellated tables.
  - `make_combined.py` joins the per-product outputs into combined tables and merged all-galaxy products.
  - `gen_column_descr.py` generates `.rst` column-description tables from produced outputs.
- The pipelines are config-driven:
  - `config_tables.json` controls naming, tile shape/size, annulus width, FoV radius, version, and notes.
  - `config_data_path.json` contains the external data path templates used by the scripts.
  - `format_*.csv` files in `pipelines/` define final column order, units, display formats, and descriptions.

## Key conventions

- Keep the boundary between reusable infrastructure and PHANGS-specific science logic:
  - general table/image/WCS helpers belong in `mega_table/`
  - survey- or product-specific ingestion logic belongs in `pipelines/`
- Pipeline builders follow a consistent lifecycle: initialize geometry, add raw measurements, compute higher-level derived quantities, then call `format(...)` with column/unit/description metadata, and finally stamp table metadata such as `GALAXY`, `VERSION`, and `TBLNOTE`. Preserve that ordering when extending the products.
- Table reconstruction depends on metadata. The `read()` classmethods in `mega_table/table.py` expect keys such as `TBLTYPE`, geometry metadata, and stored coordinates/radii to still exist.
- Prefer the existing aggregation helpers over ad hoc loops:
  - `reduce_image_input()` for FITS/HDU/ndarray normalization
  - `calc_image_stats()` and `calc_catalog_stats()` for per-region summaries
  - `resample_image()` for sampling at seed/aperture centers
  - `deproject()` for galaxy-plane coordinates
- Preserve Astropy-style unit handling. Columns are usually `Quantity` columns in `QTable`, and the code frequently converts to explicit output units taken from `format_*.csv` or FITS `BUNIT`.
- Follow Astropy-style public docstrings when adding public classes/functions/methods. This codebase already uses the NumPy/Astropy sectioned docstring style heavily in `mega_table/core.py`, `mega_table/table.py`, and `mega_table/utils.py`.
- Keep optional dependencies local to the functionality that needs them, matching current patterns such as local imports of `reproject`, `CO_conversion_factor`, and `aplpy`. Do not make the core package import heavier without a strong reason.
- Reuse `mega_table/utils.py` for shared generic helpers instead of duplicating scientific utility code in multiple pipeline scripts.
- Use explicit warnings/exceptions instead of vague failures. The library code already prefers `warnings.warn(...)` and specific exceptions such as `ValueError`; preserve that style.
- The package export surface is intentionally small: `mega_table/__init__.py` re-exports the public table classes and does not carry implementation logic.
- The pipeline files use mixin-style multiple inheritance for product-specific behavior (for example, PHANGS-specific methods mixed into radial/tessellated table classes). If you extend this pattern, keep the bases orthogonal and avoid burying shared logic in geometry subclasses.
- Many pipeline scripts hard-code `config_dir` and `work_dir` to PHANGS filesystem locations under `/data/...`. Treat those as environment assumptions for the production scripts, not portable defaults for general users of the library.
