# NYC Collision Heatmaps and Inferred Camera Locations, 2021–2025

Yihan's executed notebook deliverable for the descriptive spatial analysis in [issue #12](https://github.com/Group-7-Capstone/Group-7-Capstone/issues/12).

## Open the notebook

[NYC_Collision_Heatmaps_and_Inferred_Camera_Locations_2021_2025.ipynb](NYC_Collision_Heatmaps_and_Inferred_Camera_Locations_2021_2025.ipynb)

The notebook contains saved tables and 17 embedded, high-resolution figures. Download it and open it in Jupyter or VS Code if GitHub cannot preview the approximately 47 MiB file. Reading the saved results does not require downloading the source datasets or rerunning cells.

## Scope and results

- Study period: January 1, 2021 through December 31, 2025; all five NYC boroughs.
- Sources: NYC Open Data Crashes, Person, Vehicles, and annual Parking Violations Issued datasets; supplied LION street network.
- Six cumulative heatmaps and six annual comparison figures: collisions, involved people, involved vehicles, all tickets, speed-camera tickets, and red-light-camera tickets.
- Separate speed/red-light inferred-location maps, two collision overlays, and one mapping-coverage figure.
- Full selected-period extracts: 487,914 collisions; 1,697,877 involved-person records; 985,702 involved-vehicle records; 80,714,373 unique tickets.
- Mapped records: 449,308 collisions (92.1%); 1,549,051 person records (91.2%); 907,515 vehicle records (92.1%); 65,104,669 tickets (80.7%).
- Inferred enforcement locations: 2,593 speed and 229 red light, with camera type and recorded direction kept separate.

Counts are concentrations of recorded events, not exposure-adjusted rates or causal effects. Inferred locations are approximate enforcement intersections or addresses, not verified camera poles or physical device counts. First observed tickets are not installation dates. Direction text does not independently verify the enforced carriageway.

## Reproducibility and validation

This is an archived, executed deliverable from the separate local `camera-location-analysis` project. It is not yet integrated into this repository's `data/ → preprocessing/ → model/ → validation/` interfaces. The notebook imports that project's analysis modules and reads its frozen caches; those modules and caches must be available to rerun the analysis. This folder alone is not a standalone execution environment.

The original project retains the processing scripts, pinned Python environment, source receipts, complete data caches, camera CSV/GeoJSON, unresolved-location records, and standalone 300 dpi PNG exports. These data files and the local environment are not committed here, in accordance with the repository rules.

The original clean-kernel run completed 14 code cells with no error outputs, saved 17 figures, and passed 377 reconciliation checks. Ten geocoding regression tests passed. Codex visually desk-reviewed 100 stratified candidate locations against ticket text and LION geometry; this was not independent human or field verification. A Saint John street-name normalization-label issue and complex-junction limitations are documented in the notebook and local review register.

The uploaded notebook is byte-for-byte identical to the completed local notebook. SHA-256: `e52b3a794c47893ac190560b6e52a2117a132e7d16b19f961c2165a1daff011e`.

Repository test status at upload: the checkout has no `tests/` directory or test modules. `uv run pytest` was attempted but could not start because `uv` is unavailable on the upload host. Notebook JSON, sequential execution counts, all 17 saved image outputs, absence of error outputs, and the source/copy checksum were verified separately. No team pipeline code or dependency files were changed.
