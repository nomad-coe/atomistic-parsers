<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## H5MD

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

The `HDF5Parser` subclass of `FileParser` reads the mainfile as an HDF5 file via `h5py` and extracts values by runtime path lookup (`get_value`/`get_attribute`), so it declares no static `Quantity(...)` definitions to report.
