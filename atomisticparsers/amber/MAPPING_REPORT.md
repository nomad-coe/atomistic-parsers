<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## AMBER

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

It reads data through `BasicParser` from `simulationparsers.utils`, configured with regex patterns for the text log (program version, total energy, atom positions/numbers) and auxiliary `.inpcrd`/`.prmtop` files, rather than declaring any `Quantity` file-parser class.
