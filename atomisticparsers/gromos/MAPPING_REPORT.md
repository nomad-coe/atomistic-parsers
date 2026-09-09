<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## GROMOS

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

The parser configures a shared `BasicParser` (`simulationparsers.utils.BasicParser`) with regex patterns for program version, positions, total energy, pressure, and timestep rather than declaring a `TextParser`/`XMLParser`/`FileParser` subclass with `Quantity(...)` definitions.
