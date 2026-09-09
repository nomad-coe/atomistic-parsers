<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## libAtoms

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

The parser delegates to `simulationparsers.utils.BasicParser`, configuring it with regex string patterns (passed as keyword arguments) rather than declaring a `TextParser`/`XMLParser`/`FileParser` subclass with `Quantity(...)` objects.
