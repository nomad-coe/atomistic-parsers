<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## ASE

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

The parser reads structures and trajectories through ASE's `ase.io.trajectory.Trajectory` reader (via the shared `ASETrajParser` base) and populates the archive programmatically rather than declaring `Quantity(...)`-based file-parser classes.
