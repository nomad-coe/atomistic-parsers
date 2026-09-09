<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## GROMACS / md.log

**Summary:** 6 mapped, 8 unmapped quantities (42.86% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `energies` | Unmapped | — |
| `step_info` | Unmapped | — |
| `time_start` | Mapped | `runschema.run.TimeRun.date_start` |
| `host_info` | Unmapped | — |
| `module_version` | Unmapped | — |
| `execution_path` | Unmapped | — |
| `working_path` | Unmapped | — |
| `header` | Mapped | `runschema.run.Program.version` |
| `header` | Mapped | `runschema.run.Program.version` |
| `input_parameters` | Mapped | `runschema.method.Method.force_field`<br>`simulationworkflowschema.MolecularDynamics.method`<br>`simulationworkflowschema.GeometryOptimization.method` |
| `maximum_force` | Mapped | `simulationworkflowschema.GeometryOptimizationResults.final_force_maximum` |
| `step` | Unmapped | — |
| `averages` | Unmapped | — |
| `time_end` | Mapped | `runschema.run.TimeRun.date_end` |

## GROMACS / mdout.mdp

**Summary:** 1 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `input_parameters` | Mapped | `runschema.method.Method.force_field`<br>`simulationworkflowschema.MolecularDynamics.method`<br>`simulationworkflowschema.GeometryOptimization.method` |

## GROMACS / dhdl.xvg

**Summary:** 0 mapped, 4 unmapped quantities (0.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `title` | Unmapped | — |
| `xaxis` | Unmapped | — |
| `yaxis` | Unmapped | — |
| `column_headers` | Unmapped | — |

## GROMACS / ener.edr

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.
