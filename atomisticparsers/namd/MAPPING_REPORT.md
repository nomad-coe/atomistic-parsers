<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## NAMD / ConfigParser

**Summary:** 0 mapped, 1 unmapped quantities (0.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `parameter` | Unmapped | — |

## NAMD / MainfileParser

**Summary:** 6 mapped, 8 unmapped quantities (42.86% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `version_arch` | Mapped | `runschema.run.program.version` |
| `config_file` | Unmapped | — |
| `simulation_parameters` | Mapped | `runschema.run.system.atoms.lattice_vectors` |
| `simulation_parameters.parameter` | Unmapped | — |
| `simulation_parameters.cell` | Mapped | `runschema.run.system.atoms.lattice_vectors` |
| `simulation_parameters.output_file` | Unmapped | — |
| `simulation_parameters.coordinate_file` | Unmapped | — |
| `simulation_parameters.structure_file` | Unmapped | — |
| `simulation_parameters.parameter_file` | Unmapped | — |
| `step` | Mapped | `runschema.run.calculation.energy.total.value`<br>`runschema.run.calculation.energy.electronic.value`<br>`runschema.run.calculation.energy.van_der_waals.value`<br>`runschema.run.calculation.energy.contributions.value`<br>`runschema.run.calculation.temperature`<br>`runschema.run.calculation.pressure` |
| `timing` | Mapped | `runschema.run.calculation.time_calculation`<br>`runschema.run.calculation.time_physical` |
| `total_time` | Mapped | `runschema.run.calculation.time_calculation`<br>`runschema.run.calculation.time_physical` |
| `property_names` | Unmapped | — |
| `coordinates_write_step` | Unmapped | — |
