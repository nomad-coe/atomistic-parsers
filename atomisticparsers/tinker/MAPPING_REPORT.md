<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## Tinker / KeyParser

**Summary:** 0 mapped, 1 unmapped quantities (0.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `key_val` | Unmapped | — |

## Tinker / RunParser

**Summary:** 1 mapped, 2 unmapped quantities (33.33% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `molecular_dynamics` | Mapped | `simulationworkflowschema.MolecularDynamics.method.integration_timestep`<br>`simulationworkflowschema.MolecularDynamics.method.thermodynamic_ensemble` |
| `geometry_optimization` | Unmapped | — |
| `single_point` | Unmapped | — |

## Tinker / OutParser

**Summary:** 9 mapped, 22 unmapped quantities (29.03% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `iteration` | Unmapped | — |
| `iteration.step` | Mapped | `runschema.calculation.Calculation.energy.total.value` |
| `program_version` | Mapped | `runschema.run.Program.version` |
| `vibrate` | Unmapped | — |
| `vibrate.eigenvalues` | Unmapped | — |
| `vibrate.frequencies` | Mapped | `runschema.calculation.VibrationalFrequencies.value` |
| `minimize` | Unmapped | — |
| `minimize.method` | Mapped | `simulationworkflowschema.GeometryOptimizationMethod.method` |
| `minimize.x_tiner_final_function_value` | Unmapped | — |
| `minimize.x_tinker_final_rms_gradient` | Unmapped | — |
| `minimize.x_tinker_final_gradient_norm` | Unmapped | — |
| `dynamic` | Unmapped | — |
| `dynamic.instantaneous_values` | Unmapped | — |
| `dynamic.instantaneous_values.step` | Mapped | `runschema.calculation.Calculation.step` |
| `dynamic.instantaneous_values.time` | Unmapped | — |
| `dynamic.instantaneous_values.potential` | Mapped | `runschema.calculation.Calculation.energy.total.value`<br>`runschema.calculation.Calculation.energy.potential.value` |
| `dynamic.instantaneous_values.kinetic` | Mapped | `runschema.calculation.Calculation.energy.total.value`<br>`runschema.calculation.Calculation.energy.kinetic.value` |
| `dynamic.instantaneous_values.lattice_lengths` | Unmapped | — |
| `dynamic.instantaneous_values.lattice_angles` | Unmapped | — |
| `dynamic.instantaneous_values.frame` | Unmapped | — |
| `dynamic.instantaneous_values.coordinate_file` | Unmapped | — |
| `dynamic.average_values` | Unmapped | — |
| `dynamic.average_values.step` | Unmapped | — |
| `dynamic.average_values.time` | Unmapped | — |
| `dynamic.average_values.energy_total` | Unmapped | — |
| `dynamic.average_values.potential` | Unmapped | — |
| `dynamic.average_values.kinetic` | Unmapped | — |
| `dynamic.average_values.temperature` | Mapped | `runschema.calculation.Calculation.temperature` |
| `dynamic.average_values.pressure` | Mapped | `runschema.calculation.Calculation.pressure` |
| `dynamic.average_values.density` | Unmapped | — |
| `run` | Unmapped | — |
