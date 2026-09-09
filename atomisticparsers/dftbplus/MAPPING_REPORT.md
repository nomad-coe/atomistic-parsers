<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## DFTB+ / DetailedParser

**Summary:** 10 mapped, 11 unmapped quantities (47.62% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `coordinates` | Unmapped | — |
| `charges` | Unmapped | — |
| `eigenvalues` | Unmapped | — |
| `occupations` | Unmapped | — |
| `eigenvalues_occupations` | Mapped | `runschema.calculation.Calculation.eigenvalues.energies`<br>`runschema.calculation.Calculation.eigenvalues.occupations` |
| `eigenvalues_occupations.kpoint` | Mapped | `runschema.calculation.Calculation.eigenvalues.energies`<br>`runschema.calculation.Calculation.eigenvalues.occupations` |
| `fermi_level` | Mapped | `runschema.calculation.Calculation.energy.fermi` |
| `energy_x_dftbp_band` | Unmapped | — |
| `energy_x_dftbp_ts` | Unmapped | — |
| `energy_x_dftbp_band_free` | Unmapped | — |
| `energy_x_dftbp_band_t0` | Unmapped | — |
| `energy_sum_eigenvalues` | Mapped | `runschema.calculation.Calculation.energy.sum_eigenvalues.value` |
| `energy_x_dftbp_scc` | Unmapped | — |
| `energy_electronic` | Mapped | `runschema.calculation.Calculation.energy.electronic.value` |
| `energy_nuclear_repulsion` | Mapped | `runschema.calculation.Calculation.energy.nuclear_repulsion.value` |
| `energy_x_dftbp_dispersion` | Unmapped | — |
| `energy_total` | Mapped | `runschema.calculation.Calculation.energy.total.value` |
| `energy_x_dftbp_total_mermin` | Unmapped | — |
| `pressure` | Mapped | `runschema.calculation.Calculation.pressure` |
| `forces` | Mapped | `runschema.calculation.Calculation.forces.total.value` |
| `dipole` | Mapped | `runschema.calculation.Calculation.multipoles.dipole.total` |

## DFTB+ / OutParser

**Summary:** 5 mapped, 10 unmapped quantities (33.33% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `program_version` | Mapped | `runschema.run.Program.version` |
| `input_file` | Unmapped | — |
| `processed_input_file` | Unmapped | — |
| `parser_version` | Unmapped | — |
| `sk_files` | Unmapped | — |
| `input_parameters` | Unmapped | — |
| `input_parameters.key_val` | Unmapped | — |
| `input_parameters.kpoints_weights` | Unmapped | — |
| `step` | Unmapped | — |
| `step.scf` | Mapped | `runschema.calculation.Calculation.scf_iteration.energy.total.value`<br>`runschema.calculation.Calculation.scf_iteration.energy.change` |
| `step.energy_total` | Mapped | `runschema.calculation.Calculation.energy.total.value` |
| `step.energy_total_t0` | Mapped | `runschema.calculation.Calculation.energy.total_t0.value` |
| `step.energy_x_dftbp_total_mermin` | Unmapped | — |
| `step.pressure` | Mapped | `runschema.calculation.Calculation.pressure` |
| `step.maximum_force` | Unmapped | — |
