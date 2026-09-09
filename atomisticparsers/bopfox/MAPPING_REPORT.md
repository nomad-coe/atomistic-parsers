<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## BOPfox / ModelsbxParser

**Summary:** 3 mapped, 2 unmapped quantities (60.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `model` | Mapped | `runschema.method.Method.tb.xtb.name`<br>`runschema.method.Method.force_field.model.name` |
| `model.name` | Mapped | `runschema.method.Method.tb.xtb.name`<br>`runschema.method.Method.force_field.model.name` |
| `model.parameters` | Unmapped | — |
| `model.atom` | Unmapped | — |
| `model.bond` | Mapped | `runschema.method.Interaction.name`<br>`runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.parameters`<br>`runschema.method.Interaction.atom_labels` |

## BOPfox / StrucbxParser

**Summary:** 4 mapped, 1 unmapped quantities (80.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `lattice_constant` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `lattice_vectors` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `coordinate_type` | Mapped | `runschema.system.Atoms.positions` |
| `label_position` | Mapped | `runschema.system.Atoms.labels`<br>`runschema.system.Atoms.positions` |
| `magnetisation` | Unmapped | — |

## BOPfox / XYZParser

**Summary:** 1 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `frame` | Mapped | `runschema.system.Atoms.labels`<br>`runschema.system.Atoms.positions`<br>`runschema.calculation.Energy.total.value`<br>`runschema.calculation.Energy.total.values_per_atom`<br>`runschema.calculation.Forces.total.value` |

## BOPfox / InfoxParser

**Summary:** 0 mapped, 1 unmapped quantities (0.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `parameter` | Unmapped | — |

## BOPfox / MainfileParser

**Summary:** 24 mapped, 7 unmapped quantities (77.42% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `energy` | Mapped | `runschema.calculation.Calculation.energy` |
| `energy.contribution` | Mapped | `runschema.calculation.Energy.total`<br>`runschema.calculation.Energy.electrostatic`<br>`runschema.calculation.Energy.nuclear_repulsion`<br>`runschema.calculation.Energy.contributions` |
| `energy.contribution.type` | Mapped | `runschema.calculation.Energy.total`<br>`runschema.calculation.Energy.electrostatic`<br>`runschema.calculation.Energy.nuclear_repulsion`<br>`runschema.calculation.EnergyEntry.kind` |
| `energy.contribution.atomic` | Mapped | `runschema.calculation.EnergyEntry.values_per_atom` |
| `energy.contribution.total` | Mapped | `runschema.calculation.EnergyEntry.value` |
| `forces` | Mapped | `runschema.calculation.Calculation.forces` |
| `forces.contribution` | Mapped | `runschema.calculation.Forces.total`<br>`runschema.calculation.Forces.contributions` |
| `forces.contribution.type` | Mapped | `runschema.calculation.Forces.total`<br>`runschema.calculation.ForcesEntry.kind` |
| `forces.contribution.atomic` | Mapped | `runschema.calculation.ForcesEntry.value` |
| `stress` | Mapped | `runschema.calculation.Calculation.stress` |
| `stress.total` | Mapped | `runschema.calculation.StressEntry.value` |
| `stress.contribution` | Mapped | `runschema.calculation.Stress.total`<br>`runschema.calculation.Stress.contributions` |
| `stress.contribution.type` | Mapped | `runschema.calculation.Stress.total`<br>`runschema.calculation.StressEntry.kind` |
| `stress.contribution.atomic` | Mapped | `runschema.calculation.StressEntry.values_per_atom` |
| `energy_fermi` | Mapped | `runschema.calculation.Energy.fermi` |
| `charges` | Mapped | `runschema.calculation.Calculation.charges` |
| `charges.n_electrons` | Mapped | `runschema.calculation.Charges.n_electrons` |
| `charges.charge` | Mapped | `runschema.calculation.Charges.value` |
| `magnetic_moments` | Mapped | `runschema.calculation.Charges.orbital_projected` |
| `magnetic_moments.mag_mom` | Mapped | `runschema.calculation.ChargesValue.atom_index`<br>`runschema.calculation.ChargesValue.orbital`<br>`runschema.calculation.ChargesValue.spin_z` |
| `onsite_levels` | Unmapped | — |
| `onsite_levels.energy` | Unmapped | — |
| `program_version` | Mapped | `runschema.run.Program.version` |
| `simulation` | Unmapped | — |
| `simulation.parameter` | Unmapped | — |
| `lattice_vectors` | Unmapped | — |
| `label_position` | Unmapped | — |
| `n_atoms` | Mapped | `runschema.calculation.EnergyEntry.value` |
| `relaxation` | Mapped | `runschema.calculation.Run.calculation` |
| `relaxation.cycle` | Mapped | `runschema.calculation.Run.calculation` |
| `md_column_names` | Unmapped | — |
