<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## DL_POLY / TrajParser

**Summary:** 5 mapped, 1 unmapped quantities (83.33% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `frame` | Mapped | `runschema.system.System` |
| `frame.info` | Unmapped | — |
| `frame.lattice_vectors` | Mapped | `runschema.system.System.atoms.lattice_vectors` |
| `frame.atoms` | Mapped | `runschema.system.System.atoms` |
| `frame.atoms.label` | Mapped | `runschema.system.System.atoms.labels` |
| `frame.atoms.array` | Mapped | `runschema.system.System.atoms.positions`<br>`runschema.system.System.atoms.velocities`<br>`runschema.calculation.Calculation.forces.total.value` |

## DL_POLY / FieldParser

**Summary:** 14 mapped, 4 unmapped quantities (77.78% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `units` | Unmapped | — |
| `neutral_groups` | Unmapped | — |
| `molecule_types` | Unmapped | — |
| `molecule` | Mapped | `runschema.method.Method.molecule_parameters` |
| `molecule.label_nummols` | Mapped | `runschema.method.MoleculeParameters.label` |
| `molecule.atoms` | Mapped | `runschema.method.AtomParameters.label`<br>`runschema.method.AtomParameters.mass`<br>`runschema.method.AtomParameters.charge` |
| `molecule.shell` | Unmapped | — |
| `molecule.bonds` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `molecule.angles` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `molecule.constraints` | Mapped | `runschema.system.System.constraint.kind`<br>`runschema.system.System.constraint.atom_indices`<br>`runschema.system.System.constraint.parameters` |
| `molecule.dihedrals` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `molecule.inversions` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `molecule.rigid` | Mapped | `runschema.system.System.constraint.kind`<br>`runschema.system.System.constraint.atom_indices` |
| `molecule.teth` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `vdw` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `tbp` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `fbp` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `metal` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |

## DL_POLY / MainfileParser

**Summary:** 7 mapped, 3 unmapped quantities (70.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `program_version_date` | Mapped | `runschema.run.Program.version` |
| `program_name` | Mapped | `runschema.run.Program.name` |
| `control_parameters` | Mapped | `simulationworkflowschema.molecular_dynamics.MolecularDynamics.method.thermodynamic_ensemble`<br>`simulationworkflowschema.molecular_dynamics.MolecularDynamics.method.integration_timestep` |
| `control_parameters.parameter` | Mapped | `simulationworkflowschema.molecular_dynamics.MolecularDynamics.method.thermodynamic_ensemble`<br>`simulationworkflowschema.molecular_dynamics.MolecularDynamics.method.integration_timestep` |
| `system_specification` | Unmapped | — |
| `system_specification.energy_unit` | Unmapped | — |
| `properties` | Mapped | `runschema.calculation.Calculation` |
| `properties.names` | Mapped | `runschema.calculation.Calculation.energy.total.value`<br>`runschema.calculation.Calculation.temperature`<br>`runschema.calculation.Calculation.pressure` |
| `properties.instantaneous` | Mapped | `runschema.calculation.Calculation.energy.total.value`<br>`runschema.calculation.Calculation.temperature`<br>`runschema.calculation.Calculation.pressure`<br>`runschema.calculation.Calculation.step`<br>`runschema.calculation.Calculation.time` |
| `properties.average` | Unmapped | — |
