<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
## xTB / OutParser

**Summary:** 33 mapped, 47 unmapped quantities (41.25% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `setup` | Unmapped | — |
| `setup.parameter` | Unmapped | — |
| `summary` | Mapped | `runschema.calculation.Calculation.energy` |
| `summary.energy_total` | Mapped | `runschema.calculation.Energy.total.value` |
| `summary.x_xtb_gradient_norm` | Unmapped | — |
| `summary.x_xtb_hl_gap` | Unmapped | — |
| `summary.energy_x_xtb_scc` | Unmapped | — |
| `summary.energy_x_xtb_isotropic_es` | Unmapped | — |
| `summary.energy_x_xtb_anisotropic_es` | Unmapped | — |
| `summary.energy_x_xtb_anisotropic_xc` | Unmapped | — |
| `summary.energy_x_xtb_dispersion` | Unmapped | — |
| `summary.energy_electrostatic` | Mapped | `runschema.calculation.Energy.electrostatic.value` |
| `summary.energy_x_xtb_repulsion` | Unmapped | — |
| `summary.energy_x_xtb_halogen_bond_corr` | Unmapped | — |
| `summary.energy_x_xtb_add_restraining` | Unmapped | — |
| `summary.charge_total` | Unmapped | — |
| `eigenvalues` | Mapped | `runschema.calculation.BandEnergies.occupations`<br>`runschema.calculation.BandEnergies.energies`<br>`runschema.calculation.BandEnergies.kpoints` |
| `hl_gap` | Unmapped | — |
| `energy_fermi` | Unmapped | — |
| `dipole` | Mapped | `runschema.calculation.Multipoles.dipole` |
| `dipole.q` | Unmapped | — |
| `dipole.full` | Mapped | `runschema.calculation.MultipolesEntry.total` |
| `quadrupole` | Mapped | `runschema.calculation.Multipoles.quadrupole` |
| `quadrupole.q` | Unmapped | — |
| `quadrupole.full` | Mapped | `runschema.calculation.MultipolesEntry.total` |
| `quadrupole.q_dip` | Unmapped | — |
| `file` | Unmapped | — |
| `model` | Mapped | `runschema.method.Method.tb.xtb` |
| `model.reference` | Mapped | `runschema.method.xTB.reference` |
| `model.contribution` | Mapped | `runschema.method.xTB.hamiltonian`<br>`runschema.method.xTB.coulomb`<br>`runschema.method.xTB.repulsion`<br>`runschema.method.xTB.contributions` |
| `model.contribution.name` | Mapped | `runschema.method.Interaction.type` |
| `model.contribution.parameters` | Mapped | `runschema.method.Interaction.parameters` |
| `scf_iteration` | Mapped | `runschema.calculation.Calculation.scf_iteration` |
| `scf_iteration.step` | Mapped | `runschema.calculation.ScfIteration.energy.total.value`<br>`runschema.calculation.ScfIteration.energy.change` |
| `scf_iteration.converged` | Unmapped | — |
| `cycle` | Mapped | `runschema.calculation.Calculation` |
| `cycle.energy_total` | Mapped | `runschema.calculation.Energy.total.value` |
| `cycle.energy_change` | Mapped | `runschema.calculation.Energy.change` |
| `cycle.scf_iteration` | Mapped | `runschema.calculation.Calculation.scf_iteration` |
| `cycle.scf_iteration.step` | Mapped | `runschema.calculation.ScfIteration.energy.total.value`<br>`runschema.calculation.ScfIteration.energy.change` |
| `cycle.scf_iteration.time` | Unmapped | — |
| `converged` | Unmapped | — |
| `final_structure` | Unmapped | — |
| `final_structure.atom_labels` | Unmapped | — |
| `final_structure.atom_positions` | Unmapped | — |
| `final_single_point` | Mapped | `runschema.calculation.Calculation` |
| `traj_file` | Unmapped | — |
| `x_xtb_md_time` | Unmapped | — |
| `timestep` | Unmapped | — |
| `x_xtb_scc_accuracy` | Unmapped | — |
| `x_xtb_temperature` | Unmapped | — |
| `x_xtb_max_steps` | Unmapped | — |
| `x_xtb_block_length` | Unmapped | — |
| `x_xtb_dumpstep_trj` | Unmapped | — |
| `x_xtb_dumpstep_coords` | Unmapped | — |
| `x_xtb_h_atoms_mass` | Unmapped | — |
| `x_xtb_n_degrees_freedom` | Unmapped | — |
| `x_xtb_shake_bonds` | Unmapped | — |
| `x_xtb_berendsen` | Unmapped | — |
| `cycle` | Mapped | `simulationworkflowschema.molecular_dynamics` thermodynamics step: `runschema.calculation.Calculation.step`<br>`runschema.calculation.Calculation.temperature`<br>`runschema.calculation.Energy.total.potential`<br>`runschema.calculation.Energy.total.kinetic`<br>`runschema.calculation.Energy.total.value` |
| `program_version` | Mapped | `runschema.run.Program.version` |
| `date_start` | Mapped | `runschema.run.TimeRun.date_start` |
| `date_end` | Mapped | `runschema.run.TimeRun.date_end` |
| `calculation_setup` | Unmapped | — |
| `calculation_setup.parameter` | Unmapped | — |
| `gfnff` | Mapped | `runschema.calculation.Calculation` |
| `gfn1` | Mapped | `runschema.calculation.Calculation` |
| `gfn2` | Mapped | `runschema.calculation.Calculation` |
| `ancopt` | Mapped | `simulationworkflowschema.GeometryOptimization` |
| `md` | Mapped | `simulationworkflowschema.MolecularDynamics` |
| `property` | Mapped | `runschema.calculation.Calculation.multipoles` |
| `geometry` | Unmapped | — |
| `energy_total` | Unmapped | — |
| `gradient_norm` | Unmapped | — |
| `hl_gap` | Unmapped | — |
| `topo_file` | Unmapped | — |
| `footer` | Mapped | `runschema.calculation.Calculation.time_physical`<br>`runschema.calculation.Calculation.time_calculation` |
| `footer.end_time` | Unmapped | — |
| `footer.wall_time` | Mapped | `runschema.calculation.Calculation.time_physical`<br>`runschema.calculation.Calculation.time_calculation` |
| `footer.cpu_time` | Unmapped | — |

## xTB / CoordParser

**Summary:** 7 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `coord_unit` | Mapped | `runschema.system.Atoms.positions` |
| `positions_labels` | Mapped | `runschema.system.Atoms.positions`<br>`runschema.system.Atoms.labels` |
| `periodic` | Mapped | `runschema.system.Atoms.periodic` |
| `lattice_unit` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `lattice` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `cell_unit` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `cell` | Mapped | `runschema.system.Atoms.lattice_vectors` |

## xTB / TrajParser

**Summary:** 3 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `frame` | Mapped | `runschema.system.Atoms` |
| `frame.positions` | Mapped | `runschema.system.Atoms.positions` |
| `frame.labels` | Mapped | `runschema.system.Atoms.labels` |
