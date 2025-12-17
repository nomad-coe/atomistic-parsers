# GitHub Copilot Instructions for NOMAD Parser Development

## Overview

This document provides guidance for GitHub Copilot when working with NOMAD atomistic parsers. Each parser extracts computational results from molecular dynamics and force field simulation output files and maps them to NOMAD's unified runschema.

## Table of Contents

1. [Part 1: Editing Guidelines](#part-1-editing-guidelines)
   - Parser Feature Documentation
   - Editing Guidelines for FEATURES.yml
   - Maintenance Guidelines
2. [Part 2: Reference Documentation](#part-2-reference-documentation)
   - NOMAD Runschema Terminology
   - Common Code Patterns
3. [Part 3: Templates](#part-3-templates)
   - YAML Schema for FEATURES.yml

---

## Part 1: Editing Guidelines

## Parser Feature Documentation

Each parser has a `FEATURES.yml` file in its directory that documents its capabilities using standardized runschema terminology. These files serve as a reference for:
- Understanding what data each parser extracts
- Identifying which runschema sections are populated
- Recognizing special features and capabilities
- Maintaining consistency across parser implementations

### File Location

Parser feature files are located at:
```
atomisticparsers/{parser_name}/FEATURES.yml
```

### Editing Guidelines

**IMPORTANT**: When editing FEATURES.yml files:
- Always add a `metadata` section at the top with:
  - `last_updated`: Current timestamp in quotes (YYYY-MM-DD format, e.g., "2025-12-11")
  - `updated_by`: The model name in quotes (i.e. YOU) that made the edits (e.g., "GitHub Copilot", "Claude Sonnet 4.5", "GPT-4", etc.)
- The model name should be retained in the file to track which AI assisted with the documentation
- Do NOT annotate every line with the model name, only include it in the metadata section
- Update the timestamp each time the file is modified
- Both values must be enclosed in quotes

Example metadata section:
```yaml
metadata:
  last_updated: "2025-12-11"
  updated_by: "GitHub Copilot"

parser:
  name: GROMACS
  ...
```

### Version Control Practices

#### Commit Guidelines

When committing FEATURES.yml changes:
- Commit FEATURES.yml changes separately from code changes when possible
- Use descriptive commit messages with proper prefixes:
  - `docs: update FEATURES.yml - add OAuth2 support`
  - `feat(docs): document new workflow capabilities`
- Prefix with `docs:` or `feat(docs):` for documentation changes

#### Before Committing

Checklist:
- ✅ Ensure metadata section is updated with current timestamp
- ✅ Include your AI model name in `updated_by` field
- ✅ Validate YAML syntax (run `yamllint` or equivalent if available)
- ✅ Review diff to ensure no unintended changes

Example commit message:
```
docs: update FEATURES.yml metadata

- Added new authentication features
- Updated by: Claude Sonnet 4.5
- Last updated: "2025-12-12"
```

### Style and Formatting Guidelines

#### YAML Formatting
- **Indentation**: Use 2 spaces (no tabs)
- **Quotes**: Always quote string values in metadata section
- **Colons**: Strings containing colons must be quoted
- **Comments**: Use `#` for inline comments, keep them concise
- **Lists**: Use `-` for list items with consistent indentation

#### Content Style
- **Descriptions**: Keep concise and factual
- **Comments**: Focus on "what" the parser extracts, not "how" it's implemented
- **Consistency**: Use the same terminology across all parser FEATURES.yml files
- **Order**: Follow the template order (run, method, system, calculation, workflow)

### Common Issues and Solutions

#### Problem: YAML syntax errors after AI edits
**Solution**:
- Check indentation (must be 2 spaces, no tabs)
- Ensure strings with colons are quoted
- Validate with `yamllint` or online YAML validator
- Look for missing closing quotes or brackets

#### Problem: Merge conflicts in metadata section
**Solution**:
- Keep the most recent timestamp
- Combine model names if both made changes (e.g., "GitHub Copilot, Claude Sonnet 4.5")
- Preserve all feature additions from both versions
- Resolve conflicts in favor of more complete documentation

#### Problem: AI removed or reformatted existing content
**Solution**:
- Review `git diff` carefully before committing
- Instruct AI to "add X without modifying existing entries"
- Use explicit constraints in your AI prompts
- Revert unintended changes and re-run with clearer instructions

#### Problem: Inconsistent terminology across parsers
**Solution**:
- Review other parser FEATURES.yml files for reference
- Use standardized runschema terms from Part 2 of this document
- Consult the YAML template in Part 3
- Ask for clarification if terminology is ambiguous

---

## Part 3: Templates

### YAML Schema for FEATURES.yml

```yaml
metadata:
  last_updated: "YYYY-MM-DD"
  updated_by: "Model Name (e.g., GitHub Copilot)"

parser:
  name: Parser Name
  description: Brief description of the parser
  homepage: http://parser-homepage.org
  mainfile_patterns:
    - "*.log"
    - "*.out"
  supported_file_formats:
    - log
    - out

runschema_capabilities:
  run:
    - program  # name, version
    - time_run  # timing information

  method:
    - force_field  # force field description
    - force_field.model  # topology and interactions
    - tb  # tight-binding method (for DFTB+, xTB, BOPfox)
    - tb.xtb  # extended tight-binding specifics
    - atom_parameters  # per-atom masses, charges
    - neighbor_searching  # neighbor list parameters
    - force_calculations  # cutoffs, methods
    # Add other method components as applicable

  system:
    - atoms  # positions, species, lattice_vectors, periodic
    - atoms.velocities  # if MD parser extracts velocities
    - atoms_group  # molecules, residues, chains
    # Add other system components as applicable

  calculation:
    # Energy components (list what the parser extracts)
    - energy.total
    - energy.potential
    - energy.kinetic
    - energy.coulomb
    - energy.van_der_waals
    # Add other energy components as applicable

    # Forces and stress
    - forces.total  # if parser extracts forces
    - stress.total  # if parser extracts stress

    # Thermodynamics
    - thermodynamics.temperature
    - thermodynamics.pressure
    - thermodynamics.volume
    # Add other thermodynamic properties as applicable

    # Time-dependent properties
    - time_physical  # for MD trajectories
    - step  # time step number

    # Vibrational properties
    - vibrational_frequencies  # if phonons/vibrations calculated

    # Constraints
    - constraint  # geometric constraints on atoms

  workflow:
    - single_point  # if single frame/calculation
    - geometry_optimization  # if energy minimization supported
    - molecular_dynamics  # if MD supported
    - elastic  # if elastic properties calculated
    # Add other workflow types as applicable

special_features:
  # List parser-specific advanced capabilities
  - "Feature description 1"
  - "Feature description 2"

notes:
  # Optional: Additional implementation notes
  - "Note 1"
  - "Note 2"
```

---

## Appendix: NOMAD Runschema Terminology Reference

The runschema is NOMAD's unified data model for computational materials science. It consists of hierarchical sections:

### 1. Run Section (`runschema.run`)

Top-level container for a complete calculation run.

**Key components:**
- `Program` - Software metadata
  - `name` - Code name (e.g., "GROMACS", "LAMMPS")
  - `version` - Software version
  - `compilation_host` - Where compiled
- `TimeRun` - Execution timing
  - `date_start`, `date_end` - Timestamps
  - `cpu1_start`, `cpu1_end` - CPU time
  - `wall_start`, `wall_end` - Wall clock time

### 2. Method Section (`runschema.method`)

Describes the computational methodology used.

**Key components:**

#### Force Field Method
- `ForceField` - Force field description
  - `model` - Force field model/topology
  - `Model` with `Interaction` - Force field interactions
    - `type` - Interaction type: "bonds", "angles", "dihedrals", "impropers", "pairs", "coulomb", "lennard_jones", "buckingham", "morse", "lj_cut"
    - `parameters` - Interaction parameters
    - `atom_labels` - Atoms involved
    - `atom_indices` - Atom indices
    - `n_interactions` - Number of interactions of this type
  - Force fields supported in this repository: AMBER, GROMOS, OPLS, COMPASS (via specific parsers or force field parameters)

#### Tight-Binding Methods (Semi-empirical)
- `TB` - Tight-binding method container
  - `name` - Method name: "DFTB", "xTB"
  - `xTB` - Extended tight-binding (xTB) specifics
    - `name` - xTB variant: "GFN1-xTB", "GFN2-xTB", "GFN-FF"
    - `hamiltonian` - Hamiltonian interactions
    - `coulomb` - Coulomb interactions
    - `repulsion` - Repulsion interactions
    - `contributions` - Other contributions (dispersion, halogen bonding, etc.)
    - `reference` - Method reference/citation

#### Atom-Specific Parameters
- `AtomParameters` - Per-atom settings
  - `mass` - Atomic mass
  - `charge` - Atomic charge
  - `label` - Atom label/type
  - `atom_number` - Element number
  - `kind` - Atom kind/type in force field

#### Neighbor Searching
- `NeighborSearching` - Neighbor list parameters
  - `cutoff_radius` - Neighbor search cutoff
  - `update_frequency` - Neighbor list update frequency

#### Force Calculations
- `ForceCalculations` - Non-bonded force parameters
  - `vdw_cutoff` - Van der Waals cutoff
  - `coulomb_cutoff` - Coulomb cutoff
  - `vdw_method` - VdW calculation method
  - `coulomb_method` - Coulomb calculation method

### 3. System Section (`runschema.system`)

Describes the atomic structure and configuration.

**Key components:**

- `System` - Complete atomic configuration
  - `Atoms` - Atomic structure
    - `labels` - Atomic symbols (e.g., ["C", "C", "O", "H"])
    - `positions` - Atomic coordinates in Cartesian (Å)
    - `lattice_vectors` - Unit cell vectors (3x3 matrix, Å)
    - `periodic` - Periodicity flags [x, y, z]
    - `velocities` - Atomic velocities (for MD)
  - `AtomsGroup` - Subsets of atoms (molecules, residues, chains)
    - `label` - Group name (e.g., "SOL", "protein")
    - `atom_indices` - Indices of atoms in group
    - `is_molecule` - Whether group is a molecule
  - `Symmetry` - Space group and symmetry operations (if applicable)
  - `Constraint` - Constrained atoms or geometric constraints

### 4. Calculation Section (`runschema.calculation`)

Contains results from a single-point or time-step calculation.

**Key components:**

#### Energy
- `Energy` with `EnergyEntry` - Energy values
  - `total` - Total energy (most common)
  - `potential` - Potential energy
  - `kinetic` - Kinetic energy
  - `coulomb` - Coulombic energy
  - `van_der_waals` - Van der Waals energy
  - `bonded` - Bonded interactions energy
  - `angle` - Angle interaction energy
  - `dihedral` - Dihedral interaction energy
  - `improper` - Improper dihedral energy
  - Each entry has:
    - `value` - Energy value
    - `contributions` - Breakdown of components

#### Forces and Stress
- `Forces` with `ForcesEntry` - Atomic forces
  - `total` - Total forces on atoms (N_atoms x 3)
  - `value` - Force array
  - `contributions` - Force component breakdown

- `Stress` with `StressEntry` - Stress tensor
  - `total` - Total stress (3x3 matrix)
  - `value` - Stress tensor
  - `contributions` - Stress component breakdown

#### Thermodynamics
- `Thermodynamics` - Thermodynamic properties
  - `pressure` - Pressure
  - `temperature` - Temperature
  - `enthalpy` - Enthalpy
  - `entropy` - Entropy
  - `volume` - System volume

#### Time-dependent Properties
- `time_physical` - Physical time (for MD trajectories)
- `time_calculation` - Computation time
- `step` - Time step number

#### Vibrational Properties
- `VibrationalFrequencies` - Phonon/vibrational frequencies
  - `value` - Frequencies (cm⁻¹)
  - `intensities` - IR intensities
  - `raman_intensities` - Raman intensities

#### Constraints
- `Constraint` - Geometric constraints
  - `kind` - Constraint type: "fix_xyz", "fix_xy", "fix_xz", "fix_yz", "fix_x", "fix_y", "fix_z"
  - `atom_indices` - Constrained atoms

### 5. Workflow Section (`simulationworkflowschema`)

Describes the type of calculation workflow.

**Workflow types:**

#### SinglePoint
Basic single-point energy calculation or single MD frame.

#### GeometryOptimization
Structure relaxation to minimize forces (energy minimization).
- `GeometryOptimizationMethod` - Optimization settings
  - `method` - Optimizer algorithm
    - "steepest_descent", "conjugate_gradient", "L-BFGS"
  - `type` - Optimization type: "atomic" (atomic positions), "cell" (lattice), "atomic_and_cell"
  - `convergence_tolerance_force_maximum` - Force convergence criterion
  - `convergence_tolerance_energy_difference` - Energy convergence criterion
  - `convergence_tolerance_displacement_maximum` - Displacement convergence criterion
  - `optimization_steps_maximum` - Maximum number of optimization steps
  - `save_frequency` - Frequency of saving optimization trajectory
- `GeometryOptimizationResults` - Optimization results
  - `energies` - Energy at each optimization step
  - `steps` - Step numbers
  - `optimization_steps` - Total number of optimization steps
  - `final_energy_difference` - Final energy change
  - `final_force_maximum` - Maximum force in final structure
  - `is_converged` - Whether optimization converged

#### MolecularDynamics
Time-dependent simulation.
- `MolecularDynamicsMethod` - MD settings
  - `ensemble_type` - Statistical ensemble
    - "NVE" (microcanonical), "NVT" (canonical), "NPT" (isothermal-isobaric)
  - `timestep` - Integration timestep
  - `integrator_type` - Integration algorithm
    - "leap_frog", "velocity_verlet", "langevin", "verlet"
  - `n_steps` - Total number of MD steps
  - `coordinate_save_frequency` - Frequency of saving coordinates
  - `velocity_save_frequency` - Frequency of saving velocities
  - `force_save_frequency` - Frequency of saving forces
  - `thermodynamics_save_frequency` - Frequency of saving thermodynamic data
  - `ThermostatParameters` - Temperature control
    - `type` - "Nose-Hoover", "Berendsen", "Langevin", "velocity_rescaling", "Andersen"
    - `target_temperature` - Target temperature
    - `coupling_constant` - Thermostat coupling time
  - `BarostatParameters` - Pressure control (NPT only)
    - `type` - "Berendsen", "Parrinello-Rahman", "MTTK"
    - `target_pressure` - Target pressure
    - `coupling_constant` - Barostat coupling time
- `MolecularDynamicsResults` - MD calculation results
  - `EnsembleProperty` - Ensemble-averaged properties
    - `radial_distribution_functions` - Radial distribution function (RDF/g(r))
    - `radial_distribution_function_values` - RDF values over trajectory
  - `CorrelationFunction` - Time correlation functions
    - `mean_squared_displacements` - Mean squared displacement (MSD)
    - `mean_squared_displacement_values` - MSD values over time
    - `diffusion_constant` - Diffusion coefficient (from MSD)

#### Elastic
Elastic properties calculation.
- `ElasticMethod` - Elastic calculation settings
  - Strain method, energy/stress approach
- `ElasticResults` - Elastic calculation results
  - `elastic_constants_matrix_second_order` - Elastic constant matrix (C_ij)
  - `compliance_matrix_second_order` - Compliance matrix (S_ij)
  - `bulk_modulus` - Bulk modulus
  - `shear_modulus` - Shear modulus

### Common Patterns in Parser Implementation
metadata:
  last_updated: "YYYY-MM-DD"
  updated_by: "Model Name (e.g., GitHub Copilot)"

parser:
  name: "Parser Name"
  description: "Brief description"
  homepage: "https://..."
  mainfile_patterns:
    - "pattern1"
    - "pattern2"
  supported_file_formats:
    - "format1"
    - "format2"

runschema_capabilities:
  run:
    - program  # name, version
    - time_run  # timing information

  method:
    - force_field  # force field description
    - force_field.model  # topology and interactions
    - tb  # tight-binding method (for DFTB+, xTB, BOPfox)
    - tb.xtb  # extended tight-binding specifics
    - atom_parameters  # per-atom masses, charges
    - neighbor_searching  # neighbor list parameters
    - force_calculations  # cutoffs, methods
    # Add other method components as applicable

  system:
    - atoms  # positions, species, lattice_vectors, periodic
    - atoms.velocities  # if MD parser extracts velocities
    - atoms_group  # molecules, residues, chains
    # Add other system components as applicable

  calculation:
    # Energy components (list what the parser extracts)
    - energy.total
    - energy.potential
    - energy.kinetic
    - energy.coulomb
    - energy.van_der_waals
    # Add other energy components as applicable

    # Forces and stress
    - forces.total  # if parser extracts forces
    - stress.total  # if parser extracts stress

    # Thermodynamics
    - thermodynamics.temperature
    - thermodynamics.pressure
    - thermodynamics.volume
    # Add other thermodynamic properties as applicable

    # Time-dependent properties
    - time_physical  # for MD trajectories
    - step  # time step number

    # Vibrational properties
    - vibrational_frequencies  # if phonons/vibrations calculated

    # Constraints
    - constraint  # geometric constraints on atoms

  workflow:
    - single_point  # if single frame/calculation
    - geometry_optimization  # if energy minimization supported
    - molecular_dynamics  # if MD supported
    - elastic  # if elastic properties calculated
    # Add other workflow types as applicable

special_features:
  # List parser-specific advanced capabilities
  - "Feature description 1"
  - "Feature description 2"

notes:
  # Optional: Additional implementation notes
  - "Note 1"
  - "Note 2"
```

### Guidelines for Maintaining Feature Files

#### When Adding New Parsers

1. Create `FEATURES.yml` in the parser directory
2. Add metadata section with current date (in quotes) and your model name (in quotes)
3. Analyze the parser implementation to identify:
   - Which runschema sections are populated
   - What properties are extracted
   - Special capabilities or unique features
4. Use the YAML schema above as a template
5. Focus on runschema terminology, not code-specific names
6. **Do NOT annotate individual lines** with model names - only update the metadata section
7. Keep the file clean and readable without inline attribution comments

#### When Updating Existing Parsers

1. **Always update the `metadata` section**:
   - Update `last_updated` with current date (in quotes)
   - Update `updated_by` with your model name (in quotes)
   - If multiple models contributed, combine names: `"GitHub Copilot, Claude Sonnet 4.5"`
2. Add new runschema sections to `runschema_capabilities`
3. Document new special features
4. Keep descriptions concise and standardized
5. **Do NOT add inline comments** attributing specific lines to specific models
6. Review the diff to ensure you're only adding/modifying intended content

#### Best Practices

- Always update metadata when editing
- Use runschema terminology consistently across all feature files
- List only capabilities that are actually implemented
- Group related capabilities logically
- Include parser-specific features in `special_features`
- Keep descriptions focused on "what" not "how"
- Reference official schema documentation for ambiguous cases

---

## Part 2: Reference Documentation

### NOMAD Runschema Terminology

The runschema is NOMAD's unified data model for computational materials science. It consists of hierarchical sections:

### Schema Documentation References

- Main runschema: `packages/nomad-schema-plugin-run/runschema/`
  - `run.py` - Run section
  - `method.py` - Method section
  - `system.py` - System section
  - `calculation.py` - Calculation section
- Workflow schema: `packages/nomad-schema-plugin-simulation-workflow/simulationworkflowschema/`

### Common Patterns in Parser Implementation

#### Energy Extraction Pattern
```python
sec_energy = calculation.Energy()
sec_energy.total = EnergyEntry(value=total_energy * ureg.kJ / ureg.mol)
sec_energy.potential = EnergyEntry(value=potential_energy * ureg.kJ / ureg.mol)
sec_energy.kinetic = EnergyEntry(value=kinetic_energy * ureg.kJ / ureg.mol)
sec_calculation.energy = sec_energy
```

#### Forces Extraction Pattern
```python
sec_forces = calculation.Forces()
sec_forces.total = ForcesEntry(value=forces_array * ureg.kJ / ureg.mol / ureg.nm)
sec_calculation.forces = sec_forces
```

#### Force Field Pattern
```python
sec_force_field = method.ForceField()
sec_model = method.Model()
sec_model.contributions.append(
    method.Interaction(
        type='bonds',
        parameters={'k': k_value, 'r0': r0_value},
        atom_labels=['C', 'C'],
        atom_indices=[i, j],
        n_interactions=n_bonds
    )
)
sec_force_field.model = sec_model
sec_method.force_field = sec_force_field
```

#### Tight-Binding Pattern
```python
sec_tb = method.TB()
sec_method.tb = sec_tb
sec_tb.name = 'DFTB'  # or 'xTB'

# For xTB methods
sec_xtb = method.xTB()
sec_tb.xtb = sec_xtb
sec_xtb.name = 'GFN2-xTB'
sec_xtb.hamiltonian.append(
    method.Interaction(
        type='hamiltonian',
        parameters={'shell': shell_params}
    )
)
```

#### Vibrational Frequencies Pattern
```python
from runschema.calculation import VibrationalFrequencies

sec_vibrations = VibrationalFrequencies()
sec_vibrations.value = frequencies * ureg.cm**-1
sec_vibrations.intensities = ir_intensities
sec_calculation.vibrational_frequencies.append(sec_vibrations)
```

#### Constraints Pattern
```python
from runschema.system import Constraint

constraints = []
for constrained_atoms in fixed_atom_lists:
    constraint = Constraint()
    constraint.kind = 'fix_xyz'  # or 'fix_xy', 'fix_x', etc.
    constraint.atom_indices = constrained_atoms
    constraints.append(constraint)
sec_system.constraint = constraints
```

#### Workflow Detection Pattern
```python
if self.is_energy_minimization():
    workflow = GeometryOptimization()
    workflow.method = GeometryOptimizationMethod(
        method="steepest_descent",
        convergence_tolerance_force_maximum=1e-3 * ureg.kJ / ureg.mol / ureg.nm
    )
elif self.is_molecular_dynamics():
    workflow = MolecularDynamics()
    workflow.method = MolecularDynamicsMethod(
        ensemble_type="NVT",
        timestep=2.0 * ureg.fs,
        integrator_type="leap_frog"
    )
elif self.is_elastic_calculation():
    workflow = Elastic()
    workflow.method = ElasticMethod()
    workflow.results = ElasticResults()
    workflow.results.elastic_constants_matrix_second_order = elastic_matrix * ureg.GPa
```

### Normalized MD Properties

**Important**: Many MD workflow properties are automatically populated by NOMAD's normalizers, not by parsers. These normalized properties are calculated from trajectory data and thermodynamic time series stored in the `run.calculation[]` sections.

#### Properties Populated by Normalization

The following properties are automatically calculated for all MD workflows and should **not** be populated by parsers:

#### From ThermodynamicsResults (inherited by MolecularDynamicsResults):
- `temperature` - array of temperature values from calculations
- `pressure` - array of pressure values from calculations
- `helmholtz_free_energy` - free energy calculations
- `heat_capacity_c_v` / `heat_capacity_c_p` - heat capacities
- `heat_capacity_c_v_specific` - specific heat capacity (derived)
- `vibrational_free_energy` - vibrational contributions
- `vibrational_internal_energy` - vibrational energy
- `vibrational_entropy` - entropy calculations
- `gibbs_free_energy` - Gibbs free energy
- `entropy` - total entropy
- `enthalpy` - enthalpy calculations
- `internal_energy` - internal energy

#### From MolecularDynamicsResults:
- `radial_distribution_functions` (RDF) - calculated from trajectory using MDAnalysis
  - Computed for molecular bead groups
  - Multiple time intervals for convergence analysis
  - Contains bins and values for each molecular pair type
- `mean_squared_displacements` (MSD) - calculated from trajectory
  - Computed per molecule type
  - Includes diffusion constant calculation via Einstein relation
  - Error estimates via Pearson correlation coefficient
- `radius_of_gyration` - calculated from trajectory for polymers
  - Computed per molecule/chain
  - Stored in both calculation and workflow results
  - Time-dependent property
- `correlation_functions` - generic time correlation functions
  - Can be direction-specific (x, y, z, xyz)
- `ensemble_properties` - generic ensemble averages
  - Any static observable from trajectory averaging

#### What Parsers Should Populate

Parsers should focus on:
1. **Raw trajectory data**: Store system snapshots in `run.system[]` with positions, velocities, forces
2. **Per-frame thermodynamics**: Store temperature, pressure, volume, energy in `calculation.thermodynamics`
3. **Workflow method parameters**: ensemble type, thermostat, barostat, timestep, etc.
4. **Basic workflow results**: `n_steps`, `trajectory` reference, `finished_normally`

The normalizers will automatically:
- Calculate ensemble averages
- Compute structural properties (RDF, Rg)
- Determine transport properties (MSD, diffusion)
- Generate correlation functions

#### Example: What NOT to do
```python
# DON'T calculate RDF in parser - normalizer does this
workflow.results.radial_distribution_functions = compute_rdf(trajectory)

# DON'T calculate MSD in parser - normalizer does this
workflow.results.mean_squared_displacements = compute_msd(trajectory)

# DON'T calculate radius of gyration in parser - normalizer does this
workflow.results.radius_of_gyration = compute_rg(trajectory)
```

#### Example: What TO do
```python
# DO store trajectory snapshots
for frame in trajectory:
    sec_system = run.m_create(System)
    sec_atoms = sec_system.m_create(Atoms)
    sec_atoms.positions = frame.positions * ureg.angstrom
    sec_atoms.velocities = frame.velocities * ureg.angstrom / ureg.fs

    # DO store per-frame thermodynamics
    sec_calc = run.m_create(Calculation)
    sec_calc.system_ref = sec_system
    sec_thermo = sec_calc.m_create(ThermodynamicValues)
    sec_thermo.temperature = frame.temperature * ureg.kelvin
    sec_thermo.pressure = frame.pressure * ureg.bar
    sec_thermo.volume = frame.volume * ureg.angstrom**3

# DO set workflow method parameters
workflow.method.ensemble_type = "NPT"
workflow.method.timestep = 2.0 * ureg.fs

# DO set basic workflow results
workflow.results.n_steps = len(trajectory)
workflow.results.trajectory = run.system  # reference to systems
workflow.results.finished_normally = True
```

## YAML Validation and Quality Assurance

### Visual Validation (Works in GitHub Copilot)

When editing FEATURES.yml files, watch for these **common YAML syntax errors**:

| Visual Indicator | Error Type | Fix |
|-----------------|------------|-----|
| String contains `:` without quotes | `mapping values are not allowed here` | Add quotes: `"value: with colon"` |
| Misaligned items at same level | Inconsistent indentation | Use exactly 2 spaces per level |
| Jagged/uneven indentation | Incorrect nesting | Align child items 2 spaces from parent |
| Special characters in unquoted text | Parser errors | Add quotes around the value |

**Manual validation checklist** (no tools required):
- [ ] All lines use spaces (not tabs) - look for irregular spacing
- [ ] Indentation increases by exactly 2 spaces per level
- [ ] Metadata section: both `last_updated` and `last_updated_by` are quoted strings
- [ ] Strings with colons, special chars, or starting with `#` are quoted
- [ ] List items at the same level are aligned vertically
- [ ] No trailing spaces at end of lines

### Automated Validation Tools (Local Development)

For local development environments (not available in GitHub Copilot web interface):

**Command-line validation** (if yamllint is installed):
```bash
yamllint atomisticparsers/gromacs/FEATURES.yml
```

**Python validation script** (if Python available):
```python
import yaml
with open('FEATURES.yml', 'r') as f:
    try:
        yaml.safe_load(f)
        print("✓ Valid YAML")
    except yaml.YAMLError as e:
        print(f"✗ YAML Error: {e}")
```

## Usage with GitHub Copilot

When working on parser code, Copilot can reference these feature files to:
- Suggest appropriate runschema sections for extracted data
- Recommend common patterns for similar properties
- Identify missing capabilities that should be implemented
- Ensure consistency with other parsers

To help Copilot understand your intent:
- Mention the parser name in comments
- Reference runschema sections explicitly
- Use standard property names from the schema
- Comment on what data you're extracting in runschema terms
