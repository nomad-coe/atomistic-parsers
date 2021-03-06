#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import numpy as np
import logging
import re
import datetime

import panedr
try:
    import MDAnalysis
    from MDAnalysis.topology.tpr import utils as tpr_utils, setting as tpr_setting
except Exception:
    logging.warn('Required module MDAnalysis not found.')
    MDAnalysis = False

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity, FileParser
from nomad.datamodel.metainfo.simulation.run import Run, Program, TimeRun
from nomad.datamodel.metainfo.simulation.method import (
    Method, ForceField, Model, Interaction, AtomParameters
)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms, AtomsGroup
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry, Forces, ForcesEntry, Thermodynamics
)
from nomad.datamodel.metainfo.workflow import (
    DiffusionConstantValues, MeanSquaredDisplacement, MeanSquaredDisplacementValues,
    RadialDistributionFunction, RadialDistributionFunctionValues,
    Workflow, MolecularDynamics
)
from .metainfo.gromacs import x_gromacs_section_control_parameters, x_gromacs_section_input_output_files
from atomisticparsers.utils import MDAnalysisParser


MOL = 6.022140857e+23


class GromacsLogParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        def str_to_header(val_in):
            val = [v.split(':', 1) for v in val_in.strip().split('\n')]
            return {v[0].strip(): v[1].strip() for v in val if len(v) == 2}

        def str_to_input_parameters(val_in):
            re_array = re.compile(r'\s*([\w\-]+)\[[\d ]+\]\s*=\s*\{*(.+)')
            re_scalar = re.compile(r'\s*([\w\-]+)\s*=\s*(.+)')
            parameters = dict()
            val = val_in.strip().split('\n')
            for val_n in val:
                val_scalar = re_scalar.match(val_n)
                if val_scalar:
                    parameters[val_scalar.group(1)] = val_scalar.group(2)
                    continue
                val_array = re_array.match(val_n)
                if val_array:
                    parameters.setdefault(val_array.group(1), [])
                    value = [float(v) for v in val_array.group(2).rstrip('}').split(',')]
                    parameters[val_array.group(1)].append(value[0] if len(value) == 1 else value)
            return parameters

        def str_to_energies(val_in):
            energy_keys_re = re.compile(r'(.+?)(?:  |\Z| P)')
            keys = []
            values = []
            energies = dict()
            for val in val_in.strip().split('\n'):
                val = val.strip()
                if val[0].isalpha():
                    keys = [k.strip() for k in energy_keys_re.findall(val)]
                    keys = ['P%s' % k if k.startswith('res') else k for k in keys if k]
                else:
                    values = val.split()
                    for n, key in enumerate(keys):

                        if key == 'Temperature':
                            energies[key] = float(values[n]) * ureg.kelvin
                        elif key.startswith('Pres'):
                            key = key.rstrip(' (bar)')
                            energies[key] = float(values[n]) * ureg.bar
                        else:
                            energies[key] = float(values[n]) / MOL * ureg.kJ
            return energies

        def str_to_step_info(val_in):
            val = val_in.strip().split('\n')
            keys = val[0].split()
            values = [float(v) for v in val[1].split()]
            return {key: values[n] for n, key in enumerate(keys)}

        thermo_quantities = [
            Quantity(
                'energies',
                r'Energies \(kJ/mol\)\s*([\s\S]+?)\n\n',
                str_operation=str_to_energies, convert=False),
            Quantity(
                'step_info',
                r'(Step.+\n[\d\.\- ]+)',
                str_operation=str_to_step_info, convert=False)]

        self._quantities = [
            Quantity('time_start', r'Log file opened on (.+)', flatten=False),
            Quantity(
                'host_info',
                r'Host:\s*(\S+)\s*pid:\s*(\d+)\s*rank ID:\s*(\d+)\s*number of ranks:\s*(\d*)'),
            Quantity('module_version', r'GROMACS:\s*(.+?),\s*VERSION\s*(\S+)', flatten=False),
            Quantity('execution_path', r'Executable:\s*(.+)'),
            Quantity('working_path', r'Data prefix:\s*(.+)'),
            # TODO cannot understand treatment of the command line in the old parser
            Quantity(
                'header',
                r'(?:GROMACS|Gromacs) (version:[\s\S]+?)\n\n', str_operation=str_to_header),
            Quantity(
                'input_parameters',
                r'Input Parameters:\s*([\s\S]+?)\n\n', str_operation=str_to_input_parameters),
            Quantity(
                'step',
                r'(Step\s*Time[\s\S]+?Energies[\s\S]+?\n\n)',
                repeats=True, sub_parser=TextParser(quantities=thermo_quantities)),
            Quantity(
                'averages',
                r'A V E R A G E S  ====>([\s\S]+?\n\n\n)',
                sub_parser=TextParser(quantities=thermo_quantities)),
            Quantity('time_end', r'Finished \S+ on rank \d+ (.+)', flatten=False)]

    def get_pbc(self):
        pbc = self.get('input_parameters', {}).get('pbc', 'xyz')
        return ['x' in pbc, 'y' in pbc, 'z' in pbc]

    def get_sampling_settings(self):
        input_parameters = self.get('input_parameters', {})
        integrator = input_parameters.get('integrator', 'md').lower()
        if integrator in ['l-bfgs', 'cg', 'steep']:
            sampling_method = 'geometry_optimization'
        elif integrator in ['bd']:
            sampling_method = 'langevin_dynamics'
        else:
            sampling_method = 'molecular_dynamics'

        ensemble_type = 'NVE' if sampling_method == 'molecular_dynamics' else None
        tcoupl = input_parameters.get('tcoupl', 'no').lower()
        if tcoupl != 'no':
            ensemble_type = 'NVT'
            pcoupl = input_parameters.get('pcoupl', 'no').lower()
            if pcoupl != 'no':
                ensemble_type = 'NPT'

        return dict(
            sampling_method=sampling_method, integrator_type=integrator,
            ensemble_type=ensemble_type)

    def get_tpstat_settings(self):
        input_parameters = self.get('input_parameters', {})
        target_T = input_parameters.get('ref-t', 0) * ureg.kelvin

        thermostat_type = None
        tcoupl = input_parameters.get('tcoupl', 'no').lower()
        if tcoupl != 'no':
            thermostat_type = 'Velocity Rescaling' if tcoupl == 'v-rescale' else tcoupl.title()

        thermostat_tau = input_parameters.get('tau-t', 0) * ureg.ps

        # TODO infer langevin_gamma [s] from bd_fric
        # bd_fric = self.get('bd-fric', 0, unit='amu/ps')
        langevin_gamma = None

        target_P = input_parameters.get('ref-p', 0) * ureg.bar
        # if P is array e.g. for non-isotropic pressures, get average since metainfo is float
        if hasattr(target_P, 'shape'):
            target_P = np.average(target_P)

        barostat_type = None
        pcoupl = input_parameters.get('pcoupl', 'no').lower()
        if pcoupl != 'no':
            barostat_type = pcoupl.title()

        barostat_tau = input_parameters.get('tau-p', 0) * ureg.ps

        return dict(
            target_T=target_T, thermostat_type=thermostat_type, thermostat_tau=thermostat_tau,
            target_P=target_P, barostat_type=barostat_type, barostat_tau=barostat_tau,
            langevin_gamma=langevin_gamma)


class GromacsEDRParser(FileParser):
    def __init__(self):
        super().__init__(None)
        self._energy_keys = [
            'LJ (SR)', 'Coulomb (SR)', 'Potential', 'Kinetic En.', 'Total Energy',
            'Vir-XX', 'Vir-XY', 'Vir-XZ', 'Vir-YX', 'Vir-YY', 'Vir-YZ', 'Vir-ZX', 'Vir-ZY',
            'Vir-ZZ']
        self._pressure_keys = [
            'Pressure', 'Pres-XX', 'Pres-XY', 'Pres-XZ', 'Pres-YX', 'Pres-YY', 'Pres-YZ',
            'Pres-ZX', 'Pres-ZY', 'Pres-ZZ']
        self._temperature_keys = ['Temperature']
        self._time_keys = ['Time']

    @property
    def fileedr(self):
        if self._file_handler is None:
            try:
                self._file_handler = panedr.edr_to_df(self.mainfile)
            except Exception:
                self.logger.error('Error reading edr file.')

        return self._file_handler

    def parse(self, key):
        if self.fileedr is None:
            return

        val = self.fileedr.get(key, None)
        if self._results is None:
            self._results = dict()

        if val is not None:
            val = np.asarray(val)
        if key in self._energy_keys:
            val = val / MOL * ureg.kJ
        elif key in self._temperature_keys:
            val = val * ureg.kelvin
        elif key in self._pressure_keys:
            val = val * ureg.bar
        elif key in self._time_keys:
            val = val * ureg.ps

        self._results[key] = val

    def keys(self):
        return list(self.fileedr.keys())

    @property
    def length(self):
        return self.fileedr.shape[0]


class GromacsMDAnalysisParser(MDAnalysisParser):
    def __init__(self):
        super().__init__(None)

    def get_interactions(self):
        interactions = super().get_interactions()

        # add force field parameters
        try:
            interactions.extend(self.get_force_field_parameters())
        except Exception:
            self.logger.error('Error parsing force field parameters.')

        self._results['interactions'] = interactions

        return interactions

    def get_force_field_parameters(self):
        # read force field parameters not saved by MDAnalysis
        # copied from MDAnalysis.topology.tpr.utils
        # TODO maybe a better implementation exists
        if MDAnalysis.__version__ != '2.0.0':
            self.logger.warn('Incompatible version of MDAnalysis.')

        with open(self.mainfile, 'rb') as f:
            data = tpr_utils.TPXUnpacker(f.read())

        interactions = []

        # read header
        header = tpr_utils.read_tpxheader(data)
        # address compatibility issue
        if header.fver >= tpr_setting.tpxv_AddSizeField and header.fgen >= 27:
            actual_body_size = len(data.get_buffer()) - data.get_position()
            if actual_body_size == 4 * header.sizeOfTprBody:
                self.logger.error('Unsupported tpr format.')
                return interactions
            data = tpr_utils.TPXUnpacker2020.from_unpacker(data)

        # read other unimportant parts
        if header.bBox:
            tpr_utils.extract_box_info(data, header.fver)
        if header.ngtc > 0:
            if header.fver < 69:
                tpr_utils.ndo_real(data, header.ngtc)
            tpr_utils.ndo_real(data, header.ngtc)
        if not header.bTop:
            return interactions

        tpr_utils.do_symstr(data, tpr_utils.do_symtab(data))
        data.unpack_int()
        ntypes = data.unpack_int()
        # functional types
        functypes = tpr_utils.ndo_int(data, ntypes)
        data.unpack_double() if header.fver >= 66 else 12.0
        data.unpack_real()
        # read the ffparams
        for i in functypes:
            parameters = []
            if i in [
                tpr_setting.F_ANGLES, tpr_setting.F_G96ANGLES,
                tpr_setting.F_BONDS, tpr_setting.F_G96BONDS,
                tpr_setting.F_HARMONIC, tpr_setting.F_IDIHS
            ]:
                parameters.append(data.unpack_real())  # rA
                parameters.append(data.unpack_real())  # krA
                parameters.append(data.unpack_real())  # rB
                parameters.append(data.unpack_real())  # krB

            elif i in [tpr_setting.F_RESTRANGLES]:
                parameters.append(data.unpack_real())  # harmonic.rA
                parameters.append(data.unpack_real())  # harmonic.krA
            elif i in [tpr_setting.F_LINEAR_ANGLES]:
                parameters.append(data.unpack_real())  # linangle.klinA
                parameters.append(data.unpack_real())  # linangle.aA
                parameters.append(data.unpack_real())  # linangle.klinB
                parameters.append(data.unpack_real())  # linangle.aB);
            elif i in [tpr_setting.F_FENEBONDS]:
                parameters.append(data.unpack_real())  # fene.bm
                parameters.append(data.unpack_real())  # fene.kb
            elif i in [tpr_setting.F_RESTRBONDS]:
                parameters.append(data.unpack_real())  # restraint.lowA
                parameters.append(data.unpack_real())  # restraint.up1A
                parameters.append(data.unpack_real())  # restraint.up2A
                parameters.append(data.unpack_real())  # restraint.kA
                parameters.append(data.unpack_real())  # restraint.lowB
                parameters.append(data.unpack_real())  # restraint.up1B
                parameters.append(data.unpack_real())  # restraint.up2B
                parameters.append(data.unpack_real())  # restraint.kB
            elif i in [
                tpr_setting.F_TABBONDS, tpr_setting.F_TABBONDSNC,
                tpr_setting.F_TABANGLES, tpr_setting.F_TABDIHS,
            ]:
                parameters.append(data.unpack_real())  # tab.kA
                parameters.append(data.unpack_int())  # tab.table
                parameters.append(data.unpack_real())  # tab.kB
            elif i in [tpr_setting.F_CROSS_BOND_BONDS]:
                parameters.append(data.unpack_real())  # cross_bb.r1e
                parameters.append(data.unpack_real())  # cross_bb.r2e
                parameters.append(data.unpack_real())  # cross_bb.krr
            elif i in [tpr_setting.F_CROSS_BOND_ANGLES]:
                parameters.append(data.unpack_real())  # cross_ba.r1e
                parameters.append(data.unpack_real())  # cross_ba.r2e
                parameters.append(data.unpack_real())  # cross_ba.r3e
                parameters.append(data.unpack_real())  # cross_ba.krt
            elif i in [tpr_setting.F_UREY_BRADLEY]:
                parameters.append(data.unpack_real())  # u_b.theta
                parameters.append(data.unpack_real())  # u_b.ktheta
                parameters.append(data.unpack_real())  # u_b.r13
                parameters.append(data.unpack_real())  # u_b.kUB
                if header.fver >= 79:
                    parameters.append(data.unpack_real())  # u_b.thetaB
                    parameters.append(data.unpack_real())  # u_b.kthetaB
                    parameters.append(data.unpack_real())  # u_b.r13B
                    parameters.append(data.unpack_real())  # u_b.kUBB
            elif i in [tpr_setting.F_QUARTIC_ANGLES]:
                parameters.append(data.unpack_real())  # qangle.theta
                parameters.append(tpr_utils.ndo_real(data, 5))   # qangle.c
            elif i in [tpr_setting.F_BHAM]:
                parameters.append(data.unpack_real())  # bham.a
                parameters.append(data.unpack_real())  # bham.b
                parameters.append(data.unpack_real())  # bham.c
            elif i in [tpr_setting.F_MORSE]:
                parameters.append(data.unpack_real())  # morse.b0
                parameters.append(data.unpack_real())  # morse.cb
                parameters.append(data.unpack_real())  # morse.beta
                if header.fver >= 79:
                    parameters.append(data.unpack_real())  # morse.b0B
                    parameters.append(data.unpack_real())  # morse.cbB
                    parameters.append(data.unpack_real())  # morse.betaB
            elif i in [tpr_setting.F_CUBICBONDS]:
                parameters.append(data.unpack_real())  # cubic.b0g
                parameters.append(data.unpack_real())  # cubic.kb
                parameters.append(data.unpack_real())  # cubic.kcub
            elif i in [tpr_setting.F_CONNBONDS]:
                pass
            elif i in [tpr_setting.F_POLARIZATION]:
                parameters.append(data.unpack_real())  # polarize.alpha
            elif i in [tpr_setting.F_ANHARM_POL]:
                parameters.append(data.unpack_real())  # anharm_polarize.alpha
                parameters.append(data.unpack_real())  # anharm_polarize.drcut
                parameters.append(data.unpack_real())  # anharm_polarize.khyp
            elif i in [tpr_setting.F_WATER_POL]:
                parameters.append(data.unpack_real())  # wpol.al_x
                parameters.append(data.unpack_real())  # wpol.al_y
                parameters.append(data.unpack_real())  # wpol.al_z
                parameters.append(data.unpack_real())  # wpol.rOH
                parameters.append(data.unpack_real())  # wpol.rHH
                parameters.append(data.unpack_real())  # wpol.rOD
            elif i in [tpr_setting.F_THOLE_POL]:
                parameters.append(data.unpack_real())  # thole.a
                parameters.append(data.unpack_real())  # thole.alpha1
                parameters.append(data.unpack_real())  # thole.alpha2
                parameters.append(data.unpack_real())  # thole.rfac

            elif i in [tpr_setting.F_LJ]:
                parameters.append(data.unpack_real())  # lj_c6
                parameters.append(data.unpack_real())  # lj_c9
            elif i in [tpr_setting.F_LJ14]:
                parameters.append(data.unpack_real())  # lj14_c6A
                parameters.append(data.unpack_real())  # lj14_c12A
                parameters.append(data.unpack_real())  # lj14_c6B
                parameters.append(data.unpack_real())  # lj14_c12B
            elif i in [tpr_setting.F_LJC14_Q]:
                parameters.append(data.unpack_real())  # ljc14.fqq
                parameters.append(data.unpack_real())  # ljc14.qi
                parameters.append(data.unpack_real())  # ljc14.qj
                parameters.append(data.unpack_real())  # ljc14.c6
                parameters.append(data.unpack_real())  # ljc14.c12
            elif i in [tpr_setting.F_LJC_PAIRS_NB]:
                parameters.append(data.unpack_real())  # ljcnb.qi
                parameters.append(data.unpack_real())  # ljcnb.qj
                parameters.append(data.unpack_real())  # ljcnb.c6
                parameters.append(data.unpack_real())  # ljcnb.c12

            elif i in [
                tpr_setting.F_PIDIHS, tpr_setting.F_ANGRES,
                tpr_setting.F_ANGRESZ, tpr_setting.F_PDIHS,
            ]:
                parameters.append(data.unpack_real())  # pdihs_phiA
                parameters.append(data.unpack_real())  # pdihs_cpA
                parameters.append(data.unpack_real())  # pdihs_phiB
                parameters.append(data.unpack_real())  # pdihs_cpB
                parameters.append(data.unpack_int())  # pdihs_mult

            elif i in [tpr_setting.F_RESTRDIHS]:
                parameters.append(data.unpack_real())  # pdihs.phiA
                parameters.append(data.unpack_real())  # pdihs.cpA
            elif i in [tpr_setting.F_DISRES]:
                parameters.append(data.unpack_int())  # disres.label
                parameters.append(data.unpack_int())  # disres.type
                parameters.append(data.unpack_real())  # disres.low
                parameters.append(data.unpack_real())  # disres.up1
                parameters.append(data.unpack_real())  # disres.up2
                parameters.append(data.unpack_real())  # disres.kfac

            elif i in [tpr_setting.F_ORIRES]:
                parameters.append(data.unpack_int())  # orires.ex
                parameters.append(data.unpack_int())  # orires.label
                parameters.append(data.unpack_int())  # orires.power
                parameters.append(data.unpack_real())  # orires.c
                parameters.append(data.unpack_real())  # orires.obs
                parameters.append(data.unpack_real())  # orires.kfac

            elif i in [tpr_setting.F_DIHRES]:
                if header.fver < 72:
                    parameters.append(data.unpack_int())  # idum
                    parameters.append(data.unpack_int())  # idum
                parameters.append(data.unpack_real())  # dihres.phiA
                parameters.append(data.unpack_real())  # dihres.dphiA
                parameters.append(data.unpack_real())  # dihres.kfacA
                if header.fver >= 72:
                    parameters.append(data.unpack_real())  # dihres.phiB
                    parameters.append(data.unpack_real())  # dihres.dphiB
                    parameters.append(data.unpack_real())  # dihres.kfacB

            elif i in [tpr_setting.F_POSRES]:
                parameters.append(tpr_utils.do_rvec(data))  # posres.pos0A
                parameters.append(tpr_utils.do_rvec(data))  # posres.fcA
                parameters.append(tpr_utils.do_rvec(data))  # posres.pos0B
                parameters.append(tpr_utils.do_rvec(data))  # posres.fcB

            elif i in [tpr_setting.F_FBPOSRES]:
                parameters.append(data.unpack_int())   # fbposres.geom
                parameters.append(tpr_utils.do_rvec(data))       # fbposres.pos0
                parameters.append(data.unpack_real())  # fbposres.r
                parameters.append(data.unpack_real())  # fbposres.k

            elif i in [tpr_setting.F_CBTDIHS]:
                parameters.append(tpr_utils.ndo_real(data, tpr_setting.NR_CBTDIHS))  # cbtdihs.cbtcA

            elif i in [tpr_setting.F_RBDIHS]:
                parameters.append(tpr_utils.ndo_real(data, tpr_setting.NR_RBDIHS))  # iparams_rbdihs_rbcA
                parameters.append(tpr_utils.ndo_real(data, tpr_setting.NR_RBDIHS))  # iparams_rbdihs_rbcB

            elif i in [tpr_setting.F_FOURDIHS]:
                # Fourier dihedrals
                parameters.append(tpr_utils.ndo_real(data, tpr_setting.NR_RBDIHS))  # rbdihs.rbcA
                parameters.append(tpr_utils.ndo_real(data, tpr_setting.NR_RBDIHS))  # rbdihs.rbcB

            elif i in [tpr_setting.F_CONSTR, tpr_setting.F_CONSTRNC]:
                parameters.append(data.unpack_real())  # dA
                parameters.append(data.unpack_real())  # dB

            elif i in [tpr_setting.F_SETTLE]:
                parameters.append(data.unpack_real())  # settle.doh
                parameters.append(data.unpack_real())  # settle.dhh

            elif i in [tpr_setting.F_VSITE1]:
                pass

            elif i in [tpr_setting.F_VSITE2, tpr_setting.F_VSITE2FD]:
                parameters.append(data.unpack_real())  # vsite.a

            elif i in [tpr_setting.F_VSITE3, tpr_setting.F_VSITE3FD, tpr_setting.F_VSITE3FAD]:
                parameters.append(data.unpack_real())  # vsite.a

            elif i in [tpr_setting.F_VSITE3OUT, tpr_setting.F_VSITE4FD, tpr_setting.F_VSITE4FDN]:
                parameters.append(data.unpack_real())  # vsite.a
                parameters.append(data.unpack_real())  # vsite.b
                parameters.append(data.unpack_real())  # vsite.c

            elif i in [tpr_setting.F_VSITEN]:
                parameters.append(data.unpack_int())  # vsiten.n
                parameters.append(data.unpack_real())  # vsiten.a

            elif i in [tpr_setting.F_GB12, tpr_setting.F_GB13, tpr_setting.F_GB14]:
                # /* We got rid of some parameters in version 68 */
                if header.fver < 68:
                    parameters.append(data.unpack_real())  # rdum
                    parameters.append(data.unpack_real())  # rdum
                    parameters.append(data.unpack_real())  # rdum
                    parameters.append(data.unpack_real())  # rdum
                parameters.append(data.unpack_real())  # gb.sar
                parameters.append(data.unpack_real())  # gb.st
                parameters.append(data.unpack_real())  # gb.pi
                parameters.append(data.unpack_real())  # gb.gbr
                parameters.append(data.unpack_real())  # gb.bmlt

            elif i in [tpr_setting.F_CMAP]:
                parameters.append(data.unpack_int())  # cmap.cmapA
                parameters.append(data.unpack_int())  # cmap.cmapB
            else:
                raise NotImplementedError(f"unknown functype: {i}")
            interactions.append(dict(
                type=tpr_setting.interaction_types[i][1], parameters=parameters))

        return interactions


class GromacsParser:
    def __init__(self):
        self.log_parser = GromacsLogParser()
        self.traj_parser = GromacsMDAnalysisParser()
        self.energy_parser = GromacsEDRParser()
        self._metainfo_mapping = {
            'LJ (SR)': 'Leonard-Jones', 'Coulomb (SR)': 'coulomb',
            'Potential': 'potential', 'Kinetic En.': 'kinetic'}
        self._frame_rate = None
        # max cumulative number of atoms for all parsed trajectories to calculate sampling rate
        self._cum_max_atoms = 1000000

    @property
    def frame_rate(self):
        if self._frame_rate is None:
            n_atoms = self.traj_parser.get('n_atoms', 0)
            n_frames = self.traj_parser.get('n_frames', 0)
            if n_atoms == 0 or n_frames == 0:
                self._frame_rate = 1
            else:
                cum_atoms = n_atoms * n_frames
                self._frame_rate = 1 if cum_atoms <= self._cum_max_atoms else cum_atoms // self._cum_max_atoms
        return self._frame_rate

    def get_gromacs_file(self, ext):
        files = [d for d in self._gromacs_files if d.endswith(ext)]

        if len(files) == 1:
            return os.path.join(self._maindir, files[0])

        # we assume that the file has the same basename as the log file e.g.
        # out.log would correspond to out.tpr and out.trr and out.edr
        for f in files:
            if f.rsplit('.', 1)[0] == self._basename:
                return os.path.join(self._maindir, f)

        for f in files:
            if f.rsplit('.', 1)[0].startswith(self._basename):
                return os.path.join(self._maindir, f)

        # if the files are all named differently, we guess that the one that does not
        # share the same basename would be file we are interested in
        # e.g. in a list of files out.log someout.log out.tpr out.trr another.tpr file.trr
        # we guess that the out.* files belong together and the rest that does not share
        # a basename would be grouped together
        counts = []
        for f in files:
            count = 0
            for reff in self._gromacs_files:
                if f.rsplit('.', 1)[0] == reff.rsplit('.', 1)[0]:
                    count += 1
            if count == 1:
                return os.path.join(self._maindir, f)
            counts.append(count)

        return os.path.join(self._maindir, files[counts.index(min(counts))])

    def parse_thermodynamic_data(self):
        sec_run = self.archive.run[-1]

        # forces = self.traj_parser.get('forces')
        n_frames = self.traj_parser.get('n_frames')
        for n in range(n_frames):
            if (n % self.frame_rate) > 0:
                continue
            sec_scc = sec_run.m_create(Calculation)
            sec_scc.forces = Forces(total=ForcesEntry(value=self.traj_parser.get_forces(n)))
            sec_scc.system_ref = sec_run.system[n // self.frame_rate]
            sec_scc.method_ref = sec_run.method[-1]

        # get it from edr file
        if self.energy_parser.keys():
            thermo_data = self.energy_parser
            n_evaluations = self.energy_parser.length
        else:
            # try to get it from log file
            steps = self.log_parser.get('step', [])
            thermo_data = dict()
            for n, step in enumerate(steps):
                n = int(step.get('step_info', {}).get('Step', n))
                if step.energies is None:
                    continue
                keys = step.energies.keys()
                for key in keys:
                    thermo_data.setdefault(key, [None] * n_frames)
                    thermo_data[key][n] = step.energies.get(key)
                info = step.get('step_info', {})
                thermo_data.setdefault('Time', [None] * n_frames)
                thermo_data['Time'][n] = info.get('Time', None)
            n_evaluations = n + 1

        if not thermo_data:
            # get it from edr file
            thermo_data = self.energy_parser
            n_evaluations = self.energy_parser.length

        create_scc = False
        if n_frames != n_evaluations:
            self.logger.warn(
                'Mismatch in number of calculations and number of thermodynamic '
                'evaluations, will create new sections')
            create_scc = True

        timestep = self.traj_parser.get('timestep')
        if timestep is None:
            timestep = self.log_parser.get('input_parameters', {}).get('dt', 1.0) * ureg.ps

        # TODO add other energy contributions, properties
        energy_keys = ['LJ (SR)', 'Coulomb (SR)', 'Potential', 'Kinetic En.']

        for n in range(n_evaluations):
            if (n % self.frame_rate) > 0:
                continue

            if create_scc:
                sec_scc = sec_run.m_create(Calculation)
            else:
                sec_scc = sec_run.calculation[n // self._frame_rate]
            sec_thermo = sec_scc.m_create(Thermodynamics)
            sec_energy = sec_scc.m_create(Energy)
            for key in thermo_data.keys():
                val = thermo_data.get(key)[n]
                if val is None:
                    continue

                if key == 'Total Energy':
                    sec_energy.total = EnergyEntry(value=val)
                elif key == 'Potential':
                    sec_thermo.potential_energy = val
                elif key == 'Kinetic En.':
                    sec_thermo.kinetic_energy = val
                elif key == 'Coulomb (SR)':
                    sec_energy.coulomb = EnergyEntry(value=val)
                elif key == 'Pressure':
                    sec_thermo.pressure = val
                elif key == 'Temperature':
                    sec_thermo.temperature = val
                elif key == 'Time':
                    sec_thermo.time_step = int((val / timestep).magnitude)
                if key in energy_keys:
                    sec_energy.contributions.append(
                        EnergyEntry(kind=self._metainfo_mapping[key], value=val))

    def parse_system(self):
        sec_run = self.archive.run[-1]

        n_frames = self.traj_parser.get('n_frames', 0)

        def get_composition(children_names):
            children_count_tup = np.unique(children_names, return_counts=True)
            formula = ''.join([f'{name}({count})' for name, count in zip(*children_count_tup)])
            return formula

        pbc = self.log_parser.get_pbc()
        for n in range(n_frames):
            if (n % self.frame_rate) > 0:
                continue
            positions = self.traj_parser.get_positions(n)
            sec_system = sec_run.m_create(System)
            if positions is None:
                continue

            sec_atoms = sec_system.m_create(Atoms)
            sec_atoms.n_atoms = self.traj_parser.get_n_atoms(n)
            sec_atoms.periodic = pbc
            sec_atoms.lattice_vectors = self.traj_parser.get_lattice_vectors(n)
            sec_atoms.labels = self.traj_parser.get_atom_labels(n)
            sec_atoms.positions = positions

            velocities = self.traj_parser.get_velocities(n)
            if velocities is not None:
                sec_atoms.velocities = velocities

        # parse atomsgroup (segments --> molecules --> residues)
        atoms_info = self.traj_parser._results['atom_info']
        atoms_moltypes = np.array(atoms_info['moltypes'])
        atoms_molnums = np.array(atoms_info['molnums'])
        atoms_resids = np.array(atoms_info['resids'])
        atoms_elements = np.array(atoms_info['elements'])
        atoms_resnames = np.array(atoms_info['resnames'])
        for segment in self.traj_parser.universe.segments:
            # we only create atomsgroup in the initial system
            sec_segment = sec_run.system[0].m_create(AtomsGroup)
            sec_segment.label = segment.segid
            sec_segment.type = 'molecule_group'
            sec_segment.index = int(segment.segindex)
            sec_segment.atom_indices = segment.atoms.ix
            sec_segment.n_atoms = len(sec_segment.atom_indices)
            sec_segment.is_molecule = False

            moltypes = np.unique(atoms_moltypes[sec_segment.atom_indices])
            moltypes_count = {}
            for moltype in moltypes:
                atom_indices = np.where(atoms_moltypes == moltype)[0]
                # mol_nums is the molecule identifier for each atom
                mol_nums = atoms_molnums[atom_indices]
                moltypes_count[moltype] = np.unique(mol_nums).shape[0]
            formula = ''.join([f'{moltype}({moltypes_count[moltype]})' for moltype in moltypes_count])
            sec_segment.composition_formula = formula

            for i_molecule, molecule in enumerate(np.unique(atoms_molnums[sec_segment.atom_indices])):
                sec_molecule = sec_segment.m_create(AtomsGroup)
                sec_molecule.index = i_molecule
                sec_molecule.atom_indices = np.where(atoms_molnums == molecule)[0]
                sec_molecule.n_atoms = len(sec_molecule.atom_indices)
                # use first particle to get the moltype
                # not sure why but this value is being cast to int, cast back to str
                sec_molecule.label = str(atoms_moltypes[sec_molecule.atom_indices[0]])
                sec_molecule.type = 'molecule'
                sec_molecule.is_molecule = True

                mol_resids = np.unique(atoms_resids[sec_molecule.atom_indices])
                n_res = mol_resids.shape[0]
                if n_res == 1:
                    elements = atoms_elements[sec_molecule.atom_indices]
                    sec_molecule.composition_formula = get_composition(elements)
                else:
                    for i_res, res_id in enumerate(mol_resids):
                        sec_residue = sec_molecule.m_create(AtomsGroup)
                        sec_residue.index = i_res
                        atom_indices = np.where(atoms_resids == res_id)[0]
                        sec_residue.atom_indices = np.intersect1d(atom_indices, sec_molecule.atom_indices)
                        sec_residue.n_atoms = len(sec_residue.atom_indices)
                        res_names = atoms_resnames[sec_residue.atom_indices]
                        sec_residue.label = str(res_names[0])
                        sec_residue.type = 'monomer'
                        sec_residue.is_molecule = False
                        elements = atoms_elements[sec_residue.atom_indices]
                        sec_residue.composition_formula = get_composition(elements)

                    names = atoms_resnames[sec_molecule.atom_indices]
                    ids = atoms_resids[sec_molecule.atom_indices]
                    # filter for the first instance of each residue, as to not overcount
                    __, ids_count = np.unique(ids, return_counts=True)
                    # get the index of the first atom of each residue
                    ids_firstatom = np.cumsum(ids_count)[:-1]
                    # add the 0th index manually
                    ids_firstatom = np.insert(ids_firstatom, 0, 0)
                    names_firstatom = names[ids_firstatom]
                    sec_molecule.composition_formula = get_composition(names_firstatom)

        # calculate molecular radial distribution functions
        sec_molecular_dynamics = self.archive.workflow[-1].molecular_dynamics
        try:
            rdf_results = self.traj_parser.calc_molecular_rdf()
        except Exception:
            self.logger.error('Error calculating molecular RDF.')
            rdf_results = None
        if rdf_results is not None:
            sec_rdfs = sec_molecular_dynamics.m_create(RadialDistributionFunction)
            sec_rdfs.type = 'Molecular'
            sec_rdfs.n_smooth = rdf_results.get('n_smooth')
            sec_rdfs.n_variables = 1
            sec_rdfs.variables_name = np.array(['distance'])
            for i_pair, pair_type in enumerate(rdf_results.get('types')):
                sec_rdf_values = sec_rdfs.m_create(RadialDistributionFunctionValues)
                sec_rdf_values.label = str(pair_type)
                sec_rdf_values.n_bins = len(rdf_results['bins'][i_pair]) if rdf_results.get(
                    'bins') is not None else []
                sec_rdf_values.bins = rdf_results['bins'][i_pair] if rdf_results.get(
                    'bins') is not None else []
                sec_rdf_values.value = rdf_results['value'][i_pair] if rdf_results.get(
                    'value') is not None else []

        # calculate the molecular mean squared displacements
        try:
            msd_results = self.traj_parser.calc_molecular_mean_squard_displacements()
        except Exception:
            self.logger.error('Error calculating molecular MSD.')
            msd_results = None
        if msd_results is not None:
            sec_msds = sec_molecular_dynamics.m_create(MeanSquaredDisplacement)
            sec_msds.type = 'Molecular'
            for i_type, moltype in enumerate(msd_results.get('types')):
                sec_msd_values = sec_msds.m_create(MeanSquaredDisplacementValues)
                sec_msd_values.label = str(moltype)
                sec_msd_values.n_times = len(msd_results['times'][i_type]) if msd_results.get(
                    'times') is not None else []
                sec_msd_values.times = msd_results['times'][i_type] if msd_results.get(
                    'times') is not None else []
                sec_msd_values.value = msd_results['value'][i_type] if msd_results.get(
                    'value') is not None else []
                sec_diffusion = sec_msd_values.m_create(DiffusionConstantValues)
                sec_diffusion.value = msd_results['diffusion_constant'][i_type] if msd_results.get(
                    'diffusion_constant') is not None else []
                sec_diffusion.error_type = 'Pearson correlation coefficient'
                sec_diffusion.errors = msd_results['error_diffusion_constant'][i_type] if msd_results.get(
                    'error_diffusion_constant') is not None else []

    def parse_method(self):
        sec_method = self.archive.run[-1].m_create(Method)
        sec_force_field = sec_method.m_create(ForceField)
        sec_model = sec_force_field.m_create(Model)
        try:
            n_atoms = self.traj_parser.get('n_atoms')
        except Exception:
            gro_file = self.get_gromacs_file('gro')
            self.traj_parser.mainfile = gro_file
            n_atoms = self.traj_parser.get('n_atoms', 0)

        atom_info = self.traj_parser.get('atom_info', {})
        for n in range(n_atoms):
            sec_atom = sec_method.m_create(AtomParameters)
            sec_atom.charge = atom_info.get('charges', [None] * (n + 1))[n]
            sec_atom.mass = atom_info.get('masses', [None] * (n + 1))[n]
            sec_atom.label = atom_info.get('names', [None] * (n + 1))[n]
            sec_atom.x_gromacs_atom_name = atom_info.get('atom_names', [None] * (n + 1))[n]
            sec_atom.x_gromacs_atom_resid = atom_info.get('resids', [None] * (n + 1))[n]
            sec_atom.x_gromacs_atom_resname = atom_info.get('resnames', [None] * (n + 1))[n]
            sec_atom.x_gromacs_atom_molnum = atom_info.get('molnums', [None] * (n + 1))[n]
            sec_atom.x_gromacs_atom_moltype = atom_info.get('moltypes', [None] * (n + 1))[n]

        if n_atoms == 0:
            self.logger.error('Error parsing interactions.')

        interactions = self.traj_parser.get_interactions()
        for interaction in interactions:
            sec_interaction = sec_model.m_create(Interaction)
            for key, val in interaction.items():
                setattr(sec_interaction, key, val)

    def parse_workflow(self):
        sec_workflow = self.archive.m_create(Workflow)

        sampling_settings = self.log_parser.get_sampling_settings()

        sec_workflow.type = sampling_settings['sampling_method']
        sec_md = sec_workflow.m_create(MolecularDynamics)
        sec_md.ensemble_type = sampling_settings['ensemble_type']
        sec_md.x_gromacs_integrator_type = sampling_settings['integrator_type']

        input_parameters = self.log_parser.get('input_parameters', {})
        timestep = input_parameters.get('dt', 0)
        sec_md.x_gromacs_integrator_dt = timestep
        sec_md.timestep = timestep * 1e-12

        nsteps = input_parameters.get('nsteps', 0)
        sec_md.x_gromacs_number_of_steps_requested = nsteps

        tp_settings = self.log_parser.get_tpstat_settings()

        target_T = tp_settings.get('target_T', None)
        if target_T is not None:
            sec_md.x_gromacs_thermostat_target_temperature = target_T
        thermostat_tau = tp_settings.get('thermostat_tau', None)
        if thermostat_tau is not None:
            sec_md.x_gromacs_thermostat_tau = thermostat_tau
        target_P = tp_settings.get('target_P', None)
        if target_P is not None:
            sec_md.x_gromacs_barostat_target_pressure = target_P
        barostat_tau = tp_settings.get('barostat_P', None)
        if barostat_tau is not None:
            sec_md.x_gromacs_barostat_tau = barostat_tau
        langevin_gamma = tp_settings.get('langevin_gamma', None)
        if langevin_gamma is not None:
            sec_md.x_gromacs_langevin_gamma = langevin_gamma

    def parse_input(self):
        sec_run = self.archive.run[-1]
        sec_input_output_files = sec_run.m_create(x_gromacs_section_input_output_files)

        topology_file = os.path.basename(self.traj_parser.mainfile)
        if topology_file.endswith('tpr'):
            sec_input_output_files.x_gromacs_inout_file_topoltpr = topology_file
        elif topology_file.endswith('gro'):
            sec_input_output_files.x_gromacs_inout_file_confoutgro = topology_file

        trajectory_file = os.path.basename(self.traj_parser.auxilliary_files[0])
        sec_input_output_files.x_gromacs_inout_file_trajtrr = trajectory_file

        edr_file = os.path.basename(self.energy_parser.mainfile)
        sec_input_output_files.x_gromacs_inout_file_eneredr = edr_file

        sec_control_parameters = sec_run.m_create(x_gromacs_section_control_parameters)
        input_parameters = self.log_parser.get('input_parameters', {})
        input_parameters.update(self.log_parser.get('header', {}))
        for key, val in input_parameters.items():
            key = 'x_gromacs_inout_control_%s' % key.replace('-', '').replace(' ', '_').lower()
            if hasattr(sec_control_parameters, key):
                val = str(val) if not isinstance(val, np.ndarray) else val
                setattr(sec_control_parameters, key, val)

    def init_parser(self):
        self.log_parser.mainfile = self.filepath
        self.log_parser.logger = self.logger
        self.traj_parser.logger = self.logger
        self.energy_parser.logger = self.logger
        self._frame_rate = None

    def reuse_parser(self, parser):
        self.log_parser.quantities = parser.log_parser.quantities

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self._maindir = os.path.dirname(self.filepath)
        self._gromacs_files = os.listdir(self._maindir)
        self._basename = os.path.basename(filepath).rsplit('.', 1)[0]

        self.init_parser()

        sec_run = self.archive.m_create(Run)

        header = self.log_parser.get('header', {})
        sec_run.program = Program(
            name='GROMACS', version=header.get('version', 'unknown').lstrip('VERSION '))

        sec_time_run = sec_run.m_create(TimeRun)
        for key in ['start', 'end']:
            time = self.log_parser.get('time_%s' % key)
            if time is None:
                continue
            setattr(sec_time_run, 'date_%s' % key, datetime.datetime.strptime(
                time, '%a %b %d %H:%M:%S %Y').timestamp())

        host_info = self.log_parser.get('host_info')
        if host_info is not None:
            sec_run.x_gromacs_program_execution_host = host_info[0]
            sec_run.x_gromacs_parallel_task_nr = host_info[1]
            sec_run.x_gromacs_number_of_tasks = host_info[2]

        topology_file = self.get_gromacs_file('tpr')
        # I have no idea if output trajectory file can be specified in input
        trajectory_file = self.get_gromacs_file('trr')
        self.traj_parser.mainfile = topology_file
        self.traj_parser.auxilliary_files = [trajectory_file]

        self.parse_method()

        self.parse_workflow()

        self.parse_system()

        # TODO read also from ene
        edr_file = self.get_gromacs_file('edr')
        self.energy_parser.mainfile = edr_file

        self.parse_thermodynamic_data()

        self.parse_input()
