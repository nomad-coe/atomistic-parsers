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
from nomad.config.models.plugins import ParserEntryPoint


class EntryPoint(ParserEntryPoint):
    def load(self):
        from nomad.parsing import MatchingParserInterface
        from . import (
            amber,
            asap,
            bopfox,
            dftbplus,
            dlpoly,
            gromacs,
            gromos,
            gulp,
            h5md,
            lammps,
            libatoms,
            namd,
            tinker,
            xtb,
            utils,
        )

        return MatchingParserInterface(self.parser_class_name)


amber_parser_entry_point = EntryPoint(
    name='parsers/amber',
    description='NOMAD parser for AMBER.',
    parser_class_name='atomisticparsers.amber.AmberParser',
)

asap_parser_entry_point = EntryPoint(
    name='parsers/asap',
    description='NOMAD parser for ASAP.',
    parser_class_name='atomisticparsers.asap.AsapParser',
)

bopfox_parser_entry_point = EntryPoint(
    name='parsers/bopfox',
    description='NOMAD parser for BOPFOX.',
    parser_class_name='atomisticparsers.bopfox.BOPfoxParser',
)

dftbplus_parser_entry_point = EntryPoint(
    name='parsers/dftbplus',
    description='NOMAD parser for DFTBPLUS.',
    parser_class_name='atomisticparsers.dftbplus.DFTBPlusParser',
)

dlpoly_parser_entry_point = EntryPoint(
    name='parsers/dlpoly',
    description='NOMAD parser for DLPOLY.',
    parser_class_name='atomisticparsers.dlpoly.DLPolyParser',
)

gromacs_parser_entry_point = EntryPoint(
    name='parsers/gromacs',
    description='NOMAD parser for GROMACS.',
    parser_class_name='atomisticparsers.gromacs.GromacsParser',
)

gromos_parser_entry_point = EntryPoint(
    name='parsers/gromos',
    description='NOMAD parser for GROMOS.',
    parser_class_name='atomisticparsers.gromos.GromosParser',
)

gulp_parser_entry_point = EntryPoint(
    name='parsers/gulp',
    description='NOMAD parser for GULP.',
    parser_class_name='atomisticparsers.gulp.GulpParser',
)

h5md_parser_entry_point = EntryPoint(
    name='parsers/h5md',
    description='NOMAD parser for H5MD.',
    parser_class_name='atomisticparsers.h5md.H5MDParser',
)

lammps_parser_entry_point = EntryPoint(
    name='parsers/lammps',
    description='NOMAD parser for LAMMPS.',
    parser_class_name='atomisticparsers.lammps.LammpsParser',
)

libatoms_parser_entry_point = EntryPoint(
    name='parsers/libatoms',
    description='NOMAD parser for LIBATOMS.',
    parser_class_name='atomisticparsers.libatoms.LibAtomsParser',
)

namd_parser_entry_point = EntryPoint(
    name='parsers/namd',
    description='NOMAD parser for NAMD.',
    parser_class_name='atomisticparsers.namd.NAMDParser',
)

tinker_parser_entry_point = EntryPoint(
    name='parsers/tinker',
    description='NOMAD parser for TINKER.',
    parser_class_name='atomisticparsers.tinker.TinkerParser',
)

xtb_parser_entry_point = EntryPoint(
    name='parsers/xtb',
    description='NOMAD parser for XTB.',
    parser_class_name='atomisticparsers.xtb.XTBParser',
)
