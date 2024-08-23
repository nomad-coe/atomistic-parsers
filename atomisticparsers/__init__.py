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
from pydantic import Field

from nomad.config.models.plugins import ParserEntryPoint


class EntryPoint(ParserEntryPoint):
    parser_class_name: str = Field(
        description="""
        The fully qualified name of the Python class that implements the parser.
        This class must have a function `def parse(self, mainfile, archive, logger)`.
    """
    )

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
    python_package='atomisticparsers.amber',
    mainfile_contents_re=r'\s*Amber\s[0-9]+\s[A-Z]+\s*[0-9]+',
    parser_class_name='atomisticparsers.amber.AmberParser',
)

asap_parser_entry_point = EntryPoint(
    name='parsers/asap',
    description='NOMAD parser for ASAP.',
    python_package='atomisticparsers.asap',
    mainfile_binary_header_re=b'AFFormatASE\\-Trajectory',
    mainfile_mime_re='application/octet-stream',
    mainfile_name_re=r'.*.traj$',
    parser_class_name='atomisticparsers.asap.AsapParser',
)

bopfox_parser_entry_point = EntryPoint(
    name='parsers/bopfox',
    description='NOMAD parser for BOPFOX.',
    python_package='atomisticparsers.bopfox',
    mainfile_contents_re=r'\-+\s+BOPfox \(v',
    parser_class_name='atomisticparsers.bopfox.BOPfoxParser',
)

dftbplus_parser_entry_point = EntryPoint(
    name='parsers/dftbplus',
    description='NOMAD parser for DFTBPLUS.',
    python_package='atomisticparsers.dftbplus',
    mainfile_contents_re=r'\|  DFTB\+',
    mainfile_mime_re='text/.*',
    parser_class_name='atomisticparsers.dftbplus.DFTBPlusParser',
)

dlpoly_parser_entry_point = EntryPoint(
    name='parsers/dlpoly',
    description='NOMAD parser for DLPOLY.',
    python_package='atomisticparsers.dlpoly',
    mainfile_contents_re=r'\*\*\s+DL_POLY.+\*\*',
    parser_class_name='atomisticparsers.dlpoly.DLPolyParser',
)

gromacs_parser_entry_point = EntryPoint(
    name='parsers/gromacs',
    description='NOMAD parser for GROMACS.',
    python_package='atomisticparsers.gromacs',
    mainfile_contents_re=r'gmx mdrun, (VERSION|version)[\s\S]*Input Parameters:',
    parser_class_name='atomisticparsers.gromacs.GromacsParser',
)

gromos_parser_entry_point = EntryPoint(
    name='parsers/gromos',
    description='NOMAD parser for GROMOS.',
    python_package='atomisticparsers.gromos',
    mainfile_contents_re=r'Bugreports to http://www.gromos.net',
    parser_class_name='atomisticparsers.gromos.GromosParser',
)

gulp_parser_entry_point = EntryPoint(
    name='parsers/gulp',
    description='NOMAD parser for GULP.',
    python_package='atomisticparsers.gulp',
    mainfile_contents_re=(r'\s*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\s*\s*\*\s*GENERAL UTILITY '
        r'LATTICE PROGRAM\s*\*\s*'),
    parser_class_name='atomisticparsers.gulp.GulpParser',
)

h5md_parser_entry_point = EntryPoint(
    name='parsers/h5md',
    description='NOMAD parser for H5MD.',
    python_package='atomisticparsers.h5md',
    mainfile_binary_header_re=b'^\\x89HDF',
    mainfile_contents_dict={'__has_all_keys': ['h5md']},
    mainfile_mime_re='(application/x-hdf)',
    mainfile_name_re=r'^.*\.(h5|hdf5)$',
    parser_class_name='atomisticparsers.h5md.H5MDParser',
)

lammps_parser_entry_point = EntryPoint(
    name='parsers/lammps',
    description='NOMAD parser for LAMMPS.',
    python_package='atomisticparsers.lammps',
    mainfile_contents_re=r'^LAMMPS\s+\(.+\)',
    parser_class_name='atomisticparsers.lammps.LammpsParser',
)

libatoms_parser_entry_point = EntryPoint(
    name='parsers/libatoms',
    description='NOMAD parser for LIBATOMS.',
    python_package='atomisticparsers.libatoms',
    mainfile_contents_re=r'\s*<GAP_params\s',
    parser_class_name='atomisticparsers.libatoms.LibAtomsParser',
)

namd_parser_entry_point = EntryPoint(
    name='parsers/namd',
    description='NOMAD parser for NAMD.',
    python_package='atomisticparsers.namd',
    mainfile_contents_re=r'\s*Info:\s*NAMD\s*[0-9.]+\s*for\s*',
    mainfile_mime_re='text/.*',
    parser_class_name='atomisticparsers.namd.NAMDParser',
)

tinker_parser_entry_point = EntryPoint(
    name='parsers/tinker',
    description='NOMAD parser for TINKER.',
    python_package='atomisticparsers.tinker',
    mainfile_contents_re=r'TINKER  ---  Software Tools for Molecular Design',
    parser_class_name='atomisticparsers.tinker.TinkerParser',
)

xtb_parser_entry_point = EntryPoint(
    name='parsers/xtb',
    description='NOMAD parser for XTB.',
    python_package='atomisticparsers.xtb',
    mainfile_contents_re=r'x T B\s+\|\s+\|\s+=',
    parser_class_name='atomisticparsers.xtb.XTBParser',
)
