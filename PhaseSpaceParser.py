#####################################################################################
#   Project: IAEA Phase Space Analysis
#   Class: PhaseSpaceParser
#   Description: Reads IAEA phase space files. The header is parsed for the record
#   layout, byte order and history/particle counts, then each fixed length particle
#   record in the phase space file is unpacked into a Particle and added to a
#   ParticleAnalyzer.
#   Author: Campbell Haasch
#   Institution: University of Wisconsin-Madison; Department of Medical Physics
#   Lab: RED Lab/UWMRRC
#   Last Updated: 2026-10-26
#.  *Assisted by Claude Opus 5.5
#####################################################################################

from ParticleAnalyzer import Particle, ParticleAnalyzer
import struct
import math
#order of the variables flagged in $RECORD_CONTENTS (and given in $RECORD_CONSTANT when not stored)
RECORD_VARIABLES = ['x', 'y', 'z', 'u', 'v', 'w', 'weight']
#IAEA byte order codes mapped to struct/numpy endianness characters
BYTE_ORDERS = {'1234': '<', '4321': '>'}

class PhaseSpaceParser():
    """
    Parser for IAEA format phase space files (.IAEAheader + .IAEAphsp).

    Functions:
     - read_phase_space(header_filename: str, phase_space_filename: str, particle_analyzer: ParticleAnalyzer):
       Parses the header, then reads every particle record into the particle analyzer.
     - parse_phase_space_header(header_filename: str):
       Reads the record contents, record constants, record length, byte order, original histories
       and particle count into self.header_info and builds the record layout used to unpack particles.
    **Note: Particle weight is initially stored in MeV but converted to keV**
    """

    def __init__(self, filetype:str):
        if filetype in ['IAEA']:
            self.filetype = filetype
        else:
            raise ValueError("Wrong filetype entered for phase space parser")
        self.header_info={}

    """Parse the header, then read each fixed length particle record into the particle analyzer.
       The original histories from the header are passed to the analyzer for the efficiency analysis"""
    def read_phase_space(self, header_filename, phase_space_filename, particle_analyzer):
        self.parse_phase_space_header(header_filename)
        particle_analyzer.orig_histories = self.header_info['orig_histories']

        with open(phase_space_filename, "rb") as file:
            while chunk := file.read(self.header_info["record_length"]):
                particle_analyzer.add_particle(self._parse_particle(chunk))


    """Unpack one record into a Particle: particle type, energy, stored variables, extra floats, extra longs.
       Variables that are not stored are filled from the header's record constants"""
    def _parse_particle(self, byte_chunk):
        if len(byte_chunk) != self.record_struct.size:
            raise ValueError(f"Particle record is {len(byte_chunk)} bytes, expected {self.record_struct.size}")
        values = self.record_struct.unpack(byte_chunk)
        particle_type, energy = values[0], values[1]
        n_stored = len(self.stored_variables)
        n_extra_floats = self.header_info['record_contents']['n_extra_floats']
        variables = {**self.header_info['record_constants'], **dict(zip(self.stored_variables, values[2:2 + n_stored]))}
        extra_floats = list(values[2 + n_stored:2 + n_stored + n_extra_floats])
        extra_longs = list(values[2 + n_stored + n_extra_floats:])

        #a negative energy marks the first particle of a new history, the sign of the
        #particle type carries the sign of w, which is otherwise derived from u and v.
        #IAEA energies are in MeV, particles hold energy in keV
        new_history = energy < 0
        if 'w' in variables:
            w = variables['w']
        else:
            w = math.copysign(math.sqrt(max(0.0, 1 - variables['u']**2 - variables['v']**2)), particle_type)

        return Particle(variables['x'], variables['y'], variables['z'], variables['u'], variables['v'], variables['weight'],
                        w=w, energy=abs(energy) * 1000, particle_type=abs(particle_type), new_history=new_history,
                        extra_floats=extra_floats, extra_longs=extra_longs)

    """Split the header into {KEYWORD: [value lines]} with // comments and blank lines removed."""
    def _read_header_sections(self, header_filename):
        sections = {}
        current_key = None
        with open(header_filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.split('//')[0].strip()
                if line.startswith('$'):
                    current_key, _, value = line[1:].partition(':')
                    sections[current_key] = []
                    if value.strip():
                        sections[current_key].append(value.strip())
                elif line and current_key is not None:
                    sections[current_key].append(line)
        return sections

    """Read the record contents, record constants, record length, byte order, original histories and
       particle count into self.header_info, and build the struct layout used to unpack particle records"""
    def parse_phase_space_header(self, header_filename):
        sections = self._read_header_sections(header_filename)

        #record contents: stored flags for x, y, z, u, v, w, weight, then the number of
        #extra floats and extra longs, then a type code for each extra float/long
        contents = [int(value) for value in sections['RECORD_CONTENTS']]
        stored = dict(zip(RECORD_VARIABLES, (bool(flag) for flag in contents[:7])))
        n_extra_floats = contents[7]
        n_extra_longs = contents[8]
        extra_types = contents[9:]
        self.header_info['record_contents'] = {
            'stored': stored,
            'n_extra_floats': n_extra_floats,
            'n_extra_longs': n_extra_longs,
            'extra_float_types': extra_types[:n_extra_floats],
            'extra_long_types': extra_types[n_extra_floats:n_extra_floats + n_extra_longs],
        }

        #record constants: one value for each variable that is not stored, in RECORD_VARIABLES order
        constants = [float(value) for value in sections.get('RECORD_CONSTANT', [])]
        not_stored = [var for var in RECORD_VARIABLES if not stored[var]]
        if len(constants) != len(not_stored):
            raise ValueError(f"Expected {len(not_stored)} record constants for {not_stored}, found {len(constants)}")
        self.header_info['record_constants'] = dict(zip(not_stored, constants))

        #record length in bytes, checked against the contents: particle type (1 byte), energy (4 bytes),
        #4 bytes per stored x, y, z, u, v, weight (w is derived from u and v) and per extra float/long
        record_length = int(sections['RECORD_LENGTH'][0])
        n_stored_floats = sum(stored[var] for var in RECORD_VARIABLES if var != 'w')
        expected_length = 1 + 4 + 4 * (n_stored_floats + n_extra_floats + n_extra_longs)
        if record_length != expected_length:
            raise ValueError(f"Record length {record_length} does not match record contents ({expected_length} bytes)")
        self.header_info['record_length'] = record_length

        byte_order = sections['BYTE_ORDER'][0]
        if byte_order not in BYTE_ORDERS:
            raise ValueError(f"Unsupported byte order {byte_order}")
        self.header_info['byte_order'] = byte_order
        self.header_info['endianness'] = BYTE_ORDERS[byte_order]

        #record layout for unpacking particles: signed char type, float energy, a float per stored
        #variable (w is never stored), a float per extra float and a signed int per extra long
        self.stored_variables = [var for var in RECORD_VARIABLES if var != 'w' and stored[var]]
        self.record_struct = struct.Struct(BYTE_ORDERS[byte_order] + 'bf' + 'f' * (len(self.stored_variables) + n_extra_floats)
                                           + 'i' * n_extra_longs)

        self.header_info['orig_histories'] = int(sections['ORIG_HISTORIES'][0])
        self.header_info['particles'] = int(sections['PARTICLES'][0])

        return self.header_info
