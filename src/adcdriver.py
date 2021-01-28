from veloxchem import mpi_master
from scipy import constants
import os
import numpy as np
import re


class AdcDriver:
    """
    Implements ADC driver.

    :param comm:
        The MPI communicator.
    :param ostream:
        The output stream.

    Instance variable
        - comm: The MPI communicator.
        - rank: The MPI rank.
        - nodes: Number of MPI processes.
        - ostream: The output stream.
        - adc_tol: convergence tolerance for for the
        - adc_method: adc level of theory to be used; possible values: adc0,
          adc1, adc2, adc2-x, adc3 and corresp. cvs variants
        - adc_states: number of singlet and triplet excited states to be
          computed
        - adc_singlets: number of singlet excited states to be computed
        - adc_triplets: number of triplet excited states to be computed
        - adc_spin_filp: number of excited states; computed using the spin-flip
          variant of ADC
        - adc_core_orbitals: only valid with cvs-adc; orbitals to be considered
          part of the core space (array)
        - adc_frozen_core: occupied orbitals to be considered inactive during
          the MP and ADC calculation (array)
        - adc_frozen_virtual: virtual orbitals to be considered inactive during
          the MP and ADC calculation (array)
        - print_states: print detailed information about each excited state
        - adc_ecd: compute rotatory strengths for all excited states
          (True/False)
    """

    def __init__(self, comm, ostream):
        """
        Initializes ADC driver.
        """

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

        # default ADC settings
        self.adc_tol = 1e-6
        self.adc_method = 'adc2'
        self.adc_states = 3
        self.adc_singlets = None
        self.adc_triplets = None
        self.adc_spin_flip = None
        self.adc_core_orbitals = None
        self.adc_frozen_core = None
        self.adc_frozen_virtual = None
        self.print_states = False
        self.adc_ecd = True
        self.cpp = False
        self.frequencies = None
        self.damping = None

    def update_settings(self, adc_dict, scf_drv=None):
        """
        Updates settings in ADC driver.

        :param adc_dict:
            The dictionary of ADC settings.
        """

        if 'tol' in adc_dict:
            self.adc_tol = float(adc_dict['tol'])
        elif scf_drv is not None:
            self.adc_tol = max(scf_drv.conv_thresh / 100, 1e-6)
            if scf_drv.conv_thresh > self.adc_tol:
                error_text = os.linesep + os.linesep
                error_text += '*** SCF convergence threshold '
                error_text += '({:.1e}) '.format(scf_drv.conv_thresh)
                error_text += 'needs to be lower than ADC convergence '
                error_text += 'tolerance ({:.1e})'.format(self.adc_tol)
                error_text += os.linesep
                raise ValueError(error_text)

        if 'states' in adc_dict:
            self.adc_states = int(adc_dict['states'])

        if 'singlets' in adc_dict:
            self.adc_singlets = int(adc_dict['singlets'])
            self.adc_states = None

        if 'triplets' in adc_dict:
            self.adc_triplets = int(adc_dict['triplets'])
            self.adc_states = None
            self.adc_singlets = None

        if 'spin_flip' in adc_dict:
            self.adc_spin_flip = int(adc_dict['spin_flip'])
            self.adc_states = None
            self.adc_singlets = None
            self.adc_triplets = None

        if 'method' in adc_dict:
            if 'cpp' in adc_dict['method']:
                self.cpp = True
                self.adc_method = adc_dict['method'].split()[0]
            else:
                self.adc_method = adc_dict['method']

        if 'frequencies' in adc_dict:
            self.frequencies = adc_dict['frequencies']
            self.adc_triplets = None
            self.adc_states = None
            self.adc_singlets = None
            self.adc_spin_flip = None

        if 'damping' in adc_dict:
            self.damping = float(adc_dict['damping'])

        if 'core_orbitals' in adc_dict:
            self.adc_core_orbitals = self.parse_orbital_input(
                adc_dict['core_orbitals'])

        if 'frozen_core' in adc_dict:
            self.adc_frozen_core = self.parse_orbital_input(
                adc_dict['frozen_core'])

        if 'frozen_virtual' in adc_dict:
            self.adc_frozen_virtual = self.parse_orbital_input(
                adc_dict['frozen_virtual'])

        if 'print_states' in adc_dict:
            key = adc_dict['print_states'].lower()
            self.print_states = True if key in ['yes', 'y'] else False

        if 'ecd' in adc_dict:
            key = adc_dict['ecd'].lower()
            self.adc_ecd = False if key in ['no', 'n'] else True

    @staticmethod
    def parse_orbital_input(orbs):
        """
        Parses input orbital indices (1-based) and returns a list of orbital
        indices (0-based).

        Examples (input -> output):
            1              -> [0]
            (1, 3)         -> [0, 2]
            '1 - 5'        -> [0, 1, 2, 3, 4]
            [1, 2, 3, 4]   -> [0, 1, 2, 3]
            '1-3, 4, 5-7'  -> [0, 1, 2, 3, 4, 5, 6]

        :param orbs:
            The input orbital indices (can be integer, list, tuple, or string).
        :return:
            A list of orbital indices.
        """

        if isinstance(orbs, int):
            return [orbs - 1]
        elif isinstance(orbs, (list, tuple)):
            return [x - 1 for x in orbs]
        elif isinstance(orbs, str):
            output = []
            for x in orbs.replace(',', ' ').split():
                if '-' in x:
                    z = [int(y) for y in x.split('-')]
                    output += list(range(z[0] - 1, z[-1], 1))
                else:
                    output.append(int(x) - 1)
            return output

    @staticmethod
    def parse_frequencies(input_frequencies):
        """
        Parses input frequencies for the respondo response library.

        Input example: "0.4-0.5 (0.002), 0.7-0.9 (0.001)"

        :param input_frequencies:
            The string of input frequencies.
        :return:
            an ndarray of frequencies required by respondo
        """
        if isinstance(input_frequencies, np.ndarray):
            return input_frequencies

        frequencies = []
        for w in input_frequencies.split(','):
            if '-' in w:
                m = re.search(r'^(.*)-(.*)\((.*)\)$', w)
                if m is None:
                    m = re.search(r'^(.*)-(.*)-(.*)$', w)

                frequencies += list(
                    np.arange(
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                    ))
            elif w:
                frequencies.append(float(w))
        return np.array(frequencies)

    def compute(self, task, scf_drv, verbose=True):
        """
        Performs ADC calculation.

        :param task:
            The gator task.
        :param scf_drv:
            The converged SCF driver.
        """

        scf_drv.task = task

        if self.rank == mpi_master():
            if verbose:
                self.print_header()

            try:
                import adcc
            except ImportError:
                error_text = os.linesep + os.linesep
                error_text += '*** Unable to import adcc. ' + os.linesep
                error_text += '*** Please download and install '
                error_text += 'from https://github.com/adc-connect/adcc'
                error_text += os.linesep
                raise ImportError(error_text)

            if self.cpp:
                if self.frequencies is None:
                    error_text = os.linesep + os.linesep
                    error_text += '*** Please define a frequency range (a.u.)'
                    error_text += ' for the cpp solver.'
                    error_text += os.linesep + '*** Example:' + os.linesep
                    error_text += 'frequencies: 0.40-0.60 (0.05)'
                    error_text += os.linesep
                    raise ValueError(error_text)
                if self.damping is None:
                    error_text = os.linesep + os.linesep
                    error_text += '*** Please define a damping parameter '
                    error_text += '(a.u.) for the cpp solver.'
                    error_text += os.linesep + 'Example:' + os.linesep
                    error_text += 'damping: 0.001'
                    error_text += os.linesep
                    raise ValueError(error_text)
                try:
                    import respondo
                except ImportError:
                    error_text = os.linesep + os.linesep
                    error_text += '*** Unable to import respondo. ' + os.linesep
                    error_text += '*** Please install from conda or '
                    error_text += 'https://github.com/gator-program/respondo'
                    error_text += os.linesep
                    raise ImportError(error_text)

                adc_drv = adcc.ReferenceState(
                    scf_drv,
                    core_orbitals=self.adc_core_orbitals,
                    frozen_core=self.adc_frozen_core,
                    frozen_virtual=self.adc_frozen_virtual)

                frequencies = self.parse_frequencies(self.frequencies)
                all_pol = [
                    respondo.complex_polarizability(adc_drv,
                                                    method=self.adc_method,
                                                    omega=w,
                                                    gamma=self.damping,
                                                    conv_tol=self.adc_tol)
                    for w in frequencies
                ]
                cross_sections = (
                    respondo.polarizability.one_photon_absorption_cross_section(
                        np.array(all_pol), frequencies))

                if verbose:
                    self.print_cpp_results(frequencies, cross_sections)

                return frequencies, cross_sections

            else:

                # set number of threads in adcc
                adcc.set_n_threads(int(os.environ['OMP_NUM_THREADS']))

                adc_drv = adcc.run_adc(scf_drv,
                                       method=self.adc_method,
                                       core_orbitals=self.adc_core_orbitals,
                                       n_states=self.adc_states,
                                       n_singlets=self.adc_singlets,
                                       n_triplets=self.adc_triplets,
                                       n_spin_flip=self.adc_spin_flip,
                                       frozen_core=self.adc_frozen_core,
                                       frozen_virtual=self.adc_frozen_virtual,
                                       conv_tol=self.adc_tol)

                if verbose:
                    self.print_excited_states(adc_drv)

                if self.print_states:
                    self.print_detailed_states(adc_drv)

                if verbose:
                    self.print_convergence(adc_drv)

                return adc_drv

    def print_header(self):
        """
        Prints header for the ADC driver.
        """

        self.ostream.print_blank()
        text = 'Algebraic Diagrammatic Construction (ADC)'
        self.ostream.print_header(text)
        self.ostream.print_header('=' * (len(text) + 2))
        self.ostream.print_blank()

        str_width = 60
        cur_str = 'ADC method                   : {:s}'.format(self.adc_method)
        if self.cpp:
            cur_str += ' (cpp)'
        self.ostream.print_header(cur_str.ljust(str_width))
        if self.adc_states is not None:
            cur_str = 'Number of States             : {:d}'.format(
                self.adc_states)
            self.ostream.print_header(cur_str.ljust(str_width))
        elif self.adc_singlets is not None:
            cur_str = 'Number of Singlet States     : {:d}'.format(
                self.adc_singlets)
            self.ostream.print_header(cur_str.ljust(str_width))
        elif self.adc_triplets is not None:
            cur_str = 'Number of Triplet States     : {:d}'.format(
                self.adc_triplets)
            self.ostream.print_header(cur_str.ljust(str_width))
        elif self.adc_spin_flip is not None:
            cur_str = 'Number of States, Spin-Flip  : {:d}'.format(
                self.adc_spin_flip)
            self.ostream.print_header(cur_str.ljust(str_width))
        else:
            freqs = [f.strip() for f in self.frequencies.split(',')]
            cur_str = 'Frequencies (a.u.)           : {:s}'.format(freqs[0])
            self.ostream.print_header(cur_str.ljust(str_width))
            for f in freqs[1:]:
                cur_str = '                               {:s}'.format(f)
                self.ostream.print_header(cur_str.ljust(str_width))

        if self.damping is not None:
            cur_str = 'Damping                      : {:f} a.u.'.format(
                self.damping)
            self.ostream.print_header(cur_str.ljust(str_width))

        if self.adc_core_orbitals is not None:
            cur_str = 'CVS-ADC, Core Orbital Space  :'
            for orb in self.adc_core_orbitals:
                cur_str += ' {:d}'.format(orb + 1)
                # '+1' converts from run_adc indexing (starts at 0) back to
                # input indexing (starts at 1)
            self.ostream.print_header(cur_str.ljust(str_width))

        if self.adc_frozen_core is not None:
            cur_str = 'Frozen Core Orbital Space    :'
            for orb in self.adc_frozen_core:
                cur_str += ' {:d}'.format(orb + 1)
                # '+1' converts from run_adc indexing (starts at 0) back to
                # input indexing (starts at 1)
            self.ostream.print_header(cur_str.ljust(str_width))

        if self.adc_frozen_virtual is not None:
            cur_str = 'Frozen Virtual Orbital Space :'
            for orb in self.adc_frozen_virtual:
                cur_str += ' {:d}'.format(orb + 1)
                # '+1' converts from run_adc indexing (starts at 0) back to
                # input indexing (starts at 1)
            self.ostream.print_header(cur_str.ljust(str_width))

        cur_str = 'Convergence threshold        : {:.1e}'.format(self.adc_tol)
        self.ostream.print_header(cur_str.ljust(str_width))

        self.ostream.print_blank()
        self.ostream.flush()

    def print_convergence(self, adc_drv):
        """
        Prints finish header to output stream.
        """

        end = ' All went well!'
        if not hasattr(adc_drv, 'converged'):
            self.ostream.print_header('NOT CONVERGED')
            end = ' Did NOT converge.'

        self.ostream.print_header('End of ADC calculation.' + end)
        self.ostream.print_blank()
        self.ostream.flush()

    def print_cpp_results(self, frequencies, cross_sections):
        from scipy import constants
        eV = constants.value("Hartree energy in eV")

        text = 'ADC, Complex Polarization Propagator'
        self.ostream.print_blank()
        self.ostream.print_header(text)
        self.ostream.print_header('-' * (len(text) + 2))
        self.ostream.print_blank()
        valstr = '{} | {} | {} '.format('  Frequency (a.u)  ',
                                        '  Frequency (eV)',
                                        '  Cross Section (a.u.)  ')
        self.ostream.print_header(valstr)
        self.ostream.print_header('-' * (len(text) + 25))
        for i in range(len(frequencies)):
            valstr = ' {:10.7f} {:18.7f} {:18.7f} '.format(
                frequencies[i], eV * frequencies[i], cross_sections[i])
            self.ostream.print_header(valstr)

        self.ostream.print_blank()

    def print_excited_states(self, adc_drv):
        """
        Prints excited state information to output stream.

        :param adc_drv:
            The ADC driver.
        """

        eV = constants.value('Hartree energy in eV')

        self.ostream.print_blank()
        text = 'ADC Summary of Results'
        self.ostream.print_header(text)
        self.ostream.print_header('-' * (len(text) + 2))
        self.ostream.print_blank()
        try:
            self.ostream.print_block(
                adc_drv.describe(rotatory_strengths=self.adc_ecd))
        except TypeError:
            if self.adc_ecd is False:
                self.ostream.print_block(adc_drv.describe())
            else:
                error_text = os.linesep + os.linesep
                error_text += '*** Rotatory strengths not available.'
                error_text += os.linesep
                error_text += '*** Please update your adcc version. '
                error_text += 'See https://github.com/adc-connect/adcc. '
                error_text += os.linesep
                raise TypeError(error_text)

        if hasattr(adc_drv, 'pe_ptss_correction'):
            text = 'Polarizable Embedding Perturbative Corrections'
            self.ostream.print_blank()
            self.ostream.print_header(text)
            self.ostream.print_header('-' * (len(text) + 2))
            self.ostream.print_blank()
            valstr = '{} | {} | {} | {} | {}'.format('Index',
                                                     'Excitation Energy',
                                                     'Uncorrected Energy',
                                                     'ptSS Correction',
                                                     'ptLR Correction')
            self.ostream.print_header(valstr)
            valstr = ' {}   {}   {}   {}   {}'.format('  #  ',
                                                      '      (eV)      ',
                                                      '      (ev)      ',
                                                      '     (eV)     ',
                                                      '      (eV)      ')
            self.ostream.print_header(valstr)
            self.ostream.print_header('-' * (len(text) + 32))
            for i in range(len(adc_drv.excitation_energy)):
                valstr = ' {:3d} {:18.7f} {:18.7f} {:18.7f} {:17.7f}'.format(
                    i, eV * adc_drv.excitation_energy[i],
                    eV * adc_drv.excitation_energy_uncorrected[i],
                    eV * adc_drv.pe_ptss_correction[i],
                    eV * adc_drv.pe_ptlr_correction[i])
                self.ostream.print_header(valstr)

            self.ostream.print_blank()

    def print_detailed_states(self, adc_drv):
        """
        Prints excited state information to output stream.

        :param adc_drv:
            The ADC driver.
        """

        self.ostream.print_blank()
        text = 'ADC Excited States'
        self.ostream.print_header(text)
        self.ostream.print_header('-' * (len(text) + 2))
        self.ostream.print_blank()
        self.ostream.print_block(
            adc_drv.describe_amplitudes(index_format="homolumo"))
        self.ostream.print_blank()
