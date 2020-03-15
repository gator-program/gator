from adcc import run_adc
from veloxchem import mpi_master
from veloxchem import hartree_in_ev
import sys


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
          part of the core space (int or array)
        - adc_frozen_core: occupied orbitals to be considered inactive during
          the MP and ADC calculation (int or array)
        - adc_frozen_virtual: virtual orbitals to be considered inactive during
          the MP and ADC calculation (int or array)
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
        self.adc_tol = 1e-4
        self.adc_method = 'adc2'
        self.adc_states = 3
        self.adc_singlets = None
        self.adc_triplets = None
        self.adc_spin_flip = None
        self.adc_core_orbitals = None
        self.adc_frozen_core = None
        self.adc_frozen_virtual = None

    def update_settings(self, adc_dict):
        """
        Updates settings in ADC driver.

        :param adc_dict:
            The dictionary of ADC settings.
        """

        if 'tol' in adc_dict:
            self.adc_tol = float(adc_dict['tol'])

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
            self.adc_method = adc_dict['method']

        if 'core_orbitals' in adc_dict:
            orbs = [int(orb) for orb in adc_dict['core_orbitals'].split()]
            if len(orbs) == 1:
                self.adc_core_orbitals = orbs[0]
            else:
                self.adc_core_orbitals = orbs

        if 'frozen_core' in adc_dict:
            orbs = [int(orb) for orb in adc_dict['frozen_core'].split()]
            if len(orbs) == 1:
                self.adc_frozen_core = orbs[0]
            else:
                self.adc_frozen_core = orbs

        if 'frozen_virtual' in adc_dict:
            orbs = [int(orb) for orb in adc_dict['frozen_virtual'].split()]
            if len(orbs) == 1:
                self.adc_frozen_virtual = orbs[0]
            else:
                self.adc_frozen_virtual = orbs

    def compute(self, task, scf_drv):
        """
        Performs ADC calculation.

        :param task:
            The gator task.
        :param scf_drv:
            The converged SCF driver.
        """

        scf_drv.task = task

        if self.rank == mpi_master():
            self.print_header()
            self.ostream.print_info('Running ADC calculation...')
            self.ostream.print_blank()

            original_stdout = sys.stdout
            sys.stdout = self.ostream.stream
            adc_drv = run_adc(scf_drv,
                              method=self.adc_method,
                              core_orbitals=self.adc_core_orbitals,
                              n_states=self.adc_states,
                              n_singlets=self.adc_singlets,
                              n_triplets=self.adc_triplets,
                              n_spin_flip=self.adc_spin_flip,
                              frozen_core=self.adc_frozen_core,
                              frozen_virtual=self.adc_frozen_virtual,
                              conv_tol=self.adc_tol)
            sys.stdout = original_stdout

            self.ostream.print_info('ADC calculation completed.')
            self.ostream.print_blank()

            self.print_excited_states(adc_drv)

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
        cur_str = "ADC method                   : {:s}".format(self.adc_method)
        self.ostream.print_header(cur_str.ljust(str_width))
        if self.adc_states is not None:
            cur_str = "Number of States             : {:d}".format(
                self.adc_states)
        elif self.adc_singlets is not None:
            cur_str = "Number of Singlet States     : {:d}".format(
                self.adc_singlets)
        elif self.adc_triplets is not None:
            cur_str = "Number of Triplet States     : {:d}".format(
                self.adc_triplets)
        else:
            cur_str = "Number of States, Spin-Flip  : {:d}".format(
                self.adc_spin_flip)
        self.ostream.print_header(cur_str.ljust(str_width))

        if self.adc_core_orbitals is not None:
            if isinstance(self.adc_core_orbitals, int):
                cur_str = "CVS-ADC, Core Orbital Space  : {:d}".format(
                    self.adc_core_orbitals)
            else:
                cur_str = "CVS-ADC, Core Orbital Space  :"
                for orb in self.adc_core_orbitals:
                    cur_str += " {:d}".format(orb)
        self.ostream.print_header(cur_str.ljust(str_width))

        if self.adc_frozen_core is not None:
            if isinstance(self.adc_frozen_core, int):
                cur_str = "Frozen Core Orbital Space    : {:d}".format(
                    self.adc_frozen_core)
            else:
                cur_str = "Frozen Core Orbital Space    :"
                for orb in self.adc_frozen_core:
                    cur_str += " {:d}".format(orb)
        self.ostream.print_header(cur_str.ljust(str_width))

        if self.adc_frozen_virtual is not None:
            if isinstance(self.adc_frozen_virtual, int):
                cur_str = "Frozen Virtual Orbital Space : {:d}".format(
                    self.adc_frozen_virtual)
            else:
                cur_str = "Frozen Virtual Orbital Space :"
                for orb in self.adc_frozen_virtual:
                    cur_str += " {:d}".format(orb)
        self.ostream.print_header(cur_str.ljust(str_width))

        cur_str = "Convergence threshold        : {:.1e}".format(self.adc_tol)
        self.ostream.print_header(cur_str.ljust(str_width))

        self.ostream.print_blank()
        self.ostream.flush()

    def print_excited_states(self, adc_drv):
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
        text = "Index |     Excitation Energy, eV    |   Oscillator Strength  "
        self.ostream.print_header(text)

        if hasattr(adc_drv, "converged"):
            self.ostream.print_header('-' * (len(text) + 2))
        else:
            self.ostream.print_header('NOT CONVERGED')
            self.ostream.print_header('-' * (len(text) + 2))

        for i in range(len(adc_drv.excitation_energies)):
            en = adc_drv.excitation_energies[i] * hartree_in_ev()
            osc = adc_drv.oscillator_strengths[i]
            exec_str = " " + (str(i + 1)).rjust(3) + 4 * " "
            exec_str += ("{:7.5f}".format(en)).center(27) + 3 * " "
            exec_str += ("{:5.5f}".format(osc)).center(17) + 3 * " "
            self.ostream.print_header(exec_str)
            self.ostream.flush()
        self.ostream.print_blank()
        self.ostream.flush()
