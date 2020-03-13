from adcc import run_adc
from veloxchem import mpi_master
from contextlib import redirect_stdout
import io
import os


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
        - adc_tol:
        - adc_method:
        - adc_states:
        - adc_singlets:
        - adc_triplets:
        - adc_core_orbitals:
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

        # ADC settings
        self.adc_tol = 1e-4
        self.adc_method = 'adc2'
        self.adc_states = 3
        self.adc_singlets = None
        self.adc_triplets = None
        self.adc_core_orbitals = None

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
        if 'triplets' in adc_dict:
            self.adc_triplets = int(adc_dict['triplets'])

        if 'method' in adc_dict:
            self.adc_method = adc_dict['method']
        if 'core_orbitals' in adc_dict:
            self.adc_core_orbitals = int(adc_dict['core_orbitals'])

    def compute(self, task, scf_drv):
        """
        Performs ADC calculation.

        :param task:
            The gator task.
        :param scf_drv:
            The converged SCF driver.
        """

        scf_drv.task = task
        width = 92

        if self.rank == mpi_master():
            self.print_header()

            # redirect stdout to string
            with io.StringIO() as buf, redirect_stdout(buf):
                adc_drv = run_adc(scf_drv,
                                  method=self.adc_method,
                                  core_orbitals=self.adc_core_orbitals,
                                  n_states=self.adc_states,
                                  n_singlets=self.adc_singlets,
                                  conv_tol=self.adc_tol)
                for line in buf.getvalue().split(os.linesep):
                    self.ostream.print_header(line.ljust(width))

            self.ostream.print_header('End of ADC calculation.'.ljust(width))
            for line in adc_drv.describe().split(os.linesep):
                self.ostream.print_header(line.ljust(width))

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
        cur_str = "Number of States             : {:d}".format(self.adc_states)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = "Convergence threshold        : {:.1e}".format(self.adc_tol)
        self.ostream.print_header(cur_str.ljust(str_width))

        # todo: print adc_singlets, adc_triplets, and adc_core_orbitals

        self.ostream.print_blank()
        self.ostream.flush()
