from veloxchem import InputParser
from veloxchem import OutputStream
from veloxchem import Molecule
from veloxchem import MolecularBasis
from veloxchem import mpi_master
import time as tm
import os


class GatorTask:

    def __init__(self, input_fname, output_fname, comm):

        self.mpi_comm = comm
        self.mpi_rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()

        if self.mpi_rank == mpi_master():
            self.input_dict = InputParser(input_fname).get_dict()
        else:
            self.input_dict = {}
        self.input_dict = self.mpi_comm.bcast(self.input_dict,
                                              root=mpi_master())

        if self.mpi_rank == mpi_master():
            self.ostream = OutputStream(output_fname)
        else:
            self.ostream = OutputStream()

        start_time = tm.time()

        self.ostream.print_separator()
        self.ostream.print_title('')
        self.ostream.print_title('GATOR 0.0')
        self.ostream.print_title('')
        self.ostream.print_title('Copyright (C) 2019-2020 GATOR developers.')
        self.ostream.print_title('All rights reserved.')
        self.ostream.print_separator()
        exec_str = 'GATOR execution started'
        if self.mpi_size > 1:
            exec_str += ' on ' + str(self.mpi_size) + ' compute nodes'
        exec_str += ' at ' + tm.asctime(tm.localtime(start_time)) + '.'
        self.ostream.print_title(exec_str)
        self.ostream.print_separator()
        self.ostream.print_blank()

        if 'OMP_NUM_THREADS' in os.environ:
            self.ostream.print_info(
                'Using {} OpenMP threads per compute node.'.format(
                    os.environ['OMP_NUM_THREADS']))
            self.ostream.print_blank()

        self.ostream.print_info('Reading input file: {}'.format(input_fname))
        self.ostream.print_blank()

        if self.mpi_rank == mpi_master():
            if 'basis_path' in self.input_dict['method_settings']:
                basis_path = self.input_dict['method_settings']['basis_path']
            else:
                basis_path = '.'
            basis_name = self.input_dict['method_settings']['basis'].upper()

            self.molecule = Molecule.from_dict(self.input_dict['molecule'])
            self.ao_basis = MolecularBasis.read(self.molecule, basis_name,
                                                basis_path, self.ostream)
            self.min_basis = MolecularBasis.read(self.molecule, 'MIN-CC-PVDZ',
                                                 basis_path)

            self.ostream.print_block(self.molecule.get_string())
            self.ostream.print_block(
                self.ao_basis.get_string("Atomic Basis", self.molecule))
        else:
            self.molecule = Molecule()
            self.ao_basis = MolecularBasis()
            self.min_basis = MolecularBasis()

        self.molecule.broadcast(self.mpi_rank, self.mpi_comm)
        self.ao_basis.broadcast(self.mpi_rank, self.mpi_comm)
        self.min_basis.broadcast(self.mpi_rank, self.mpi_comm)
