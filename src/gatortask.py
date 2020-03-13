from mpi4py import MPI
from veloxchem import InputParser
from veloxchem import OutputStream
from veloxchem import Molecule
from veloxchem import MolecularBasis
from veloxchem import mpi_master
import sys


class GatorTask:

    def __init__(self,
                 input_fname,
                 output_fname=sys.stdout,
                 comm=MPI.COMM_WORLD):

        self.mpi_comm = comm
        self.mpi_rank = comm.Get_rank()

        if self.mpi_comm.Get_rank() == mpi_master():
            self.input_dict = InputParser(input_fname).get_dict()
        else:
            self.input_dict = {}
        self.input_dict = self.mpi_comm.bcast(self.input_dict,
                                              root=mpi_master())

        if self.mpi_comm.Get_rank() == mpi_master():
            self.ostream = OutputStream(output_fname)
        else:
            self.ostream = OutputStream()

        self.ostream.print_separator()
        self.ostream.print_title('')
        self.ostream.print_title('GATOR 0.0')
        self.ostream.print_title('')
        self.ostream.print_title('Copyright (C) 2019-2020 GATOR developers.')
        self.ostream.print_title('All rights reserved.')
        self.ostream.print_separator()
        self.ostream.print_blank()

        self.ostream.print_info('Reading input file: {}'.format(input_fname))
        self.ostream.print_blank()

        if self.mpi_comm.Get_rank() == mpi_master():
            self.molecule = Molecule.from_dict(self.input_dict['molecule'])
            self.ostream.print_block(self.molecule.get_string())
        else:
            self.molecule = Molecule()

        if self.mpi_comm.Get_rank() == mpi_master():
            if 'basis_path' in self.input_dict['method_settings']:
                basis_path = self.input_dict['method_settings']['basis_path']
            else:
                basis_path = '.'
            basis_name = self.input_dict['method_settings']['basis'].upper()
            self.ao_basis = MolecularBasis.read(self.molecule, basis_name,
                                                basis_path, self.ostream)
            self.min_basis = MolecularBasis.read(self.molecule, 'MIN-CC-PVDZ',
                                                 basis_path)
            self.ostream.print_block(
                self.ao_basis.get_string("Atomic Basis", self.molecule))
        else:
            self.ao_basis = MolecularBasis()
            self.min_basis = MolecularBasis()

        self.molecule.broadcast(self.mpi_rank, self.mpi_comm)
        self.ao_basis.broadcast(self.mpi_rank, self.mpi_comm)
        self.min_basis.broadcast(self.mpi_rank, self.mpi_comm)
