from mpi4py import MPI
from veloxchem import InputParser
from veloxchem import OutputStream
from veloxchem import Molecule
from veloxchem import MolecularBasis
from veloxchem import mpi_master
import sys


class InputReader:

    def __init__(self,
                 input_fname,
                 output_fname=sys.stdout,
                 comm=MPI.COMM_WORLD):

        self.comm = comm

        if self.comm.Get_rank() == mpi_master():
            self.input_dict = InputParser(input_fname).get_dict()
        else:
            self.input_dict = {}
        self.input_dict = self.comm.bcast(self.input_dict, root=mpi_master())

        if self.comm.Get_rank() == mpi_master():
            self.ostream = OutputStream(output_fname)
        else:
            self.ostream = OutputStream()

        self.print_gator_header()
        self.ostream.print_info('Reading input file: {}'.format(input_fname))
        self.ostream.print_blank()

    def print_gator_header(self):

        self.ostream.print_separator()
        self.ostream.print_title('')
        self.ostream.print_title('GATOR 0.0')
        self.ostream.print_title('')
        self.ostream.print_title('Copyright (C) 2019-2020 GATOR developers.')
        self.ostream.print_title('All rights reserved.')
        self.ostream.print_separator()
        self.ostream.print_blank()

    def get_input_dict(self):

        return self.input_dict

    def get_output_stream(self):

        return self.ostream

    def get_molecule(self, input_dict):

        if self.comm.Get_rank() == mpi_master():
            return Molecule.from_dict(self.input_dict['molecule'])
        else:
            return Molecule()

    def get_basis(self, input_dict, molecule, ostream):

        if self.comm.Get_rank() == mpi_master():
            if 'basis_path' in self.input_dict['method_settings']:
                basis_path = self.input_dict['method_settings']['basis_path']
            else:
                basis_path = '.'
            basis_name = self.input_dict['method_settings']['basis'].upper()
            return MolecularBasis.read(molecule, basis_name, basis_path,
                                       ostream)
        else:
            return MolecularBasis()

    def get_min_basis(self, input_dict, molecule):

        if self.comm.Get_rank() == mpi_master():
            if 'basis_path' in input_dict['method_settings']:
                basis_path = input_dict['method_settings']['basis_path']
            else:
                basis_path = '.'
            return MolecularBasis.read(molecule, 'MIN-CC-PVDZ', basis_path)
        else:
            return MolecularBasis()
