from mpi4py import MPI
from .mpiutils import mpi_sanity_check
from .ioutils import InputReader
from .scfdriver import ScfRestrictedDriver
from .mp2driver import Mp2Driver
import sys


def main():

    mpi_sanity_check(sys.argv)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    input_fname = sys.argv[1]
    output_fname = sys.stdout
    if len(sys.argv) > 2:
        output_fname = sys.argv[2]

    input_reader = InputReader(input_fname, output_fname, comm)

    input_dict = input_reader.get_input_dict()
    ostream = input_reader.get_output_stream()

    molecule = input_reader.get_molecule(input_dict)
    ostream.print_block(molecule.get_string())

    basis = input_reader.get_basis(input_dict, molecule, ostream)
    min_basis = input_reader.get_min_basis(input_dict, molecule)
    ostream.print_block(basis.get_string('Atomic Basis', molecule))

    molecule.broadcast(rank, comm)
    basis.broadcast(rank, comm)
    min_basis.broadcast(rank, comm)

    scf_drv = ScfRestrictedDriver(comm, ostream)
    if 'scf' in input_dict:
        scf_drv.update_settings(input_dict['scf'])
    scf_drv.compute(molecule, basis, min_basis)

    mp2_drv = Mp2Driver(comm, ostream)
    if 'mp2' in input_dict:
        mp2_drv.update_settings(input_dict['mp2'])
    mp2_drv.compute(molecule, basis, scf_drv.mol_orbs)


if __name__ == "__main__":
    main()
