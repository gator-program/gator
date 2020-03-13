from mpi4py import MPI
import sys

from .mpiutils import mpi_sanity_check
from .gatortask import GatorTask
from .scfdriver import ScfRestrictedDriver
from .mp2driver import Mp2Driver
from .adcdriver import AdcDriver


def main():

    mpi_sanity_check(sys.argv)
    comm = MPI.COMM_WORLD

    input_fname = sys.argv[1]
    output_fname = sys.stdout
    if len(sys.argv) > 2:
        output_fname = sys.argv[2]

    task = GatorTask(input_fname, output_fname, comm)
    input_dict = task.input_dict
    ostream = task.ostream

    task_type = None
    if 'jobs' in input_dict and 'task' in input_dict['jobs']:
        task_type = input_dict['jobs']['task'].lower()

    if task_type in ['scf', 'mp2', 'adc']:
        scf_drv = ScfRestrictedDriver(comm, ostream)
        if 'scf' in input_dict:
            scf_drv.update_settings(input_dict['scf'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

    if task_type == 'mp2':
        mp2_drv = Mp2Driver(comm, ostream)
        if 'mp2' in input_dict:
            mp2_drv.update_settings(input_dict['mp2'])
        mp2_drv.compute(task.molecule, task.ao_basis, scf_drv.mol_orbs)

    if task_type == 'adc':
        adc_drv = AdcDriver(comm, ostream)
        if 'adc' in input_dict:
            adc_drv.update_settings(input_dict['adc'])
        adc_drv.compute(task, scf_drv)


if __name__ == "__main__":
    main()
