from mpi4py import MPI
import sys

from .mpiutils import mpi_sanity_check
from .gatortask import GatorTask
from .scfdriver import ScfRestrictedDriver
from .adcdriver import AdcDriver
from .mp2driver import Mp2Driver
from .adconedriver import AdcOneDriver
from .adctwodriver import AdcTwoDriver


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

    if task_type in ['scf', 'adc', 'mp2', 'adc1', 'adc2']:
        scf_dict = {}
        if 'scf' in input_dict:
            scf_dict = dict(input_dict['scf'])

        if ('conv_thresh' not in scf_dict and
                task_type in ['adc', 'mp2', 'adc2']):
            scf_dict['conv_thresh'] = '1.0e-8'

        method_dict = {}
        if 'method_settings' in input_dict:
            method_dict = dict(input_dict['method_settings'])

        method_dict['pe_options'] = {}
        if 'pe' in input_dict:
            method_dict['pe_options'] = dict(input_dict['pe'])

        scf_drv = ScfRestrictedDriver(comm, ostream)
        scf_drv.update_settings(scf_dict, method_dict)
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

        if not scf_drv.is_converged:
            return

    if task_type == 'adc':
        adc_drv = AdcDriver(comm, ostream)
        adc_dict = input_dict['adc'] if 'adc' in input_dict else {}
        adc_drv.update_settings(adc_dict, scf_drv)
        adc_drv.compute(task, scf_drv)

    if task_type == 'mp2':
        mp2_drv = Mp2Driver(comm, ostream)
        mp2_dict = input_dict['mp2'] if 'mp2' in input_dict else {}
        mp2_drv.update_settings(mp2_dict, scf_drv)
        mp2_drv.compute(task.molecule, task.ao_basis, scf_drv.scf_tensors)

    if task_type == 'adc1':
        adc_one_drv = AdcOneDriver(comm, ostream)
        adc_one_dict = input_dict['adc1'] if 'adc1' in input_dict else {}
        adc_one_drv.update_settings(adc_one_dict, scf_drv)
        adc_one_drv.compute(task.molecule, task.ao_basis, scf_drv.scf_tensors)

    if task_type == 'adc2':
        adc_two_drv = AdcTwoDriver(comm, ostream)
        adc_two_dict = input_dict['adc2'] if 'adc2' in input_dict else {}
        adc_two_drv.update_settings(adc_two_dict, scf_drv)
        adc_two_drv.compute(task.molecule, task.ao_basis, scf_drv.scf_tensors)

    task.finish()


if __name__ == "__main__":
    main()
