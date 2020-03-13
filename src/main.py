from mpi4py import MPI
import sys

from .mpiutils import mpi_sanity_check
from .gatortask import GatorTask
from .scfdriver import ScfRestrictedDriver
from .mp2driver import Mp2Driver
import adcc

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

    scf_drv = ScfRestrictedDriver(comm, ostream)
    if 'scf' in input_dict:
        scf_drv.update_settings(input_dict['scf'])
    scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)
    scf_drv.task = task

    task_type = None
    if 'jobs' in input_dict and 'task' in input_dict['jobs']:
        task_type = input_dict['jobs']['task'].lower()

    if task_type == 'mp2':
        mp2_drv = Mp2Driver(comm, ostream)
        if 'mp2' in input_dict:
            mp2_drv.update_settings(input_dict['mp2'])
        mp2_drv.compute(task.molecule, task.ao_basis, scf_drv.mol_orbs)

    if task_type == 'adc':
        adc_tol = 1e-3
        adc_method = "adc0"
        adc_states = None
        adc_singlets = None
        adc_triplets = None
        adc_core_orbitals = None

        if "tol" in input_dict["adc"]:     
            adc_tol = float(input_dict["adc"]["tol"])

        if "core_orbitals" in input_dict["adc"]:
            adc_core_orbitals = int(input_dict["adc"]["core_orbitals"])

        if "states" in input_dict["adc"]:
            adc_states = int(input_dict["adc"]["states"])

        if "singlets" in input_dict["adc"]:
            adc_singlets = int(input_dict["adc"]["singlets"])

        if "triplets" in input_dict["adc"]:
            adc_triplets = int(input_dict["adc"]["triplets"])

        if "method" in input_dict["adc"]:
            adc_method = input_dict["adc"]["method"]
    

        print("Algebraic Diagrammatic Construction Scheme for the Polarization Propagator\n")


        adc_drv = adcc.run_adc(scf_drv, method=adc_method, core_orbitals=adc_core_orbitals,
                               n_states=adc_states, n_singlets=adc_singlets, conv_tol=adc_tol)


        print("End of ADC calculation.")
        print(adc_drv.describe())

if __name__ == "__main__":
    main()
