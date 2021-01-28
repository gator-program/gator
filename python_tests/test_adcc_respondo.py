from mpi4py import MPI
import numpy as np
import unittest
import os

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.adcdriver import AdcDriver


class TestAdccRespondoIntegration(unittest.TestCase):

    def test_adcc_adc2(self):

        inpfile = os.path.join('inputs', 'water-ccpvdz-adcc.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)
        
        task = GatorTask(inpfile, None, MPI.COMM_WORLD)
        task.input_dict['scf']['checkpoint_file'] = None
        task.input_dict['scf']['conv_thresh'] = '1.0e-8'

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)
        adc_drv = AdcDriver(task.mpi_comm, task.ostream)
        adc_drv.update_settings(task.input_dict['adc'])
        adc_drv.compute(task, scf_drv)

    def test_respondo_adc2(self):

        inpfile = os.path.join('inputs', 'water-ccpvdz-respondo.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        task = GatorTask(inpfile, None, MPI.COMM_WORLD)
        task.input_dict['scf']['checkpoint_file'] = None
        task.input_dict['scf']['conv_thresh'] = '1.0e-8'

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)
        adc_drv = AdcDriver(task.mpi_comm, task.ostream)
        adc_drv.update_settings(task.input_dict['adc'])
        adc_drv.compute(task, scf_drv)


if __name__ == "__main__":
    unittest.main()
