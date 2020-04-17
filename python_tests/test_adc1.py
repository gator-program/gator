from mpi4py import MPI
import numpy as np
import unittest
import os

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.adconedriver import AdcOneDriver


class TestADC1(unittest.TestCase):

    def run_adc1(self, inpfile, data_lines):

        task = GatorTask(inpfile, None, MPI.COMM_WORLD)
        task.input_dict['scf']['checkpoint_file'] = None

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

        ref_exc_ene = [float(line.split()[1]) for line in data_lines]

        adc_drv = AdcOneDriver(task.mpi_comm, task.ostream)
        adc_drv.update_settings({'nstates': len(ref_exc_ene)})
        adc_results = adc_drv.compute(task.molecule, task.ao_basis,
                                      scf_drv.scf_tensors)

        if task.mpi_rank == mpi_master():
            exc_ene = adc_results['eigenvalues']
            self.assertTrue(np.max(np.abs(exc_ene - ref_exc_ene)) < 1.0e-6)

    def test_adc1_sto3g(self):

        inpfile = os.path.join('inputs', 'water-sto3g.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            S1:      0.48343609
            S2:      0.57420043
            S3:      0.60213699
            S4:      0.71027728
            S5:      0.82604406
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        self.run_adc1(inpfile, data_lines)

    def test_adc1_def2svp(self):

        inpfile = os.path.join('inputs', 'water-def2svp.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            S1:      0.34391555
            S2:      0.41009015
            S3:      0.42867285
            S4:      0.49455594
            S5:      0.57126810
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        self.run_adc1(inpfile, data_lines)

    def test_adc1_augccpvdz(self):

        inpfile = os.path.join('inputs', 'water-augccpvdz.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            S1:      0.32033338
            S2:      0.38142661
            S3:      0.39745700
            S4:      0.44605997
            S5:      0.45706770
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        self.run_adc1(inpfile, data_lines)

if __name__ == "__main__":
    unittest.main()
