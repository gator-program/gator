from mpi4py import MPI
import numpy as np
import unittest
import os

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.adctwodriver import AdcTwoDriver


class TestADC2(unittest.TestCase):

    def run_adc2(self, inpfile, ref_e_mp2, data_lines):

        task = GatorTask(inpfile, None, MPI.COMM_WORLD)
        task.input_dict['scf']['checkpoint_file'] = None
        task.input_dict['scf']['conv_thresh'] = '1.0e-8'

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

        ref_exc_ene = [float(line.split()[1]) for line in data_lines]
        ref_s_comp2 = [float(line.split()[2]) for line in data_lines]

        adc_drv = AdcTwoDriver(task.mpi_comm, task.ostream)
        adc_drv.update_settings({'nstates': len(ref_exc_ene)})
        adc_results = adc_drv.compute(task.molecule, task.ao_basis,
                                      scf_drv.scf_tensors)

        if task.mpi_rank == mpi_master():
            e_mp2 = adc_results['mp2_energy']
            exc_ene = adc_results['eigenvalues']
            s_comp2 = adc_results['s_components_2']
            self.assertTrue(np.max(np.abs(e_mp2 - ref_e_mp2)) < 1.0e-10)
            self.assertTrue(np.max(np.abs(exc_ene - ref_exc_ene)) < 1.0e-7)
            self.assertTrue(np.max(np.abs(s_comp2 - ref_s_comp2)) < 1.0e-4)

    def test_adc2_sto3g(self):

        inpfile = os.path.join('inputs', 'water-sto3g.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            0     0.4705131     0.985
            1     0.5725549    0.9896
            2     0.5936734    0.9837
            3     0.7129688    0.9872
            4     0.8396973    0.9913
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        ref_e_mp2 = -0.0342588021

        self.run_adc2(inpfile, ref_e_mp2, data_lines)

    def test_adc2_def2svp(self):

        inpfile = os.path.join('inputs', 'water-def2svp.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            0      0.299412    0.9523
            1     0.3767263    0.9555
            2     0.3839021    0.9515
            3     0.4627936    0.9556
            4     0.5586361    0.9644
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        ref_e_mp2 = -0.2026678840

        self.run_adc2(inpfile, ref_e_mp2, data_lines)

    def test_adc2_augccpvdz(self):

        inpfile = os.path.join('inputs', 'water-augccpvdz.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            0       0.25812    0.9298
            1     0.3171875     0.925
            2     0.3368847      0.93
            3     0.3901644    0.9323
            4     0.3948281    0.9258
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        ref_e_mp2 = -0.2209861606

        self.run_adc2(inpfile, ref_e_mp2, data_lines)


if __name__ == "__main__":
    unittest.main()
