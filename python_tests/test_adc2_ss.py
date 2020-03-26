from mpi4py import MPI
import numpy as np
import unittest
import os

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.adctwodriver import AdcTwoDriver


class TestADC2SS(unittest.TestCase):

    def run_adc2_ss(self, inpfile, data_lines):

        task = GatorTask(inpfile, None, MPI.COMM_WORLD)
        task.input_dict['scf']['checkpoint_file'] = None

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

        ref_exc_ene = [float(line.split()[1]) for line in data_lines]

        adc_drv = AdcTwoDriver(task.mpi_comm, task.ostream)
        adc_drv.update_settings({'nstates': len(ref_exc_ene)})
        adc_results = adc_drv.compute(task.molecule, task.ao_basis,
                                      scf_drv.scf_tensors)

        if task.mpi_rank == mpi_master():
            exc_ene = adc_results['eigenvalues']
            self.assertTrue(np.max(np.abs(exc_ene - ref_exc_ene)) < 1.0e-8)

    def test_adc2_ss_sto3g(self):

        inpfile = os.path.join('inputs', 'water-sto3g.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            1      0.5002980049
            2      0.5926285661
            3      0.6238801380
            4      0.7361424446
            5      0.8556359724
            6      1.1111906878
            7      1.4884689297
            8      1.5552700206
            9     20.1224696835
            10    20.1829764647
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        self.run_adc2_ss(inpfile, data_lines)

    def test_adc2_ss_def2svp(self):

        inpfile = os.path.join('inputs', 'water-def2svp.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            1      0.4097929939
            2      0.4812659311
            3      0.4941383673
            4      0.5652404672
            5      0.6415313816
            6      0.7570072816
            7      0.9394479608
            8      0.9896442746
            9      1.0549011302
            10     1.0719459404
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        self.run_adc2_ss(inpfile, data_lines)

    def test_adc2_ss_augccpvdz(self):

        inpfile = os.path.join('inputs', 'water-augccpvdz.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            1      0.3887433948
            2      0.4504166332
            3      0.4652852384
            4      0.5139515661
            5      0.5247856951
            6      0.5379187805
            7      0.5504043144
            8      0.5613549819
            9      0.5909831710
            10     0.5943317102
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        self.run_adc2_ss(inpfile, data_lines)


if __name__ == "__main__":
    unittest.main()
