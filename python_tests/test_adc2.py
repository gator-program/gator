from mpi4py import MPI
import numpy as np
import unittest
import os

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.adctwodriver import AdcTwoDriver


class TestADC2(unittest.TestCase):

    def run_adc2(self, inpfile, data_lines):

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
            self.assertTrue(np.max(np.abs(exc_ene - ref_exc_ene)) < 1.0e-6)

    def test_adc2_sto3g(self):

        inpfile = os.path.join('inputs', 'water-sto3g.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            0     0.4705131
            1     0.5725549
            2     0.5936734
            3     0.7129688
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        self.run_adc2(inpfile, data_lines)

    def test_adc2_def2svp(self):

        inpfile = os.path.join('inputs', 'water-def2svp.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        raw_data = """
            0      0.299412
            1     0.3767263
            2     0.3839022
            3     0.4627939
        """
        data_lines = raw_data.split(os.linesep)[1:-1]

        self.run_adc2(inpfile, data_lines)


if __name__ == "__main__":
    unittest.main()
