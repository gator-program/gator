from mpi4py import MPI
import unittest
import os

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.mp2driver import Mp2Driver


class TestMP2(unittest.TestCase):

    def run_mp2(self, inpfile, ref_e_mp2):

        task = GatorTask(inpfile, None, MPI.COMM_WORLD)
        task.input_dict['scf']['checkpoint_file'] = None

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

        mp2_drv = Mp2Driver(task.mpi_comm, task.ostream)
        mp2_drv.update_settings({}, scf_drv)
        e_mp2 = mp2_drv.compute(task.molecule, task.ao_basis,
                                scf_drv.scf_tensors)

        if task.mpi_rank == mpi_master():
            self.assertTrue(abs(e_mp2 - ref_e_mp2) < 1.0e-10)

    def test_mp2_sto3g(self):

        inpfile = os.path.join('inputs', 'water-sto3g.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        ref_e_mp2 = -0.034258802148

        self.run_mp2(inpfile, ref_e_mp2)

    def test_mp2_def2svp(self):

        inpfile = os.path.join('inputs', 'water-def2svp.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        ref_e_mp2 = -0.202667884590

        self.run_mp2(inpfile, ref_e_mp2)

    def test_mp2_augccpvdz(self):

        inpfile = os.path.join('inputs', 'water-augccpvdz.inp')
        if not os.path.isfile(inpfile):
            inpfile = os.path.join('python_tests', inpfile)

        ref_e_mp2 = -0.220986160311

        self.run_mp2(inpfile, ref_e_mp2)


if __name__ == "__main__":
    unittest.main()
