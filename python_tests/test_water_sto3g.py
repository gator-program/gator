from mpi4py import MPI
import unittest
import sys
import os
import shutil

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.watersto3g import WaterSto3g


class TestWaterSto3g(unittest.TestCase):

    def test_water_sto3g(self):

        inpfile = os.path.join('inputs', 'water-sto3g.inp')
        if not os.path.isdir('inputs'):
            inpfile = os.path.join('python_tests', inpfile)

        scffile = os.path.join('inputs', 'water-sto3g.scf.h5')
        if not os.path.isdir('inputs'):
            scffile = os.path.join('python_tests', scffile)

        task = GatorTask(inpfile, sys.stdout, MPI.COMM_WORLD)
        task.input_dict['scf']['checkpoint_file'] = scffile

        if task.mpi_rank == mpi_master():
            bakscffile = scffile + '.bak'
            shutil.copy(scffile, bakscffile)

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

        if task.mpi_rank == mpi_master():
            shutil.move(bakscffile, scffile)

        if task.mpi_rank == mpi_master():
            h5file = os.path.join('inputs', 'water-sto3g.adcmatrix.h5')
            if not os.path.isdir('inputs'):
                h5file = os.path.join('python_tests', h5file)
        else:
            h5file = None

        adc_drv = WaterSto3g(task.mpi_comm, task.ostream)
        adc_results = adc_drv.compute(task.molecule, task.ao_basis,
                                      scf_drv.scf_tensors, h5file)

        if task.mpi_rank == mpi_master():
            pass


if __name__ == "__main__":
    unittest.main()
