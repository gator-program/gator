from mpi4py import MPI
import numpy as np
import unittest
import h5py
import os

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.adcmatrixdriver import AdcMatrixDriver


class TestAdcMatrix(unittest.TestCase):

    def test_adc_matrix(self):

        inpfile = os.path.join('inputs', 'water-sto3g.inp')
        if not os.path.isdir('inputs'):
            inpfile = os.path.join('python_tests', inpfile)

        scffile = os.path.join('inputs', 'water-sto3g.scf.h5')
        if not os.path.isdir('inputs'):
            scffile = os.path.join('python_tests', scffile)

        task = GatorTask(inpfile, None, MPI.COMM_WORLD)
        task.input_dict['scf']['checkpoint_file'] = scffile

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

        adc_drv = AdcMatrixDriver(task.mpi_comm, task.ostream)
        adc_matrix = adc_drv.compute(task.molecule, task.ao_basis,
                                     scf_drv.scf_tensors)

        if task.mpi_rank == mpi_master():

            h5file = os.path.join('inputs', 'water-sto3g.adcmatrix.h5')
            if not os.path.isdir('inputs'):
                h5file = os.path.join('python_tests', h5file)

            hf = h5py.File(h5file, 'r')
            ref_matrix = np.array(hf.get('m'))
            hf.close()

            eigs, vecs = np.linalg.eigh(adc_matrix)
            ref_eigs, ref_vecs = np.linalg.eigh(ref_matrix)

            mat_diff = np.max(np.abs(adc_matrix - ref_matrix))
            self.assertTrue(mat_diff < 1.0e-12)

            e_diff = np.max(np.abs(eigs - ref_eigs))
            self.assertTrue(e_diff < 1.0e-12)


if __name__ == "__main__":
    unittest.main()
