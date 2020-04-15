from mpi4py import MPI
import numpy as np
import unittest
import h5py
import os

from gator.mpiutils import mpi_master
from gator.gatortask import GatorTask
from gator.scfdriver import ScfRestrictedDriver
from gator.adctwodriver import AdcTwoDriver
from gator.mointsdriver import MOIntegralsDriver


class TestAdcSigmaBuild(unittest.TestCase):

    def test_adc_sigma_build(self):

        # input and checkpoint files

        inpfile = os.path.join('inputs', 'water-sto3g.inp')
        if not os.path.isdir('inputs'):
            inpfile = os.path.join('python_tests', inpfile)

        scffile = os.path.join('inputs', 'water-sto3g.scf.h5')
        if not os.path.isdir('inputs'):
            scffile = os.path.join('python_tests', scffile)

        # run scf

        comm = MPI.COMM_WORLD
        task = GatorTask(inpfile, None, comm)
        task.input_dict['scf']['checkpoint_file'] = scffile

        scf_drv = ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scf_drv.update_settings(task.input_dict['scf'],
                                task.input_dict['method_settings'])
        scf_drv.compute(task.molecule, task.ao_basis, task.min_basis)

        scf_tensors = scf_drv.scf_tensors
        molecule = task.molecule
        basis = task.ao_basis
        ostream = task.ostream

        # process MOs

        if task.mpi_rank == mpi_master():
            mo = scf_tensors['C']
            ea = scf_tensors['E']
        else:
            mo = None
            ea = None
        mo = comm.bcast(mo, root=mpi_master())
        ea = comm.bcast(ea, root=mpi_master())

        nocc = molecule.number_of_alpha_electrons()
        nvir = mo.shape[1] - nocc

        eocc = ea[:nocc]
        evir = ea[nocc:]
        e_oo = eocc.reshape(-1, 1) + eocc
        e_vv = evir.reshape(-1, 1) + evir
        e_ov = -eocc.reshape(-1, 1) + evir

        epsilon = {
            'o': eocc,
            'v': evir,
            'oo': e_oo,
            'vv': e_vv,
            'ov': e_ov,
        }

        # compute MO integrals

        moints_drv = MOIntegralsDriver(comm, ostream)
        moints_drv.update_settings({
            'qq_type': scf_drv.qq_type,
            'eri_thresh': scf_drv.eri_thresh
        })
        mo_indices, mo_integrals = moints_drv.compute(molecule, basis,
                                                      scf_tensors)

        # compute auxiliary matrices

        adc_drv = AdcTwoDriver(task.mpi_comm, task.ostream)

        xA_ab, xB_ij = adc_drv.compute_xA_xB(epsilon, mo_indices, mo_integrals)

        if task.mpi_rank == mpi_master():
            fa = scf_tensors['F'][0]
            fmo = np.matmul(mo.T, np.matmul(fa, mo))
            fab = fmo[nocc:, nocc:]
            fij = fmo[:nocc, :nocc]
        else:
            fab = None
            fij = None

        auxiliary_matrices = {
            'fab': fab,
            'fij': fij,
            'xA_ab': xA_ab,
            'xB_ij': xB_ij,
        }

        # read reference trial and sigma vectors

        if task.mpi_rank == mpi_master():

            h5file = os.path.join('inputs', 'water-sto3g.adcmatrix.h5')
            if not os.path.isdir('inputs'):
                h5file = os.path.join('python_tests', h5file)

            hf = h5py.File(h5file, 'r')
            ref_matrix = np.array(hf.get('m'))
            hf.close()

            ref_eigs, ref_vecs = np.linalg.eigh(ref_matrix)

            reigs = np.array((
                ref_eigs[3],
                ref_eigs[10],
                ref_eigs[14],
                ref_eigs[18],
                ref_eigs[22],
            ))

            s_dim = nocc * nvir * 4

            ref_trials = np.vstack((
                ref_vecs[:s_dim, 3],
                ref_vecs[:s_dim, 10],
                ref_vecs[:s_dim, 14],
                ref_vecs[:s_dim, 18],
                ref_vecs[:s_dim, 22],
            )).transpose()

            trial_mat = np.zeros((nocc * nvir, ref_trials.shape[1]))
            for i in range(ref_trials.shape[1]):
                trial_ov = ref_trials[:, i].reshape(nocc * 2, nvir * 2)
                trial_mat[:, i] = trial_ov[:nocc, :nvir].reshape(nocc * nvir)[:]

            ss_mat = ref_matrix[:s_dim, :s_dim]
            sd_mat = ref_matrix[:s_dim, s_dim:]
            dd_diag = np.diag(ref_matrix[s_dim:, s_dim:])

            ref_sigmas = np.zeros(ref_trials.shape)
            for i in range(ref_trials.shape[1]):
                rjb = ref_trials[:, i]
                Ab = np.dot(sd_mat.T, rjb) / (reigs[i] - dd_diag)
                ref_sigma = np.dot(ss_mat, rjb) + np.dot(sd_mat, Ab)
                ref_sigmas[:, i] = ref_sigma[:]
        else:
            trial_mat = None
            reigs = None

        trial_mat = comm.bcast(trial_mat, root=mpi_master())
        reigs = comm.bcast(reigs, root=mpi_master())

        # compute sigma vectors

        sigma_mat = adc_drv.compute_sigma(trial_mat, reigs, epsilon,
                                          auxiliary_matrices, mo_indices,
                                          mo_integrals)

        # compare sigma vectors with reference

        if task.mpi_rank == mpi_master():
            for i in range(sigma_mat.shape[1]):
                sigma = sigma_mat[:, i].reshape(nocc, nvir)
                ref_sigma = ref_sigmas[:, i].reshape(nocc * 2, nvir * 2)
                s_diff = np.max(np.abs(sigma - ref_sigma[:nocc, :nvir]))
                self.assertTrue(s_diff < 1.0e-12)


if __name__ == "__main__":
    unittest.main()
