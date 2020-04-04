from mpi4py import MPI
from veloxchem import ExcitationVector
from veloxchem import mpi_master
from veloxchem import hartree_in_ev
from veloxchem.veloxchemlib import szblock
from veloxchem import molorb
from veloxchem import get_qq_type
from veloxchem import BlockDavidsonSolver
from veloxchem import MolecularOrbitals
from .mointsdriver import MOIntegralsDriver as MOIntsDriver
import numpy as np
import time as tm


class AdcTwoDriver:
    """
    Implements ADC(2) computation driver.

    :param comm:
        The MPI communicator.
    :param ostream:
        The output stream.

    Instance variable
        - nstates: The number of excited states determined by driver.
        - triplet: The triplet excited states flag.
        - eri_thresh: The electron repulsion integrals screening
          threshold.
        - conv_thresh: The excited states convergence threshold.
        - max_iter: The maximum number of excited states driver
          iterations.
        - cur_iter: The current number of excited states driver
          iterations.
        - solver: The eigenvalues solver.
        - is_converged: The flag for excited states convergence.
        - rank: The rank of MPI process.
        - nodes: The number of MPI processes.
    """

    def __init__(self, comm, ostream):
        """
        Initializes ADC(2) computation driver.
        """

        # excited states information
        self.nstates = 3
        self.triplet = False

        # ERI settings
        self.eri_thresh = 1.0e-15
        self.qq_type = 'QQ_DEN'

        # solver setup
        self.conv_thresh = 1.0e-4
        self.max_iter = 50
        self.cur_iter = 0
        self.solver = None
        self.is_converged = None

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

    def update_settings(self, settings, scf_drv=None):
        """
        Updates settings in ADC(2) driver.

        :param settings:
            The settings for the driver.
        :param scf_drv:
            The scf driver.
        """

        # calculation type
        if 'nstates' in settings:
            self.nstates = int(settings['nstates'])
        if 'spin' in settings:
            self.triplet = (settings['spin'][0].upper() == 'T')

        # solver settings
        if 'conv_thresh' in settings:
            self.conv_thresh = float(settings['conv_thresh'])
        if 'max_iter' in settings:
            self.max_iter = int(settings['max_iter'])

        # ERI settings
        if 'eri_thresh' in settings:
            self.eri_thresh = float(settings['eri_thresh'])
        elif scf_drv is not None:
            # inherit from SCF
            self.eri_thresh = scf_drv.eri_thresh
        if 'qq_type' in settings:
            self.qq_type = settings['qq_type'].upper()
        elif scf_drv is not None:
            # inherit from SCF
            self.qq_type = scf_drv.qq_type

    def compute(self, molecule, basis, scf_tensors):
        """
        Performs ADC(2) calculation.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param scf_tensors:
            The tensors from converged SCF wavefunction.
        :return:
            A dictionary containing excitation energies.
        """

        if self.rank == mpi_master():
            mo = scf_tensors['C']
            ea = scf_tensors['E']
            mol_orbs = MolecularOrbitals([mo], [ea], molorb.rest)
        else:
            mo = None
            ea = None
            mol_orbs = MolecularOrbitals()
        mo = self.comm.bcast(mo, root=mpi_master())
        ea = self.comm.bcast(ea, root=mpi_master())

        nocc = molecule.number_of_alpha_electrons()
        nvir = mo.shape[1] - nocc

        eocc = ea[:nocc]
        evir = ea[nocc:]
        e_oo = eocc.reshape(-1, 1) + eocc
        e_vv = evir.reshape(-1, 1) + evir

        if self.rank == mpi_master():
            self.print_header()
            self.ostream.print_info(
                'Number of occupied orbitals: {:d}'.format(nocc))
            self.ostream.print_info(
                'Number of virtual orbitals: {:d}'.format(nvir))
            self.ostream.print_blank()

            fa = scf_tensors['F'][0]
            fmo = np.matmul(mo.T, np.matmul(fa, mo))
            fab = fmo[nocc:, nocc:]
            fij = fmo[:nocc, :nocc]

        # set start time

        start_time = tm.time()

        # set up trial excitation vectors on master node

        diag_mat, trial_vecs, reigs = self.gen_trial_vectors(mol_orbs, molecule)

        if self.rank == mpi_master():
            trial_mat = trial_vecs[0].zvector_to_numpy()
            for vec in trial_vecs[1:]:
                trial_mat = np.hstack((trial_mat, vec.zvector_to_numpy()))
        else:
            trial_mat = None
        trial_mat = self.comm.bcast(trial_mat, root=mpi_master())

        # block Davidson algorithm setup

        self.solver = BlockDavidsonSolver()

        # MO integrals

        moints_drv = MOIntsDriver(self.comm, self.ostream)
        moints_drv.update_settings({
            'qq_type': self.qq_type,
            'eri_thresh': self.eri_thresh
        })
        mo_indices, mo_integrals = moints_drv.compute(molecule, basis,
                                                      scf_tensors)

        # precompute xA_ab and xB_ij matrices for 2nd-order contribution

        xA_ab = np.zeros((nvir, nvir))
        for ind, (k, l) in enumerate(mo_indices['oo']):
            de = e_vv - eocc[k] - eocc[l]
            vv = mo_integrals['chem_ovov_K'][ind]
            # [ 2 (ka|lc) - (kc|la) ] (kb|lc)
            ac_ca = 2.0 * vv - vv.T
            bc = vv
            xA_ab += np.matmul(ac_ca / de, bc.T)
            xA_ab += np.matmul(ac_ca, bc.T / de)

        xB_ij = np.zeros((nocc, nocc))
        for ind, (c, d) in enumerate(mo_indices['vv']):
            de = evir[c] + evir[d] - e_oo
            oo = mo_integrals['chem_vovo_K'][ind]
            # [ 2 (ci|dk) - (ck|di) ] (cj|dk)
            ik_ki = 2.0 * oo - oo.T
            jk = oo
            xB_ij += np.matmul(ik_ki / de, jk.T)
            xB_ij += np.matmul(ik_ki, jk.T / de)

        # start iterations

        for iteration in range(self.max_iter):

            iter_start_time = tm.time()

            iter_timing = []
            t0 = tm.time()

            # precompute kc vectors for 2nd-order contribution (term C)

            kc = np.zeros((trial_mat.shape[1], 2, nocc, nvir))

            for vecind in range(trial_mat.shape[1]):
                rjb = trial_mat[:, vecind].reshape(nocc, nvir)
                kc1 = kc[vecind, 0, :, :]
                kc2 = kc[vecind, 1, :, :]

                for ind, (k, j) in enumerate(mo_indices['oo']):
                    de = e_vv - eocc[k] - eocc[j]
                    vv = mo_integrals['chem_ovov_K'][ind]
                    # [ 2 (kc|jb) - (kb|jc) ] R_jb
                    cb_bc = 2.0 * vv - vv.T
                    kc1[k, :] += np.dot(cb_bc, rjb[j, :])
                    kc2[k, :] += np.dot(cb_bc / de, rjb[j, :])

            iter_timing.append(('computing kc vectors', tm.time() - t0))
            t0 = tm.time()

            kc = self.comm.allreduce(kc, op=MPI.SUM)

            iter_timing.append(('communicating kc vectors', tm.time() - t0))
            t0 = tm.time()

            # compute sigma vectors

            sigma_mat = np.zeros(trial_mat.shape)

            for vecind in range(trial_mat.shape[1]):
                rjb = trial_mat[:, vecind].reshape(nocc, nvir)
                sigma = np.zeros(rjb.shape)

                # 0th-order contribution

                if self.rank == mpi_master():
                    sigma += np.matmul(rjb, fab.T) - np.matmul(fij, rjb)

                # 1st-order contribution

                for ind, (i, j) in enumerate(mo_indices['oo']):
                    # [ 2 (ia|jb) - (ij|ab) ] R_jb
                    vv_1 = mo_integrals['chem_ovov_K'][ind]
                    vv_2 = mo_integrals['chem_oovv_J'][ind]
                    ab = 2.0 * vv_1 - vv_2
                    sigma[i, :] += np.dot(ab, rjb[j, :])

                # 2nd-order contributions

                # term A

                sigma += 0.5 * np.matmul(rjb, xA_ab.T)

                # term B

                sigma += 0.5 * np.matmul(xB_ij, rjb)

                # term C

                kc1 = kc[vecind, 0, :, :]
                kc2 = kc[vecind, 1, :, :]

                for ind, (i, k) in enumerate(mo_indices['oo']):
                    de = e_vv - eocc[i] - eocc[k]
                    vv = mo_integrals['chem_ovov_K'][ind]
                    # -0.5 [ 2 (ia|kc) - (ic|ka) ] R_kc
                    ac_ca = -vv + 0.5 * vv.T
                    sigma[i, :] += np.dot(ac_ca / de, kc1[k, :])
                    sigma[i, :] += np.dot(ac_ca, kc2[k, :])

                sigma_mat[:, vecind] = sigma.reshape(nocc * nvir)[:]

            iter_timing.append(('computing sigma vectors', tm.time() - t0))
            t0 = tm.time()

            sigma_mat = self.comm.reduce(sigma_mat,
                                         op=MPI.SUM,
                                         root=mpi_master())

            iter_timing.append(('communicating sigma vectors', tm.time() - t0))
            t0 = tm.time()

            if self.rank == mpi_master():
                self.solver.add_iteration_data(sigma_mat, trial_mat, iteration)
                trial_mat = self.solver.compute(diag_mat)
            else:
                trial_mat = None

            iter_timing.append(('computing new trial vectors', tm.time() - t0))
            t0 = tm.time()

            trial_mat = self.comm.bcast(trial_mat, root=mpi_master())

            iter_timing.append(
                ('communicating new trial vectors', tm.time() - t0))

            if self.rank == mpi_master():
                self.print_iter_data(iteration, iter_start_time, iter_timing)

            # check convergence

            self.check_convergence(iteration)

            if self.is_converged:
                break

        # print converged excited states

        if self.rank == mpi_master():
            reigs, rnorms = self.solver.get_eigenvalues()
            self.print_convergence(start_time, reigs)
            return {'eigenvalues': reigs}
        else:
            return {}

    def print_header(self):
        """
        Prints ADC(2) driver setup header to output stream.
        """

        self.ostream.print_blank()
        self.ostream.print_header("ADC(2) Driver Setup")
        self.ostream.print_header(21 * "=")
        self.ostream.print_blank()

        str_width = 60

        cur_str = "Number Of Excited States  : " + str(self.nstates)
        self.ostream.print_header(cur_str.ljust(str_width))

        cur_str = "Max. Number Of Iterations : " + str(self.max_iter)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = "Convergence Threshold     : " + \
            "{:.1e}".format(self.conv_thresh)
        self.ostream.print_header(cur_str.ljust(str_width))

        cur_str = "ERI screening scheme      : " + get_qq_type(self.qq_type)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = "ERI Screening Threshold   : " + \
            "{:.1e}".format(self.eri_thresh)
        self.ostream.print_header(cur_str.ljust(str_width))
        self.ostream.print_blank()

        self.ostream.flush()

    def gen_trial_vectors(self, mol_orbs, molecule):
        """Generates set of trial vectors.

        :param mol_orbs:
            The molecular orbitals.
        :param molecule:
            The molecule.
        :return:
            The set of trial vectors.
        """

        if self.rank == mpi_master():

            nocc = molecule.number_of_electrons() // 2
            norb = mol_orbs.number_mos()

            zvec = ExcitationVector(szblock.aa, 0, nocc, nocc, norb, True)
            exci_list = zvec.small_energy_identifiers(mol_orbs, self.nstates)

            diag_mat = zvec.diagonal_to_numpy(mol_orbs)
            reigs = []

            trial_vecs = []
            for exci_ind in exci_list:
                trial_vecs.append(
                    ExcitationVector(szblock.aa, 0, nocc, nocc, norb, True))
                trial_vecs[-1].set_zcoefficient(1.0, exci_ind)
                reigs.append(diag_mat[exci_ind][0])

            return (diag_mat, trial_vecs, sorted(reigs))

        return (None, [], [])

    def check_convergence(self, iteration):
        """
        Checks convergence of excitation energies and set convergence flag.

        :param iteration:
            The current excited states solver iteration.
        """

        self.cur_iter = iteration

        if self.rank == mpi_master():
            self.is_converged = self.solver.check_convergence(self.conv_thresh)
        else:
            self.is_converged = False

        self.is_converged = self.comm.bcast(self.is_converged,
                                            root=mpi_master())

    def print_iter_data(self, iteration, iter_start_time, iter_timing):
        """Prints excited states solver iteration data to output stream.

        :param iteration:
            The current excited states solver iteration.
        :param iter_start_time:
            The starting time of the iteration.
        :param iter_timing:
            A list of tuple containing individual timings.
        """

        # iteration header

        exec_str = " *** Iteration: " + (str(iteration + 1)).rjust(3)
        exec_str += " * Reduced Space: "
        exec_str += (str(self.solver.reduced_space_size())).rjust(4)
        rmax, rmin = self.solver.max_min_residual_norms()
        exec_str += " * Residues (Max,Min): {:.2e} and {:.2e}".format(
            rmax, rmin)
        self.ostream.print_header(exec_str)
        self.ostream.print_blank()

        # excited states information

        reigs, rnorms = self.solver.get_eigenvalues()
        for i in range(reigs.shape[0]):
            exec_str = "State {:2d}: {:5.8f} ".format(i + 1, reigs[i])
            exec_str += "au Residual Norm: {:3.8f}".format(rnorms[i])
            self.ostream.print_header(exec_str.ljust(84))
        self.ostream.print_blank()

        # timing

        self.ostream.print_info(
            'Time spent in this iteration: {:.2f} sec.'.format(tm.time() -
                                                               iter_start_time))
        for t in iter_timing:
            self.ostream.print_info('  {:<35s} :   {:.2f} sec'.format(*t))
        self.ostream.print_blank()
        self.ostream.flush()

    def print_convergence(self, start_time, reigs):
        """
        Prints convergence and excited state information.

        :param start_time:
            The start time of calculation.
        :param reigs:
            The excitaion energies.
        """

        valstr = "*** {:d} excited states ".format(self.nstates)
        if self.is_converged:
            valstr += "converged"
        else:
            valstr += "not converged"
        valstr += " in {:d} iterations. ".format(self.cur_iter + 1)
        valstr += "Time: {:.2f}".format(tm.time() - start_time) + " sec."
        self.ostream.print_header(valstr.ljust(92))
        self.ostream.print_blank()
        self.ostream.print_blank()
        if self.is_converged:
            valstr = "ADC(2) excited states [SS block only]"
            self.ostream.print_header(valstr.ljust(92))
            self.ostream.print_header(('-' * len(valstr)).ljust(92))
            for s, e in enumerate(reigs):
                valstr = 'Excited State {:>5s}: '.format('S' + str(s + 1))
                valstr += '{:15.8f} a.u. '.format(e)
                valstr += '{:12.5f} eV'.format(e * hartree_in_ev())
                self.ostream.print_header(valstr.ljust(92))
            self.ostream.print_blank()
            self.ostream.print_blank()
        self.ostream.flush()
