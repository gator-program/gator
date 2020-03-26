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
    """Implements ADC(2) computation driver.

    Implements ADC(2) computation schheme for Hatree-Fock reference.

    Attributes
    ----------
    nstates
        The number of excited states determined by driver.
    triplet
        The triplet excited states flag.
    eri_thresh
        The electron repulsion integrals screening threshold.
    conv_thresh
        The excited states convergence threshold.
    max_iter
        The maximum number of excited states driver iterations.
    cur_iter
        The current number of excited states driver iterations.
    solver
        The eigenvalues solver.
    is_converged
        The flag for excited states convergence.
    rank
        The rank of MPI process.
    nodes
        The number of MPI processes.
    """

    def __init__(self, comm, ostream):
        """Initializes ADC(2) computation driver.

        Initializes ADC(2) computation drived to default setup.

        Parameters
        ----------
        comm
            The MPI communicator.
        ostream
            The output stream.
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

    def update_settings(self, settings):
        """Updates settings in ADC(2) driver.

        Updates settings in ADC(2) computation driver.

        Parameters
        ----------
        settings
            The settings for the driver.
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
        if 'qq_type' in settings:
            self.qq_type = settings['qq_type'].upper()

    def compute(self, molecule, basis, scf_tensors):
        """Performs ADC(2) calculation.

        Performs ADC(2) calculation using molecular data.

        Parameters
        ----------
        molecule
            The molecule.
        basis
            The AO basis set.
        scf_tensors
            The tensors from converged SCF wavefunction.
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
        eij = eocc.reshape(-1, 1) + eocc
        eab = evir.reshape(-1, 1) + evir

        if self.rank == mpi_master():
            self.print_header()

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
        moints_blocks = moints_drv.compute(molecule, basis, scf_tensors)

        # precompute ab and ij matrices for 2nd-order contribution
        # (terms A and B)

        # [ +2<kl|ac> -1<kl|ca> ] <kl|bc>
        m2_ab = np.zeros((nvir, nvir))
        for ind, (i, j) in enumerate(moints_blocks['oo_indices']):
            eabij = eab - eocc[i] - eocc[j]
            local_ab = moints_blocks['oovv'][ind, :].reshape(nvir, nvir)
            m2_ab += np.matmul(local_ab / eabij, 2.0 * local_ab.T - local_ab)
            m2_ab += np.matmul(local_ab, (2.0 * local_ab.T - local_ab) / eabij)

        # [ +2<cd|ik> -1<cd|ki> ] <cd|jk>
        m2_ij = np.zeros((nocc, nocc))
        for ind, (a, b) in enumerate(moints_blocks['vv_indices']):
            eabij = evir[a] + evir[b] - eij
            local_ij = moints_blocks['vvoo'][ind, :].reshape(nocc, nocc)
            m2_ij += np.matmul(local_ij / eabij, 2.0 * local_ij.T - local_ij)
            m2_ij += np.matmul(local_ij, (2.0 * local_ij.T - local_ij) / eabij)

        # start iterations

        for iteration in range(self.max_iter):

            # precompute kc vectors for 2nd-order contribution (term C)

            kc = np.zeros((trial_mat.shape[1], 2, nocc, nvir))

            for vecind in range(trial_mat.shape[1]):
                rjb = trial_mat[:, vecind].reshape(nocc, nvir)
                kc1 = kc[vecind, 0, :, :]
                kc2 = kc[vecind, 1, :, :]

                for ind, (k, j) in enumerate(moints_blocks['oo_indices']):
                    de = eab - eocc[k] - eocc[j]
                    vv = moints_blocks['oovv'][ind, :].reshape(nvir, nvir)
                    # [ +2<kj|cb> - <kj|bc> ] R_jb
                    # 'kjcb,jb->kc'
                    # 'kjbc,jb->kc'
                    cb_bc = 2.0 * vv - vv.T
                    kc1[k, :] += (np.matmul(cb_bc, rjb.T))[:, j]
                    kc2[k, :] += (np.matmul(cb_bc / de, rjb.T))[:, j]

            kc = self.comm.allreduce(kc, op=MPI.SUM)

            # compute sigma vectors

            sigma_mat = np.zeros(trial_mat.shape)

            for vecind in range(trial_mat.shape[1]):
                rjb = trial_mat[:, vecind].reshape(nocc, nvir)
                sigma = np.zeros(rjb.shape)

                # 0th-order contribution

                if self.rank == mpi_master():
                    sigma += np.matmul(rjb, fab.T) - np.matmul(fij, rjb)

                # 1st-order contribution

                for ind, (i, j) in enumerate(moints_blocks['oo_indices']):
                    # 'ijab,jb->ia'
                    ab = moints_blocks['oovv'][ind, :].reshape(nvir, nvir)
                    # 'ibja,jb->ia'
                    ba = moints_blocks['ovov'][ind, :].reshape(nvir, nvir)
                    ab_ba = 2.0 * ab - ba.T
                    sigma[i, :] += (np.matmul(ab_ba, rjb.T))[:, j]

                # 2nd-order contributions

                # A: 0.5 [ +2<kl|ac> -1<kl|ca> ] <kl|bc> R_jb

                sigma += 0.5 * np.matmul(rjb, m2_ab.T)

                # B: 0.5 [ +2<cd|ik> -1<cd|ki> ] <cd|jk> R_jb

                sigma += 0.5 * np.matmul(m2_ij, rjb)

                # C: -0.5 [ +2<ac|ik> -1<ac|ki> ] [ +2<jk|bc> - <jk|cb> ] R_jb

                kc1 = kc[vecind, 0, :, :]
                kc2 = kc[vecind, 1, :, :]

                for ind, (i, k) in enumerate(moints_blocks['oo_indices']):
                    de = eab - eocc[i] - eocc[k]
                    vv = moints_blocks['oovv'][ind, :].reshape(nvir, nvir)
                    # -0.5 [ +2<ac|ik> -1<ac|ki> ] R_kc
                    # 'ikac,kc->ia'
                    # 'ikca,kc->ia'
                    ac_ca = -0.5 * (2.0 * vv - vv.T)
                    sigma[i, :] += (np.matmul(ac_ca / de, kc1.T))[:, k]
                    sigma[i, :] += (np.matmul(ac_ca, kc2.T))[:, k]

                sigma_mat[:, vecind] = sigma.reshape(nocc * nvir)[:]

            sigma_mat = self.comm.reduce(sigma_mat,
                                         op=MPI.SUM,
                                         root=mpi_master())

            if self.rank == mpi_master():
                self.solver.add_iteration_data(sigma_mat, trial_mat, iteration)
                trial_mat = self.solver.compute(diag_mat)
            else:
                trial_mat = None
            trial_mat = self.comm.bcast(trial_mat, root=mpi_master())

            if self.rank == mpi_master():
                self.print_iter_data(iteration)

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
        """Prints ADC(2) driver setup header to output stream"""

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
        """Generates set of TDA trial vectors.

        Generates set of TDA trial vectors for given number of excited states
        by selecting primitive excitations wirh lowest approximate energies
        E_ai = e_a-e_i.

        Parameters
        ----------
        mol_orbs
            The molecular orbitals.
        molecule
            The molecule.
        Returns
        -------
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
        """Checks convergence of excitation energies and set convergence flag.

        Checks convergence of excitation energies and set convergence flag on
        all processes within MPI communicator.

        Parameters
        ----------
        iter
            The current excited states solver iteration.
        """

        self.cur_iter = iteration

        if self.rank == mpi_master():
            self.is_converged = self.solver.check_convergence(self.conv_thresh)
        else:
            self.is_converged = False

        self.is_converged = self.comm.bcast(self.is_converged,
                                            root=mpi_master())

    def print_iter_data(self, iter):
        """Prints excited states solver iteration data to output stream.

        Prints excited states solver iteration data to output stream.

        Parameters
        ----------
        iter
            The current excited states solver iteration.
        """

        # iteration header

        exec_str = " *** Iteration: " + (str(iter + 1)).rjust(3)
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

        # flush output stream
        self.ostream.print_blank()
        self.ostream.flush()

    def print_convergence(self, start_time, reigs):
        """Prints excited states information to output stream.

        Prints excited states information to output stream.

        Parameters
        ----------
        start_time
            The start time of SCF calculation.
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
            valstr = "ADC excited states"
            self.ostream.print_header(valstr.ljust(92))
            self.ostream.print_header(('-' * len(valstr)).ljust(92))
            for s, e in enumerate(reigs):
                valstr = 'Excited State {:>5s}: '.format('S' + str(s + 1))
                valstr += '{:15.8f} a.u. '.format(e)
                valstr += '{:12.5f} eV'.format(e * hartree_in_ev())
                self.ostream.print_header(valstr.ljust(92))
            self.ostream.print_blank()
