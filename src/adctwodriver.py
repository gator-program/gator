from mpi4py import MPI
import numpy as np
import time as tm

from veloxchem import BlockDavidsonSolver
from veloxchem import mpi_master
from veloxchem import hartree_in_ev
from veloxchem import get_qq_type

from .mointsdriver import MOIntegralsDriver
from .adconedriver import AdcOneDriver


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

        start_time = tm.time()

        # orbitals and orbital energies

        if self.rank == mpi_master():
            mo = scf_tensors['C']
            ea = scf_tensors['E']
        else:
            mo = None
            ea = None
        mo = self.comm.bcast(mo, root=mpi_master())
        ea = self.comm.bcast(ea, root=mpi_master())

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

        if self.rank == mpi_master():
            fa = scf_tensors['F'][0]
            fmo = np.matmul(mo.T, np.matmul(fa, mo))
            fab = fmo[nocc:, nocc:]
            fij = fmo[:nocc, :nocc]
        else:
            fab = None
            fij = None

        # MO integrals

        moints_drv = MOIntegralsDriver(self.comm, self.ostream)
        moints_drv.update_settings({
            'qq_type': self.qq_type,
            'eri_thresh': self.eri_thresh
        })
        mo_indices, mo_integrals = moints_drv.compute(molecule, basis,
                                                      scf_tensors)

        # precompute xA_ab and xB_ij matrices for 2nd-order contribution

        xA_ab, xB_ij = self.compute_xA_xB(epsilon, mo_indices, mo_integrals)

        auxiliary_matrices = {
            'fab': fab,
            'fij': fij,
            'xA_ab': xA_ab,
            'xB_ij': xB_ij,
        }

        # get initial guess from ADC(1)

        adc_one_drv = AdcOneDriver(self.comm, self.ostream)
        adc_one_drv.update_settings({
            'nstates': self.nstates,
            'eri_thresh': self.eri_thresh,
            'qq_type': self.qq_type
        })
        adc_one_results = adc_one_drv.compute(molecule, basis, scf_tensors,
                                              mo_indices, mo_integrals)

        # set up trial excitation vectors

        diag_mat = e_ov.copy().reshape(nocc * nvir, 1)

        if self.rank == mpi_master():
            initial_trials = adc_one_results['eigenvectors'].copy()
            initial_reigs = adc_one_results['eigenvalues'].copy()
        else:
            initial_trials = None
            initial_reigs = None
        initial_trials = self.comm.bcast(initial_trials, root=mpi_master())
        initial_reigs = self.comm.bcast(initial_reigs, root=mpi_master())

        # start ADC(2)

        if self.rank == mpi_master():
            self.print_header()
            self.ostream.print_info(
                'Number of occupied orbitals: {:d}'.format(nocc))
            self.ostream.print_info(
                'Number of virtual orbitals: {:d}'.format(nvir))
            self.ostream.print_blank()

        root_converged = [False] * self.nstates
        converged_eigs = [None] * self.nstates
        s_components_2 = [None] * self.nstates

        for root in range(self.nstates):

            trial_mat = initial_trials.copy()
            omega = initial_reigs[root]

            solver = BlockDavidsonSolver()

            self.ostream.print_header(
                '*** Starting iterations for solving root {}'.format(
                    root + 1).ljust(92))
            self.ostream.print_blank()

            for iteration in range(self.max_iter):

                omega_eigs = np.full(trial_mat.shape[1], omega)

                sigma_mat = self.compute_sigma(trial_mat, omega_eigs, epsilon,
                                               auxiliary_matrices, mo_indices,
                                               mo_integrals)

                if self.rank == mpi_master():
                    solver.add_iteration_data(sigma_mat, trial_mat, iteration)
                    trial_mat = solver.compute(diag_mat)
                    ritz_vecs = solver.ritz_vectors.copy()
                else:
                    trial_mat = None
                    ritz_vecs = None
                trial_mat = self.comm.bcast(trial_mat, root=mpi_master())
                ritz_vecs = self.comm.bcast(ritz_vecs, root=mpi_master())

                omega_eigs = np.full(ritz_vecs.shape[1], omega)

                sigma_vecs_from_ritz = self.compute_sigma(
                    ritz_vecs, omega_eigs, epsilon, auxiliary_matrices,
                    mo_indices, mo_integrals)

                d_sigma_vecs_from_ritz = self.compute_d_sigma(
                    ritz_vecs, omega_eigs, epsilon, mo_indices, mo_integrals)

                if self.rank == mpi_master():
                    tvec = ritz_vecs[:, root]
                    svec = sigma_vecs_from_ritz[:, root]
                    dsvec = d_sigma_vecs_from_ritz[:, root]
                    s_comp_square = 1.0 / (1.0 + np.dot(tvec, dsvec))
                    eig_from_ritz = np.dot(tvec, svec)
                    residual_norm = np.linalg.norm(svec - omega * tvec)
                    eig_incr = (eig_from_ritz - omega) * s_comp_square
                else:
                    s_comp_square = None
                    residual_norm = None
                    eig_incr = None
                eig_incr = self.comm.bcast(eig_incr, root=mpi_master())
                residual_norm = self.comm.bcast(residual_norm,
                                                root=mpi_master())

                self.print_iter_data(iteration, omega, eig_incr, residual_norm)

                if (abs(eig_incr) < self.conv_thresh**2 and
                        residual_norm < self.conv_thresh):
                    root_converged[root] = True
                    converged_eigs[root] = omega
                    s_components_2[root] = s_comp_square
                    self.ostream.print_blank()
                    self.ostream.print_header(
                        '    Root {} is converged: {:.8f} au'.format(
                            root + 1, omega).ljust(92))
                    self.ostream.print_blank()
                    self.ostream.flush()
                    break
                else:
                    omega += eig_incr
                    if self.rank == mpi_master():
                        if solver.trial_matrices.shape[1] > self.nstates:
                            solver.trial_matrices = ritz_vecs.copy()
                            solver.sigma_matrices = sigma_vecs_from_ritz.copy()

            if not root_converged[root]:
                self.ostream.print_blank()
                self.ostream.print_header(
                    '    Root {} is NOT converged.'.format(root + 1).ljust(92))
                self.ostream.print_blank()
                self.ostream.flush()

        # check convergence
        self.is_converged = True
        for root in range(self.nstates):
            if not root_converged[root]:
                self.is_converged = False

        # print converged excited states
        if self.rank == mpi_master():
            self.print_convergence(start_time, converged_eigs, s_components_2)
            return {
                'eigenvalues': np.array(converged_eigs),
                's_components_2': np.array(s_components_2),
            }
        else:
            return {}

    def compute_xA_xB(self, epsilon, mo_indices, mo_integrals):

        eocc = epsilon['o']
        evir = epsilon['v']
        e_oo = epsilon['oo']
        e_vv = epsilon['vv']

        nocc = eocc.shape[0]
        nvir = evir.shape[0]

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

        return xA_ab, xB_ij

    def compute_sigma(self, trial_mat, reigs, epsilon, auxiliary_matrices,
                      mo_indices, mo_integrals):

        # orbital energies

        eocc = epsilon['o']
        evir = epsilon['v']
        e_oo = epsilon['oo']
        e_vv = epsilon['vv']
        e_ov = epsilon['ov']

        nocc = eocc.shape[0]
        nvir = evir.shape[0]

        fab = auxiliary_matrices['fab']
        fij = auxiliary_matrices['fij']
        xA_ab = auxiliary_matrices['xA_ab']
        xB_ij = auxiliary_matrices['xB_ij']

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

            # single-double coupling term

            # [ 2 (ki|ld) - (kd|li) ] δac (kj|ld) δbc R_jb / (w-e_klcd)
            # [ 2 (ki|lb) - (kb|li) ] δac (ka|lj) δbd R_jb / (w-e_klcd)

            for ind, (k, l) in enumerate(mo_indices['oo']):
                omega_de = reigs[vecind] - (e_vv - eocc[k] - eocc[l])
                ov_kl = mo_integrals['chem_ooov_K'][ind]
                ind_lk = ind if k == l else (ind + 1 if k < l else ind - 1)
                ov_lk = mo_integrals['chem_ooov_K'][ind_lk]

                # [ 2 (ki|ld) - (kd|li) ] δac (kj|ld) δbc R_jb / (w-e_klcd)

                # δac (kj|ld) δbc R_jb / (w-e_klcd)
                jd = ov_kl
                ja = rjb
                da = np.matmul(jd.T, ja) / omega_de
                # [ 2 (ki|ld) - (li|kd) ] R_ad
                id_id = 2.0 * ov_kl - ov_lk
                sigma += np.matmul(id_id, da)

                # [ 2 (ki|lb) - (kb|li) ] δac (ka|lj) δbd R_jb / (w-e_klcd)

                # δac (lj|ka) δbd R_jb / (w-e_klcd)
                ja = ov_lk
                ba = np.matmul(rjb.T, ja) / omega_de
                # [ 2 (ki|lb) - (li|kb) ] R_ab
                ib_ib = 2.0 * ov_kl - ov_lk
                sigma += np.matmul(ib_ib, ba)

            # [ 2 (ac|ld) - (ad|lc) ] δik (bc|ld) δjk R_jb / (w-e_klcd)
            # [ 2 (ac|jd) - (ad|jc) ] δik (bd|ic) δjl R_jb / (w-e_klcd)

            for ind, (c, d) in enumerate(mo_indices['vv']):
                omega_de = reigs[vecind] - (evir[c] + evir[d] - e_oo)
                ov_cd = mo_integrals['chem_vovv_K'][ind]
                ind_dc = ind if c == d else (ind + 1 if c < d else ind - 1)
                ov_dc = mo_integrals['chem_vovv_K'][ind_dc]

                # [ 2 (ac|ld) - (ad|lc) ] δik (bc|ld) δjk R_jb / (w-e_klcd)

                # δik (dl|cb) δjk R_jb / (w-e_klcd)
                lb = ov_dc
                ib = rjb
                il = np.matmul(ib, lb.T) / omega_de
                # [ 2 (dl|ca) - (cl|da) ] R_kl
                la_la = 2.0 * ov_dc - ov_cd
                sigma += np.matmul(il, la_la)

                # [ 2 (ac|jd) - (ad|jc) ] δik (bd|ic) δjl R_jb / (w-e_klcd)

                # δik (ci|db) δjl R_jb / (w-e_klcd)
                ib = ov_cd
                ij = np.matmul(ib, rjb.T) / omega_de
                # [ 2 (dj|ca) - (cj|da) ] R_ij
                ja_ja = 2.0 * ov_dc - ov_cd
                sigma += np.matmul(ij, ja_ja)

            # [ -2 (ji|ld) + (jd|li) ] δac (ba|ld) δjk R_jb / (w-e_klcd)
            # [ -2 (jd|li) + (ji|ld) ] δac (bd|la) δjk R_jb / (w-e_klcd)

            # [ -2 (ab|ld) + (ad|lb) ] δik (ij|ld) δbc R_jb / (w-e_klcd)
            # [ -2 (ad|lb) + (ab|ld) ] δik (id|lj) δbc R_jb / (w-e_klcd)

            for ind, (l, d) in enumerate(mo_indices['ov']):
                omega_de = reigs[vecind] - (e_ov + evir[d] - eocc[l])
                oo_J = mo_integrals['chem_ovoo_J'][ind]
                oo_K = mo_integrals['chem_oovo_K'][ind]
                vv_J = mo_integrals['chem_ovvv_J'][ind]
                vv_K = mo_integrals['chem_ovvv_K'][ind]

                # [ -2 (ji|ld) + (jd|li) ] δac (ba|ld) δjk R_jb / (w-e_klcd)

                # δac (ba|ld) δjk R_jb / (w-e_klcd)
                ba = vv_J
                ja = np.matmul(rjb, ba) / omega_de
                # [ -2 (ij|ld) + (il|jd) ] R_ja
                ij_ij = -2.0 * oo_J + oo_K
                sigma += np.matmul(ij_ij, ja)

                # [ -2 (jd|li) + (ji|ld) ] δac (bd|la) δjk R_jb / (w-e_klcd)

                # δac (al|bd) δjk R_jb / (w-e_klcd)
                ab = vv_K
                ja = np.matmul(rjb, ab.T) / omega_de
                # [ -2 (il|jd) + (ij|ld) ] R_ja
                ij_ij = -2.0 * oo_K + oo_J
                sigma += np.matmul(ij_ij, ja)

                # [ -2 (ab|ld) + (ad|lb) ] δik (ij|ld) δbc R_jb / (w-e_klcd)

                # δik (ij|ld) δbc R_jb / (w-e_klcd)
                ij = oo_J
                ib = np.matmul(ij, rjb) / omega_de
                # [ -2 (ba|ld) + (bl|ad) ] R_ib
                ba_ba = -2.0 * vv_J + vv_K
                sigma += np.matmul(ib, ba_ba)

                # [ -2 (ad|lb) + (ab|ld) ] δik (id|lj) δbc R_jb / (w-e_klcd)

                # δik (jl|id) δbc R_jb / (w-e_klcd)
                ji = oo_K
                ib = np.matmul(ji.T, rjb) / omega_de
                # [ -2 (bl|ad) + (ba|ld) ] R_ib
                ba_ba = -2.0 * vv_K + vv_J
                sigma += np.matmul(ib, ba_ba)

            sigma_mat[:, vecind] = sigma.reshape(nocc * nvir)[:]

        sigma_mat = self.comm.reduce(sigma_mat, op=MPI.SUM, root=mpi_master())

        return sigma_mat

    def compute_d_sigma(self, trial_mat, reigs, epsilon, mo_indices,
                        mo_integrals):

        # orbital energies

        eocc = epsilon['o']
        evir = epsilon['v']
        e_oo = epsilon['oo']
        e_vv = epsilon['vv']
        e_ov = epsilon['ov']

        nocc = eocc.shape[0]
        nvir = evir.shape[0]

        # compute d_sigma vectors

        d_sigma_mat = np.zeros(trial_mat.shape)

        for vecind in range(trial_mat.shape[1]):
            rjb = trial_mat[:, vecind].reshape(nocc, nvir)
            d_sigma = np.zeros(rjb.shape)

            # single-double coupling term

            # [ 2 (ki|ld) - (kd|li) ] δac (kj|ld) δbc R_jb / (w-e_klcd)
            # [ 2 (ki|lb) - (kb|li) ] δac (ka|lj) δbd R_jb / (w-e_klcd)

            for ind, (k, l) in enumerate(mo_indices['oo']):
                omega_de = reigs[vecind] - (e_vv - eocc[k] - eocc[l])
                ov_kl = mo_integrals['chem_ooov_K'][ind]
                ind_lk = ind if k == l else (ind + 1 if k < l else ind - 1)
                ov_lk = mo_integrals['chem_ooov_K'][ind_lk]

                # [ 2 (ki|ld) - (kd|li) ] δac (kj|ld) δbc R_jb / (w-e_klcd)

                # δac (kj|ld) δbc R_jb / (w-e_klcd)
                jd = ov_kl
                ja = rjb
                da = np.matmul(jd.T, ja) / omega_de**2
                # [ 2 (ki|ld) - (li|kd) ] R_ad
                id_id = 2.0 * ov_kl - ov_lk
                d_sigma += np.matmul(id_id, da)

                # [ 2 (ki|lb) - (kb|li) ] δac (ka|lj) δbd R_jb / (w-e_klcd)

                # δac (lj|ka) δbd R_jb / (w-e_klcd)
                ja = ov_lk
                ba = np.matmul(rjb.T, ja) / omega_de**2
                # [ 2 (ki|lb) - (li|kb) ] R_ab
                ib_ib = 2.0 * ov_kl - ov_lk
                d_sigma += np.matmul(ib_ib, ba)

            # [ 2 (ac|ld) - (ad|lc) ] δik (bc|ld) δjk R_jb / (w-e_klcd)
            # [ 2 (ac|jd) - (ad|jc) ] δik (bd|ic) δjl R_jb / (w-e_klcd)

            for ind, (c, d) in enumerate(mo_indices['vv']):
                omega_de = reigs[vecind] - (evir[c] + evir[d] - e_oo)
                ov_cd = mo_integrals['chem_vovv_K'][ind]
                ind_dc = ind if c == d else (ind + 1 if c < d else ind - 1)
                ov_dc = mo_integrals['chem_vovv_K'][ind_dc]

                # [ 2 (ac|ld) - (ad|lc) ] δik (bc|ld) δjk R_jb / (w-e_klcd)

                # δik (dl|cb) δjk R_jb / (w-e_klcd)
                lb = ov_dc
                ib = rjb
                il = np.matmul(ib, lb.T) / omega_de**2
                # [ 2 (dl|ca) - (cl|da) ] R_kl
                la_la = 2.0 * ov_dc - ov_cd
                d_sigma += np.matmul(il, la_la)

                # [ 2 (ac|jd) - (ad|jc) ] δik (bd|ic) δjl R_jb / (w-e_klcd)

                # δik (ci|db) δjl R_jb / (w-e_klcd)
                ib = ov_cd
                ij = np.matmul(ib, rjb.T) / omega_de**2
                # [ 2 (dj|ca) - (cj|da) ] R_ij
                ja_ja = 2.0 * ov_dc - ov_cd
                d_sigma += np.matmul(ij, ja_ja)

            # [ -2 (ji|ld) + (jd|li) ] δac (ba|ld) δjk R_jb / (w-e_klcd)
            # [ -2 (jd|li) + (ji|ld) ] δac (bd|la) δjk R_jb / (w-e_klcd)

            # [ -2 (ab|ld) + (ad|lb) ] δik (ij|ld) δbc R_jb / (w-e_klcd)
            # [ -2 (ad|lb) + (ab|ld) ] δik (id|lj) δbc R_jb / (w-e_klcd)

            for ind, (l, d) in enumerate(mo_indices['ov']):
                omega_de = reigs[vecind] - (e_ov + evir[d] - eocc[l])
                oo_J = mo_integrals['chem_ovoo_J'][ind]
                oo_K = mo_integrals['chem_oovo_K'][ind]
                vv_J = mo_integrals['chem_ovvv_J'][ind]
                vv_K = mo_integrals['chem_ovvv_K'][ind]

                # [ -2 (ji|ld) + (jd|li) ] δac (ba|ld) δjk R_jb / (w-e_klcd)

                # δac (ba|ld) δjk R_jb / (w-e_klcd)
                ba = vv_J
                ja = np.matmul(rjb, ba) / omega_de**2
                # [ -2 (ij|ld) + (il|jd) ] R_ja
                ij_ij = -2.0 * oo_J + oo_K
                d_sigma += np.matmul(ij_ij, ja)

                # [ -2 (jd|li) + (ji|ld) ] δac (bd|la) δjk R_jb / (w-e_klcd)

                # δac (al|bd) δjk R_jb / (w-e_klcd)
                ab = vv_K
                ja = np.matmul(rjb, ab.T) / omega_de**2
                # [ -2 (il|jd) + (ij|ld) ] R_ja
                ij_ij = -2.0 * oo_K + oo_J
                d_sigma += np.matmul(ij_ij, ja)

                # [ -2 (ab|ld) + (ad|lb) ] δik (ij|ld) δbc R_jb / (w-e_klcd)

                # δik (ij|ld) δbc R_jb / (w-e_klcd)
                ij = oo_J
                ib = np.matmul(ij, rjb) / omega_de**2
                # [ -2 (ba|ld) + (bl|ad) ] R_ib
                ba_ba = -2.0 * vv_J + vv_K
                d_sigma += np.matmul(ib, ba_ba)

                # [ -2 (ad|lb) + (ab|ld) ] δik (id|lj) δbc R_jb / (w-e_klcd)

                # δik (jl|id) δbc R_jb / (w-e_klcd)
                ji = oo_K
                ib = np.matmul(ji.T, rjb) / omega_de**2
                # [ -2 (bl|ad) + (ba|ld) ] R_ib
                ba_ba = -2.0 * vv_K + vv_J
                d_sigma += np.matmul(ib, ba_ba)

            d_sigma_mat[:, vecind] = d_sigma.reshape(nocc * nvir)[:]

        d_sigma_mat = self.comm.reduce(d_sigma_mat,
                                       op=MPI.SUM,
                                       root=mpi_master())

        return d_sigma_mat

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

    def print_iter_data(self, iteration, omega, eig_incr, rnorm):
        """Prints excited states solver iteration data to output stream.

        :param iteration:
            The current excited states solver iteration.
        """

        exec_str = '    Iteration: {:d}'.format(iteration + 1)
        exec_str += ' * Eigenvalue: {:.8f}'.format(omega)
        exec_str += ' * Residual Norm: {:.2e}'.format(rnorm)
        self.ostream.print_header(exec_str.ljust(92))
        self.ostream.flush()

    def print_convergence(self, start_time, converged_eigs, s_components_2):
        """
        Prints convergence and excited state information.

        :param start_time:
            The start time of calculation.
        """

        valstr = '*** {:d} excited states '.format(self.nstates)
        if self.is_converged:
            valstr += 'converged.'
        else:
            valstr += 'NOT converged.'
        valstr += ' Time: {:.2f}'.format(tm.time() - start_time) + ' sec.'
        self.ostream.print_header(valstr.ljust(92))
        self.ostream.print_blank()
        self.ostream.print_blank()
        if self.is_converged:
            valstr = 'ADC(2) excited states'
            self.ostream.print_header(valstr.ljust(92))
            self.ostream.print_header(('-' * len(valstr)).ljust(92))
            for s, e in enumerate(converged_eigs):
                valstr = 'Excited State {:>5s}: '.format('S' + str(s + 1))
                valstr += '{:15.8f} a.u. '.format(e)
                valstr += '{:12.5f} eV'.format(e * hartree_in_ev())
                valstr += '    |v1|^2={:.4f}'.format(s_components_2[s])
                self.ostream.print_header(valstr.ljust(92))
            self.ostream.print_blank()
            self.ostream.print_blank()
        self.ostream.flush()
