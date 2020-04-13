from veloxchem import mpi_master
from veloxchem import molorb
from veloxchem import MolecularOrbitals
from veloxchem import MOIntegralsDriver
import numpy as np


class AdcMatrixDriver:
    """
    Implements ADC(2) matrix driver.

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
        Initializes ADC(2) matrix driver.
        """

        # ERI settings
        self.eri_thresh = 1.0e-15
        self.qq_type = 'QQ_DEN'

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

    def update_settings(self, settings, scf_drv=None):
        """
        Updates settings in ADC(2) matrix driver.

        :param settings:
            The settings for the driver.
        :param scf_drv:
            The scf driver.
        """

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
        Performs ADC(2) matrix calculation.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param scf_tensors:
            The tensors from converged SCF wavefunction.
        :return:
            The ADC(2) matrix.
        """

        moints_drv = MOIntegralsDriver(self.comm, self.ostream)

        if self.rank == mpi_master():
            mo = scf_tensors['C']
            ea = scf_tensors['E']

            nocc = molecule.number_of_alpha_electrons()
            nvir = mo.shape[1] - nocc

            # energy of spin orbital
            e_occ = list(ea[:nocc]) * 2
            e_vir = list(ea[nocc:]) * 2

            # spin of spin orbital
            s_occ = ['alpha'] * nocc + ['beta'] * nocc
            s_vir = ['alpha'] * nvir + ['beta'] * nvir

            # MO index of spin orbital
            ind_occ = list(range(nocc)) * 2
            ind_vir = list(range(nvir)) * 2

            # number of spin orbitals
            nocc2 = len(e_occ)
            nvir2 = len(e_vir)

            # dimension of single and double block
            s_dim = nocc2 * nvir2
            d_dim = (nocc2 * (nocc2 - 1) // 2) * (nvir2 * (nvir2 - 1) // 2)

            # MO integrals
            mol_orbs = MolecularOrbitals([mo], [ea], molorb.rest)
            phys_oovv = moints_drv.compute_in_mem(molecule, basis, mol_orbs, 'OOVV')
            phys_ovov = moints_drv.compute_in_mem(molecule, basis, mol_orbs, 'OVOV')
            phys_ooov = moints_drv.compute_in_mem(molecule, basis, mol_orbs, 'OOOV')
            phys_ovvv = moints_drv.compute_in_mem(molecule, basis, mol_orbs, 'OVVV')

            #
            # single-single block
            #

            ss_mat = np.zeros((s_dim, s_dim))

            # M^{(0)}_{ia,jb} = (e_a - e_i) \delta_{ab} \delta_{ij}

            for i in range(nocc2):
                for a in range(nvir2):
                    ia = i * nvir2 + a
                    jb = ia
                    ss_mat[ia, jb] += e_vir[a] - e_occ[i]

            # M^{(1)}_{ia,jb} = <aj||ib>
            #                 = <aj|ib> - <aj|bi>
            #                 = <ij|ab> - <ja|ib>

            for i, (ind_i, e_i, s_i) in enumerate(zip(ind_occ, e_occ, s_occ)):
                for a, (ind_a, e_a, s_a) in enumerate(zip(ind_vir, e_vir, s_vir)):
                    ia = i * nvir2 + a

                    for j, (ind_j, e_j, s_j) in enumerate(zip(ind_occ, e_occ, s_occ)):
                        for b, (ind_b, e_b, s_b) in enumerate(zip(ind_vir, e_vir, s_vir)):
                            jb = j * nvir2 + b

                            ijab = 0.0
                            if s_i == s_a and s_j == s_b:
                                ijab = phys_oovv[ind_i, ind_j, ind_a, ind_b]

                            jaib = 0.0
                            if s_j == s_i and s_a == s_b:
                                jaib = phys_ovov[ind_j, ind_a, ind_i, ind_b]

                            ss_mat[ia, jb] += ijab - jaib

            # M^{(2),A}_{ia,jb} = 0.25 \delta_{ij} \sum_{ckl} [
            #                     <ac||kl><kl||bc> / e_klac +
            #                     <ac||kl><kl||bc> / e_klbc ]
            #
            # <ac||kl> = <kl|ac> - <kl|ca>
            # <kl||bc> = <kl|bc> - <kl|cb>

            for i, (ind_i, e_i, s_i) in enumerate(zip(ind_occ, e_occ, s_occ)):
                for a, (ind_a, e_a, s_a) in enumerate(zip(ind_vir, e_vir, s_vir)):
                    ia = i * nvir2 + a

                    # \delta_{ij}
                    j, ind_j, e_j, s_j = i, ind_i, e_i, s_i
                    for b, (ind_b, e_b, s_b) in enumerate(zip(ind_vir, e_vir, s_vir)):
                        jb = j * nvir2 + b

                        # \sum_{ckl}
                        for c, (ind_c, e_c, s_c) in enumerate(zip(ind_vir, e_vir, s_vir)):
                            for k, (ind_k, e_k, s_k) in enumerate(zip(ind_occ, e_occ, s_occ)):
                                for l, (ind_l, e_l, s_l) in enumerate(zip(ind_occ, e_occ, s_occ)):

                                    e_klac = e_a + e_c - e_k - e_l
                                    e_klbc = e_b + e_c - e_k - e_l

                                    klac = 0.0
                                    if s_k == s_a and s_l == s_c:
                                        klac += phys_oovv[ind_k, ind_l, ind_a, ind_c]
                                    if s_k == s_c and s_l == s_a:
                                        klac -= phys_oovv[ind_k, ind_l, ind_c, ind_a]

                                    klbc = 0.0
                                    if s_k == s_b and s_l == s_c:
                                        klbc += phys_oovv[ind_k, ind_l, ind_b, ind_c]
                                    if s_k == s_c and s_l == s_b:
                                        klbc -= phys_oovv[ind_k, ind_l, ind_c, ind_b]

                                    ss_mat[ia, jb] += 0.25 * klac * klbc / e_klac
                                    ss_mat[ia, jb] += 0.25 * klac * klbc / e_klbc

            # M^{(2),B}_{ia,jb} = 0.25 \delta_{ab} \sum_{cdk} [
            #                     <cd||ik><jk||cd> / e_ikcd +
            #                     <cd||ik><jk||cd> / e_jkcd ]
            #
            # <cd||ik> = <ik|cd> - <ik|dc>
            # <jk||cd> = <jk|cd> - <jk|dc>

            for i, (ind_i, e_i, s_i) in enumerate(zip(ind_occ, e_occ, s_occ)):
                for a, (ind_a, e_a, s_a) in enumerate(zip(ind_vir, e_vir, s_vir)):
                    ia = i * nvir2 + a

                    # \delta_{ab}
                    b, ind_b, e_b, s_b = a, ind_a, e_a, s_a
                    for j, (ind_j, e_j, s_j) in enumerate(zip(ind_occ, e_occ, s_occ)):
                        jb = j * nvir2 + b

                        # \sum_{cdk}
                        for c, (ind_c, e_c, s_c) in enumerate(zip(ind_vir, e_vir, s_vir)):
                            for d, (ind_d, e_d, s_d) in enumerate(zip(ind_vir, e_vir, s_vir)):
                                for k, (ind_k, e_k, s_k) in enumerate(zip(ind_occ, e_occ, s_occ)):

                                    e_ikcd = e_d + e_c - e_i - e_k
                                    e_jkcd = e_d + e_c - e_j - e_k

                                    ikcd = 0.0
                                    if s_i == s_c and s_k == s_d:
                                        ikcd += phys_oovv[ind_i, ind_k, ind_c, ind_d]
                                    if s_i == s_d and s_k == s_c:
                                        ikcd -= phys_oovv[ind_i, ind_k, ind_d, ind_c]

                                    jkcd = 0.0
                                    if s_j == s_c and s_k == s_d:
                                        jkcd += phys_oovv[ind_j, ind_k, ind_c, ind_d]
                                    if s_j == s_d and s_k == s_c:
                                        jkcd -= phys_oovv[ind_j, ind_k, ind_d, ind_c]

                                    ss_mat[ia, jb] += 0.25 * ikcd * jkcd / e_ikcd
                                    ss_mat[ia, jb] += 0.25 * ikcd * jkcd / e_jkcd

            # M^{(2),C}_{ia,jb} = -0.5 \sum_{ck} [
            #                     <ac||ik><jk||bc> / e_ikac +
            #                     <ac||ik><jk||bc> / e_jkbc ]
            #
            # <ac||ik> = <ik|ac> - <ik|ca>
            # <jk||bc> = <jk|bc> - <jk|cb>

            for i, (ind_i, e_i, s_i) in enumerate(zip(ind_occ, e_occ, s_occ)):
                for a, (ind_a, e_a, s_a) in enumerate(zip(ind_vir, e_vir, s_vir)):
                    ia = i * nvir2 + a

                    for j, (ind_j, e_j, s_j) in enumerate(zip(ind_occ, e_occ, s_occ)):
                        for b, (ind_b, e_b, s_b) in enumerate(zip(ind_vir, e_vir, s_vir)):
                            jb = j * nvir2 + b

                            # \sum_{ck}
                            for c, (ind_c, e_c, s_c) in enumerate(zip(ind_vir, e_vir, s_vir)):
                                for k, (ind_k, e_k, s_k) in enumerate(zip(ind_occ, e_occ, s_occ)):

                                    e_ikac = e_a + e_c - e_i - e_k
                                    e_jkbc = e_b + e_c - e_j - e_k

                                    ikac = 0.0
                                    if s_i == s_a and s_k == s_c:
                                        ikac += phys_oovv[ind_i, ind_k, ind_a, ind_c]
                                    if s_i == s_c and s_k == s_a:
                                        ikac -= phys_oovv[ind_i, ind_k, ind_c, ind_a]

                                    jkbc = 0.0
                                    if s_j == s_b and s_k == s_c:
                                        jkbc += phys_oovv[ind_j, ind_k, ind_b, ind_c]
                                    if s_j == s_c and s_k == s_b:
                                        jkbc -= phys_oovv[ind_j, ind_k, ind_c, ind_b]

                                    ss_mat[ia, jb] -= 0.5 * ikac * jkbc / e_ikac
                                    ss_mat[ia, jb] -= 0.5 * ikac * jkbc / e_jkbc

            #
            # single-double block
            #

            sd_mat = np.zeros((s_dim, d_dim))

            # M^{(1)}_{ia,klcd} = <kl||id> \delta_{ac}
            #                   - <kl||ic> \delta_{ad}
            #                   - <al||cd> \delta_{ik}
            #                   + <ak||cd> \delta_{il}

            for i, (ind_i, e_i, s_i) in enumerate(zip(ind_occ, e_occ, s_occ)):
                for a, (ind_a, e_a, s_a) in enumerate(zip(ind_vir, e_vir, s_vir)):
                    ia = i * nvir2 + a

                    count_klcd = 0
                    for k, (ind_k, e_k, s_k) in enumerate(zip(ind_occ, e_occ, s_occ)):
                        for l, (ind_l, e_l, s_l) in enumerate(zip(ind_occ[:k], e_occ[:k], s_occ[:k])):
                            for c, (ind_c, e_c, s_c) in enumerate(zip(ind_vir, e_vir, s_vir)):
                                for d, (ind_d, e_d, s_d) in enumerate(zip(ind_vir[:c], e_vir[:c], s_vir[:c])):
                                    klcd = count_klcd
                                    count_klcd += 1

                                    # <kl||id> \delta_{ac}
                                    klid = 0.0
                                    if s_k == s_i and s_l == s_d:
                                        klid += phys_ooov[ind_k, ind_l, ind_i, ind_d]
                                    if s_l == s_i and s_k == s_d:
                                        klid -= phys_ooov[ind_l, ind_k, ind_i, ind_d]
                                    if a == c:
                                        sd_mat[ia, klcd] += klid

                                    # - <kl||ic> \delta_{ad}
                                    klic = 0.0
                                    if s_k == s_i and s_l == s_c:
                                        klic += phys_ooov[ind_k, ind_l, ind_i, ind_c]
                                    if s_l == s_i and s_k == s_c:
                                        klic -= phys_ooov[ind_l, ind_k, ind_i, ind_c]
                                    if a == d:
                                        sd_mat[ia, klcd] -= klic

                                    # - <la||dc> \delta_{ik}
                                    ladc = 0.0
                                    if s_l == s_d and s_a == s_c:
                                        ladc += phys_ovvv[ind_l, ind_a, ind_d, ind_c]
                                    if s_l == s_c and s_a == s_d:
                                        ladc -= phys_ovvv[ind_l, ind_a, ind_c, ind_d]
                                    if i == k:
                                        sd_mat[ia, klcd] -= ladc

                                    # + <ka||dc> \delta_{il}
                                    kadc = 0.0
                                    if s_k == s_d and s_a == s_c:
                                        kadc += phys_ovvv[ind_k, ind_a, ind_d, ind_c]
                                    if s_k == s_c and s_a == s_d:
                                        kadc -= phys_ovvv[ind_k, ind_a, ind_c, ind_d]
                                    if i == l:
                                        sd_mat[ia, klcd] += kadc

            #
            # double-double block (diagonal)
            #

            dd_diag = np.zeros(d_dim)

            count_klcd = 0
            for k, (ind_k, e_k, s_k) in enumerate(zip(ind_occ, e_occ, s_occ)):
                for l, (ind_l, e_l, s_l) in enumerate(zip(ind_occ[:k], e_occ[:k], s_occ[:k])):
                    for c, (ind_c, e_c, s_c) in enumerate(zip(ind_vir, e_vir, s_vir)):
                        for d, (ind_d, e_d, s_d) in enumerate(zip(ind_vir[:c], e_vir[:c], s_vir[:c])):
                            dd_diag[count_klcd] = e_c + e_d - e_k - e_l
                            count_klcd += 1

            #
            # full matrix
            #

            adc_mat = np.zeros((s_dim + d_dim, s_dim + d_dim))
            adc_mat[:s_dim, :s_dim] = ss_mat[:, :]
            adc_mat[:s_dim, s_dim:] = sd_mat[:, :]
            adc_mat[s_dim:, :s_dim] = sd_mat.T[:, :]
            adc_mat[s_dim:, s_dim:] = np.diag(dd_diag)[:, :]

            return adc_mat

        else:
            return None
