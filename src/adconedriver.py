from mpi4py import MPI
import numpy as np
import time as tm
import math

from veloxchem import BlockDavidsonSolver
from veloxchem import ElectricDipoleIntegralsDriver
from veloxchem import mpi_master
from veloxchem import hartree_in_ev
from veloxchem import get_qq_type

from .gatortask import OutputStream
from .mointsdriver import MOIntegralsDriver


class AdcOneDriver:
    """
    Implements ADC(1) computation driver.

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

    def __init__(self, comm, ostream=OutputStream()):
        """
        Initializes ADC(1) computation driver.
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
        Updates settings in ADC(1) driver.

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

    def compute(self,
                molecule,
                basis,
                scf_tensors,
                mo_indices=None,
                mo_integrals=None):
        """
        Performs ADC(1) calculation.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param scf_tensors:
            The tensors from converged SCF wavefunction.
        :return:
            Excitation energies.
        """

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

        mo_occ = mo[:, :nocc].copy()
        mo_vir = mo[:, nocc:].copy()

        eocc = ea[:nocc]
        evir = ea[nocc:]
        e_ov = -eocc.reshape(-1, 1) + evir
        e_vv = evir.reshape(-1, 1) + evir

        epsilon = {
            'o': eocc,
            'v': evir,
            'ov': e_ov,
            'vv': e_vv,
        }

        if self.rank == mpi_master():
            self.print_header()
            self.ostream.print_info(
                'Number of occupied orbitals: {:d}'.format(nocc))
            self.ostream.print_info(
                'Number of virtual orbitals: {:d}'.format(nvir))
            self.ostream.print_blank()

        # set start time

        start_time = tm.time()

        # set up trial excitation vectors

        diag_mat = e_ov.copy().reshape(-1)

        exci_list = [(evir[a] - eocc[nocc - 1 - i], nocc - 1 - i, a)
                     for i in range(min(self.nstates, nocc))
                     for a in range(min(self.nstates, nvir))]

        exci_list = sorted(exci_list)[:self.nstates]

        trial_mat = np.zeros((nocc * nvir, len(exci_list)))
        for ind, (delta_e, i, a) in enumerate(exci_list):
            trial_ov = trial_mat[:, ind].reshape(nocc, nvir)
            trial_ov[i, a] = 1.0

        # block Davidson algorithm setup

        self.solver = BlockDavidsonSolver()

        # MO integrals

        if mo_indices is None or mo_integrals is None:
            moints_drv = MOIntegralsDriver(self.comm, self.ostream)
            moints_drv.update_settings({
                'qq_type': self.qq_type,
                'eri_thresh': self.eri_thresh
            })
            mo_indices, mo_integrals = moints_drv.compute(
                molecule, basis, scf_tensors, ['oo'])

        # start iterations

        for iteration in range(self.max_iter):

            sigma_mat = self.compute_sigma(trial_mat, epsilon, mo_indices,
                                           mo_integrals)

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
            rvecs = self.solver.ritz_vectors.copy()
        else:
            reigs = None
            rvecs = None
        rvecs = self.comm.bcast(rvecs, root=mpi_master())

        fvecs = self.compute_spectral_amplitudes(rvecs, epsilon, mo_indices,
                                                 mo_integrals)

        osc = self.compute_oscillator_strengths(molecule, basis, mo_occ, mo_vir,
                                                reigs, fvecs)

        if self.rank == mpi_master():
            self.print_convergence(start_time, reigs, osc)
            return {
                'eigenvalues': reigs,
                'eigenvectors': rvecs,
                'oscillator_strengths': osc,
            }
        else:
            return {}

    def compute_sigma(self, trial_mat, epsilon, mo_indices, mo_integrals):
        """
        Computes sigma vectors for ADC(1).

        :param trial_mat:
            The trial vectors.
        :param epsilon:
            The dictionary containing orbital energy differences.
        :param mo_indices:
            The pair indices for MO integrals.
        :param mo_integrals:
            The MO integrals.
        :return:
            The sigma vectors.
        """

        eocc = epsilon['o']
        evir = epsilon['v']

        nocc = eocc.shape[0]
        nvir = evir.shape[0]

        sigma_mat = np.zeros(trial_mat.shape)

        for vecind in range(trial_mat.shape[1]):
            # note: use copy() to ensure contiguous memory
            rjb = trial_mat[:, vecind].reshape(nocc, nvir).copy()

            sigma = np.zeros(rjb.shape)

            # 0th-order contribution

            if self.rank == mpi_master():
                sigma += np.matmul(rjb, np.diag(evir)) - np.matmul(
                    np.diag(eocc), rjb)

            # 1st-order contribution

            for ind, (i, j) in enumerate(mo_indices['oo']):
                # [ 2 (ia|jb) - (ij|ab) ] R_jb
                vv_1 = mo_integrals['chem_ovov_K'][ind]
                vv_2 = mo_integrals['chem_oovv_J'][ind]
                ab = 2.0 * vv_1 - vv_2
                sigma[i, :] += np.dot(ab, rjb[j, :])

            sigma_mat[:, vecind] = sigma.reshape(nocc * nvir)[:]

        sigma_mat = self.comm.reduce(sigma_mat, op=MPI.SUM, root=mpi_master())

        return sigma_mat

    def compute_spectral_amplitudes(self, rvecs, epsilon, mo_indices,
                                    mo_integrals):
        """
        Computes spectral amplitudes for ADC(1).

        :param rvecs:
            The one-particle excitation vectors.
        :param epsilon:
            The dictionary containing orbital energy differences.
        :param mo_indices:
            The pair indices for MO integrals.
        :param mo_integrals:
            The MO integrals.
        :return:
            The spectral amplitudes.
        """

        # F_ia
        # 0th-order: R_ia
        # 1st-order: { [ +1(aj|bi) -2(ai|bj) ] / e_abji } R_jb

        eocc = epsilon['o']
        evir = epsilon['v']
        e_vv = epsilon['vv']

        nocc = eocc.shape[0]
        nvir = evir.shape[0]

        fvecs = np.zeros(rvecs.shape)

        for vecind in range(rvecs.shape[1]):
            # note: use copy() to ensure contiguous memory
            rjb = rvecs[:, vecind].reshape(nocc, nvir).copy()

            fia = np.zeros(rjb.shape)

            for ind, (i, j) in enumerate(mo_indices['oo']):
                # { [ -2(ia|jb) +1(ib|ja) ] / e_abji } R_jb
                ab = mo_integrals['chem_ovov_K'][ind]
                de = e_vv - eocc[i] - eocc[j]
                fia[i, :] += np.dot((-2.0 * ab + ab.T) / de, rjb[j, :])

            fvecs[:, vecind] += fia.reshape(nocc * nvir)[:]

        fvecs = self.comm.reduce(fvecs, op=MPI.SUM, root=mpi_master())

        if self.rank == mpi_master():
            fvecs += rvecs

        return fvecs

    def compute_oscillator_strengths(self, molecule, basis, mo_occ, mo_vir,
                                     reigs, fvecs):
        """
        Computes oscillator strengths for ADC(1).

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param mo_occ:
            The MO coefficients of occupied orbitals.
        :param mo_vir:
            The MO coefficients of virtual orbitals.
        :param reigs:
            The excitation energies.
        :param fvecs:
            The spectrial amplitudes.
        :return:
            The oscillator strengths.
        """

        dipole_drv = ElectricDipoleIntegralsDriver(self.comm)
        dipole_mats = dipole_drv.compute(molecule, basis)

        if self.rank == mpi_master():
            sqrt_2 = math.sqrt(2.0)

            nocc = mo_occ.shape[1]
            nvir = mo_vir.shape[1]

            dipole_ints = [
                dipole_mats.x_to_numpy(),
                dipole_mats.y_to_numpy(),
                dipole_mats.z_to_numpy(),
            ]

            oscillator_strengths = []

            for s in range(self.nstates):
                exc_ene = reigs[s]
                exc_vec = sqrt_2 * fvecs[:, s].reshape(nocc, nvir).copy()

                trans_dens = np.linalg.multi_dot([mo_occ, exc_vec, mo_vir.T])

                trans_dipole = np.array(
                    [np.vdot(trans_dens, dipole_ints[d]) for d in range(3)])

                dipole_strength = np.sum(trans_dipole**2)
                oscillator_strengths.append(2.0 / 3.0 * dipole_strength *
                                            exc_ene)

            return oscillator_strengths
        else:
            return None

    def print_header(self):
        """
        Prints ADC(1) driver setup header to output stream.
        """

        self.ostream.print_blank()
        self.ostream.print_header("ADC(1) Driver Setup")
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

    def check_convergence(self, iteration):
        """
        Checks convergence of excitation energies.

        :param iter:
            The current excited states solver iteration.
        """

        self.cur_iter = iteration

        if self.rank == mpi_master():
            self.is_converged = self.solver.check_convergence(self.conv_thresh)
        else:
            self.is_converged = False

        self.is_converged = self.comm.bcast(self.is_converged,
                                            root=mpi_master())

    def print_iter_data(self, iteration):
        """
        Prints excited states solver iteration data.

        :param iteration:
            The current excited states solver iteration.
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
        for i in range(reigs[:self.nstates].shape[0]):
            exec_str = "State {:2d}: {:5.8f} ".format(i + 1, reigs[i])
            exec_str += "au Residual Norm: {:3.8f}".format(rnorms[i])
            self.ostream.print_header(exec_str.ljust(84))

        # flush output stream
        self.ostream.print_blank()
        self.ostream.flush()

    def print_convergence(self, start_time, reigs, osc):
        """
        Prints convergence and excited state information.

        :param start_time:
            The start time of calculation.
        :param reigs:
            The excitation energies.
        :param osc:
            The oscillator strengths.
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
            valstr = "ADC(1) excited states"
            self.ostream.print_header(valstr.ljust(92))
            self.ostream.print_header(('-' * len(valstr)).ljust(92))
            for s, e in enumerate(reigs[:self.nstates]):
                valstr = 'Excited State {:>5s}: '.format('S' + str(s + 1))
                valstr += '{:15.8f} a.u. '.format(e)
                valstr += '{:12.5f} eV'.format(e * hartree_in_ev())
                valstr += '    Osc.Str. {:.5f}'.format(osc[s])
                self.ostream.print_header(valstr.ljust(92))
            self.ostream.print_blank()
            self.ostream.print_blank()
        self.ostream.flush()
