from mpi4py import MPI
from veloxchem import ExcitationVector
from veloxchem import mpi_master
from veloxchem import hartree_in_ev
from veloxchem.veloxchemlib import szblock
from veloxchem import molorb
from veloxchem import get_qq_type
from veloxchem import BlockDavidsonSolver
from veloxchem import MolecularOrbitals
import numpy as np
import time as tm
import h5py


class WaterSto3g:
    """
    Implements ADC(2) computation driver for water/sto-3g.

    :param comm:
        The MPI communicator.
    :param ostream:
        The output stream.

    Instance variable
        - nstates: The number of excited states determined by driver.
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
        Initializes ADC(2) computation driver for water/sto-3g.
        """

        # excited states information
        self.nstates = 5

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

    def compute(self, molecule, basis, scf_tensors, h5file):
        """
        Performs ADC(2) calculation for water/sto-3g.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param scf_tensors:
            The tensors from converged SCF wavefunction.
        :param h5file:
            The hdf5 file containing the reference ADC(2) matrix.
        :return:
            A dictionary containing excitation energies.
        """

        start_time = tm.time()

        if self.rank == mpi_master():
            self.print_header()

            mo = scf_tensors['C']
            ea = scf_tensors['E']

            nocc = molecule.number_of_alpha_electrons()
            nvir = mo.shape[1] - nocc

            nocc2 = nocc * 2
            nvir2 = nvir * 2

            s_dim = nocc2 * nvir2
            d_dim = (nocc2 * (nocc2 - 1) // 2) * (nvir2 * (nvir2 - 1) // 2)

            hf = h5py.File(h5file, 'r')
            matrix = np.array(hf.get('m'))
            hf.close()

            assert matrix.shape == (s_dim + d_dim, s_dim + d_dim)

            ss_mat = matrix[:s_dim, :s_dim]
            sd_mat = matrix[:s_dim, s_dim:]
            dd_diag = np.diag(matrix[s_dim:, s_dim:])

            ss_diag = np.diag(ss_mat).reshape(s_dim, 1)

            mol_orbs = MolecularOrbitals([mo], [ea], molorb.rest)
            diag_mat, trial_vecs, reigs = self.gen_trial_vectors(
                mol_orbs, molecule)

            trial_mat = np.zeros((s_dim, len(trial_vecs)))
            for ind, vec in enumerate(trial_vecs):
                tvec = vec.zvector_to_numpy().reshape(nocc, nvir)
                ov_trial = trial_mat[:, ind].reshape(nocc2, nvir2)
                ov_trial[:nocc, :nvir] = tvec[:, :]
                ov_trial[nocc:, nvir:] = tvec[:, :]
                ov_trial /= np.linalg.norm(ov_trial)

            # block Davidson algorithm setup

            self.solver = BlockDavidsonSolver()

            # start iterations

            for iteration in range(self.max_iter):

                # compute sigma vectors

                sigma_mat = np.zeros(trial_mat.shape)

                for vecind in range(trial_mat.shape[1]):
                    rjb = trial_mat[:, vecind]

                    sigma = np.dot(ss_mat, rjb)

                    Ab = np.dot(sd_mat.T, rjb)
                    Ab /= (reigs[vecind] - dd_diag)
                    sigma += np.dot(sd_mat, Ab)

                    sigma_mat[:, vecind] = sigma[:]

                sigma_mat = self.comm.reduce(sigma_mat,
                                             op=MPI.SUM,
                                             root=mpi_master())

                self.solver.add_iteration_data(sigma_mat, trial_mat, iteration)
                trial_mat = self.solver.compute(ss_diag)
                reigs, rnorms = self.solver.get_eigenvalues()

                self.print_iter_data(iteration)

                # check convergence

                self.check_convergence(iteration)

                if self.is_converged:
                    break

            # print converged excited states

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
        """
        Generates set of trial vectors.

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

            return (diag_mat, trial_vecs, reigs)

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

    def print_iter_data(self, iteration):
        """
        Prints excited states solver iteration data to output stream.

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
        for i in range(reigs.shape[0]):
            exec_str = "State {:2d}: {:5.8f} ".format(i + 1, reigs[i])
            exec_str += "au Residual Norm: {:3.8f}".format(rnorms[i])
            self.ostream.print_header(exec_str.ljust(84))
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
