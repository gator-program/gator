from mpi4py import MPI
from veloxchem import BlockDavidsonSolver
from veloxchem import ExcitationVector
from veloxchem import MOIntegralsDriver
from veloxchem import MolecularOrbitals
from veloxchem import mpi_master
from veloxchem import get_qq_type
from veloxchem.veloxchemlib import szblock
from veloxchem import molorb
import numpy as np
import time as tm


class AdcOneDriver:
    """Implements ADC(1) computation driver.

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
        """Initializes ADC(1) computation driver.
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
        """
        Updates settings in ADC(1) driver.

        :param settings:
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
        """
        Performs ADC(1) calculation.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param scf_tensors:
            The tensors from converged SCF wavefunction.
        """

        if self.rank == mpi_master():
            mo = scf_tensors['C']
            ea = scf_tensors['E']
            mol_orbs = MolecularOrbitals([mo], [ea], molorb.rest)
        else:
            mol_orbs = MolecularOrbitals()
            mo = None

        mo = self.comm.bcast(mo, root=mpi_master())
        nocc = molecule.number_of_alpha_electrons()
        nvir = mo.shape[1] - nocc

        if self.rank == mpi_master():
            fa = scf_tensors['F'][0]
            fmo = np.matmul(mo.T, np.matmul(fa, mo))
            fab = fmo[nocc:, nocc:]
            fij = fmo[:nocc, :nocc]

        # set start time

        start_time = tm.time()

        # set up trial excitation vectors on master node

        diag_mat, trial_vecs = self.gen_trial_vectors(mol_orbs, molecule)

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

        # <aj||ib> = <aj|ib> - <aj|bi>
        #          = <ij|ab> - <ib|ja>
        #             OO VV     OV OV

        moints_drv = MOIntegralsDriver(self.comm, self.ostream)
        grps = [p for p in range(self.nodes)]

        # <OO|VV>
        oovv = moints_drv.compute(molecule, basis, mol_orbs, 'OOVV', grps)

        # <OV|OV>
        ovov = moints_drv.compute(molecule, basis, mol_orbs, 'OVOV', grps)

        if self.rank == mpi_master():
            self.print_header()

        # start iterations

        for iteration in range(self.max_iter):

            sigma_mat = np.zeros(trial_mat.shape)

            for vecind in range(trial_mat.shape[1]):
                cjb = trial_mat[:nocc * nvir, vecind].reshape(nocc, nvir)

                # 0th order contribution

                if self.rank == mpi_master():
                    mat = np.matmul(cjb, fab.T) - np.matmul(fij, cjb)
                else:
                    mat = np.zeros((nocc, nvir))

                # 1st order contribution

                if not self.triplet:
                    # 'ijab,jb->ia'
                    for pair in oovv.get_gen_pairs():
                        i = pair.first()
                        j = pair.second()
                        ab = oovv.to_numpy(pair)
                        mat[i, :] += (2.0 * np.matmul(ab, cjb.T))[:, j]

                # 'ibja,jb->ia'
                for pair in ovov.get_gen_pairs():
                    i = pair.first()
                    j = pair.second()
                    ba = ovov.to_numpy(pair)
                    mat[i, :] -= (np.matmul(ba.T, cjb.T))[:, j]

                sigma_mat[:, vecind] = mat.reshape(nocc * nvir)[:]

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
            self.print_convergence(start_time)
            reigs, rnorms = self.solver.get_eigenvalues()
            return reigs
        else:
            return None

    def print_header(self):
        """Prints ADC(1) driver setup header to output stream"""

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

            nocc = molecule.number_of_alpha_electrons()
            norb = mol_orbs.number_mos()

            zvec = ExcitationVector(szblock.aa, 0, nocc, nocc, norb, True)
            exci_list = zvec.small_energy_identifiers(mol_orbs, self.nstates)

            diag_mat = zvec.diagonal_to_numpy(mol_orbs)

            trial_vecs = []
            for exci_ind in exci_list:
                trial_vecs.append(
                    ExcitationVector(szblock.aa, 0, nocc, nocc, norb, True))
                trial_vecs[-1].set_zcoefficient(1.0, exci_ind)

            return diag_mat, trial_vecs

        return None, []

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
        for i in range(reigs.shape[0]):
            exec_str = "State {:2d}: {:5.8f} ".format(i + 1, reigs[i])
            exec_str += "au Residual Norm: {:3.8f}".format(rnorms[i])
            self.ostream.print_header(exec_str.ljust(84))

        # flush output stream
        self.ostream.print_blank()
        self.ostream.flush()

    def print_convergence(self, start_time):
        """
        Prints excited states information.

        :param start_time:
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
        self.ostream.flush()
