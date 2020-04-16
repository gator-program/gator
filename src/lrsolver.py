import numpy as np
import time as tm
import os

from veloxchem import mpi_master
from veloxchem import get_qq_type
from veloxchem import assert_msg_critical
from veloxchem.inputparser import parse_frequencies
from veloxchem.lrmatvecdriver import get_rhs
from veloxchem.lrmatvecdriver import construct_ed_sd_half
from veloxchem.lrmatvecdriver import remove_linear_dependence_half
from veloxchem.lrmatvecdriver import orthogonalize_gram_schmidt_half
from veloxchem.lrmatvecdriver import normalize_half

from .mointsdriver import MOIntegralsDriver
from .adctwodriver import AdcTwoDriver


class LinearResponseSolver:
    """
    Implements linear response solver.

    :param comm:
        The MPI communicator.
    :param ostream:
        The output stream.

    Instance variables
        - a_operator: The A operator.
        - a_components: Cartesian components of the A operator.
        - b_operator: The B operator.
        - b_components: Cartesian components of the B operator.
        - frequencies: The frequencies.
        - eri_thresh: The electron repulsion integrals screening threshold.
        - qq_type: The electron repulsion integrals screening scheme.
        - conv_thresh: The convergence threshold for the solver.
        - max_iter: The maximum number of solver iterations.
        - cur_iter: Index of the current iteration.
        - small_thresh: The norm threshold for a vector to be considered a zero
          vector.
        - lindep_thresh: The threshold for removing linear dependence in the
          trial vectors.
        - is_converged: The flag for convergence.
        - comm: The MPI communicator.
        - rank: The MPI rank.
        - nodes: Number of MPI processes.
        - ostream: The output stream.
        - restart: The flag for restarting from checkpoint file.
        - checkpoint_file: The name of checkpoint file.
        - checkpoint_time: The timer of checkpoint file.
        - timing: The flag for printing timing information.
        - profiling: The flag for printing profiling information.
    """

    def __init__(self, comm, ostream):
        """
        Initializes linear response solver to default setup.
        """

        # operators and frequencies
        self.a_operator = 'dipole'
        self.a_components = 'xyz'
        self.b_operator = 'dipole'
        self.b_components = 'xyz'
        self.frequencies = (0,)

        # ERI settings
        self.eri_thresh = 1.0e-15
        self.qq_type = 'QQ_DEN'

        # solver setup
        self.conv_thresh = 1.0e-4
        self.max_iter = 50
        self.cur_iter = 0
        self.small_thresh = 1.0e-10
        self.lindep_thresh = 1.0e-6
        self.is_converged = False

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

        # restart information
        self.restart = True
        self.checkpoint_file = None
        self.checkpoint_time = None

        self.timing = False
        self.profiling = False

    def update_settings(self, rsp_dict, scf_drv=None):
        """
        Updates response and method settings in linear response solver.

        :param rsp_dict:
            The dictionary of response dict.
        :param scf_drv:
            The scf driver.
        """

        if 'a_operator' in rsp_dict:
            self.a_operator = rsp_dict['a_operator'].lower()
        if 'a_components' in rsp_dict:
            self.a_components = rsp_dict['a_components'].lower()
        if 'b_operator' in rsp_dict:
            self.b_operator = rsp_dict['b_operator'].lower()
        if 'b_components' in rsp_dict:
            self.b_components = rsp_dict['b_components'].lower()
        if 'frequencies' in rsp_dict:
            self.frequencies = parse_frequencies(rsp_dict['frequencies'])

        if 'qq_type' in rsp_dict:
            self.qq_type = rsp_dict['qq_type'].upper()
        elif scf_drv is not None:
            # inherit from SCF
            self.qq_type = scf_drv.qq_type

        if 'eri_thresh' in rsp_dict:
            self.eri_thresh = float(rsp_dict['eri_thresh'])
        elif scf_drv is not None:
            # inherit from SCF
            self.eri_thresh = scf_drv.eri_thresh

        if 'conv_thresh' in rsp_dict:
            self.conv_thresh = float(rsp_dict['conv_thresh'])
        if 'max_iter' in rsp_dict:
            self.max_iter = int(rsp_dict['max_iter'])
        if 'lindep_thresh' in rsp_dict:
            self.lindep_thresh = float(rsp_dict['lindep_thresh'])

        if 'restart' in rsp_dict:
            key = rsp_dict['restart'].lower()
            self.restart = True if key == 'yes' else False
        if 'checkpoint_file' in rsp_dict:
            self.checkpoint_file = rsp_dict['checkpoint_file']

        if 'timing' in rsp_dict:
            key = rsp_dict['timing'].lower()
            self.timing = True if key in ['yes', 'y'] else False
        if 'profiling' in rsp_dict:
            key = rsp_dict['profiling'].lower()
            self.profiling = True if key in ['yes', 'y'] else False

    def compute(self, molecule, basis, scf_tensors, v1=None):
        """
        Performs linear response calculation for a molecule and a basis set.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param scf_tensors:
            The dictionary of tensors from converged SCF wavefunction.
        :param v1:
            The gradients on the right-hand side. If not provided, v1 will be
            computed for the B operator.

        :return:
            A dictionary containing response functions, solutions and a
            dictionarry containing solutions and kappa values when called from
            a non-linear response module.
        """

        if self.profiling:
            import cProfile
            import pstats
            import io
            pr = cProfile.Profile()
            pr.enable()

        if self.timing:
            self.timing_dict = {
                'reduced_space': [0.0],
                'new_trials': [0.0],
            }
            timing_t0 = tm.time()

        if self.rank == mpi_master():
            self.print_header()

        self.start_time = tm.time()
        self.checkpoint_time = self.start_time

        # sanity check
        nalpha = molecule.number_of_alpha_electrons()
        nbeta = molecule.number_of_beta_electrons()
        assert_msg_critical(
            nalpha == nbeta,
            'LinearResponseSolver: not implemented for unrestricted case')

        # process MOs

        if self.rank == mpi_master():
            mo = scf_tensors['C']
            ea = scf_tensors['E']
        else:
            mo = None
            ea = None
        mo = self.comm.bcast(mo, root=mpi_master())
        ea = self.comm.bcast(ea, root=mpi_master())

        nocc = molecule.number_of_alpha_electrons()
        # nvir = mo.shape[1] - nocc
        norb = mo.shape[1]

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

        moints_drv = MOIntegralsDriver(self.comm, self.ostream)
        moints_drv.update_settings({
            'qq_type': self.qq_type,
            'eri_thresh': self.eri_thresh
        })
        mo_indices, mo_integrals = moints_drv.compute(molecule, basis,
                                                      scf_tensors)

        # compute auxiliary matrices

        adc_drv = AdcTwoDriver(self.comm, self.ostream)

        xA_ab, xB_ij = adc_drv.compute_xA_xB(epsilon, mo_indices, mo_integrals)

        if self.rank == mpi_master():
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

        # right-hand side

        b_rhs = get_rhs(self.b_operator, self.b_components, molecule, basis,
                        scf_tensors, self.rank, self.comm)

        if self.rank == mpi_master():
            v1 = {(op, w): v for op, v in zip(self.b_components, b_rhs)
                  for w in self.frequencies}
            op_freq_keys = list(v1.keys())
            precond = {
                w: self.get_precond(ea, nocc, norb, w) for w in self.frequencies
            }

        # initial guess

        bger = None
        bung = None
        new_trials_ger = None
        new_trials_ung = None

        if self.rank == mpi_master():

            igs = self.initial_guess(v1, self.frequencies, precond)
            bger, bung = self.setup_trials(igs)

            if self.timing:
                elapsed_time = tm.time() - timing_t0
                self.timing_dict['reduced_space'][0] += elapsed_time
                timing_t0 = tm.time()

            assert_msg_critical(
                bger.any() or bung.any(),
                'LinearResponseSolver.compute: trial vector is empty')

            if bger is None or not bger.any():
                bger = np.zeros((bung.shape[0], 0))
            if bung is None or not bung.any():
                bung = np.zeros((bger.shape[0], 0))

        e2bger = adc_drv.compute_sigma(bger, np.zeros(bger.shape[1]), epsilon,
                                       auxiliary_matrices, mo_indices,
                                       mo_integrals)
        e2bung = adc_drv.compute_sigma(bung, np.zeros(bung.shape[1]), epsilon,
                                       auxiliary_matrices, mo_indices,
                                       mo_integrals)

        solutions = {}
        residuals = {}
        relative_residual_norm = {}

        if self.timing:
            self.timing_dict['new_trials'][0] += tm.time() - timing_t0
            timing_t0 = tm.time()

        # start iterations

        for iteration in range(self.max_iter):

            if self.timing:
                self.timing_dict['reduced_space'].append(0.0)
                self.timing_dict['new_trials'].append(0.0)

            if self.rank == mpi_master():
                self.cur_iter = iteration
                xvs = []

                n_ger = bger.shape[1]
                n_ung = bung.shape[1]

                e2gg = np.matmul(bger.T, e2bger) * 2.0
                e2uu = np.matmul(bung.T, e2bung) * 2.0
                s2ug = np.matmul(bung.T, bger) * 4.0

                for op, freq in op_freq_keys:
                    if (iteration > 0) and (relative_residual_norm[(op, freq)] <
                                            self.conv_thresh):
                        continue

                    v = v1[(op, freq)]

                    gradger, gradung = self.decomp_grad(v)

                    g_ger = np.matmul(bger.T, gradger) * 2.0
                    g_ung = np.matmul(bung.T, gradung) * 2.0

                    mat = np.zeros((n_ger + n_ung, n_ger + n_ung))
                    mat[:n_ger, :n_ger] = e2gg[:, :]
                    mat[:n_ger, n_ger:] = -freq * s2ug.T[:, :]
                    mat[n_ger:, :n_ger] = -freq * s2ug[:, :]
                    mat[n_ger:, n_ger:] = e2uu[:, :]

                    g = np.zeros(n_ger + n_ung)
                    g[:n_ger] = g_ger[:]
                    g[n_ger:] = g_ung[:]

                    c = np.linalg.solve(mat, g)

                    c_ger = c[:n_ger]
                    c_ung = c[n_ger:]

                    x_ger = np.matmul(bger, c_ger)
                    x_ung = np.matmul(bung, c_ung)

                    x_ger_full = np.hstack((x_ger, x_ger))
                    x_ung_full = np.hstack((x_ung, -x_ung))

                    x = x_ger_full + x_ung_full

                    r_ger = np.matmul(e2bger, c_ger) - freq * 2.0 * np.matmul(
                        bung, c_ung) - gradger
                    r_ung = np.matmul(e2bung, c_ung) - freq * 2.0 * np.matmul(
                        bger, c_ger) - gradung

                    r = np.array([r_ger, r_ung]).flatten()

                    xv = np.dot(x, v)
                    xvs.append((op, freq, xv))

                    rn = np.linalg.norm(r) * np.sqrt(2.0)
                    xn = np.linalg.norm(x)
                    relative_residual_norm[(op, freq)] = rn / xn

                    if relative_residual_norm[(op, freq)] < self.conv_thresh:
                        solutions[(op, freq)] = x
                    else:
                        residuals[(op, freq)] = r

                # write to output
                self.ostream.print_info(
                    '{:d} gerade trial vectors in reduced space'.format(n_ger))
                self.ostream.print_info(
                    '{:d} ungerade trial vectors in reduced space'.format(
                        n_ung))
                self.ostream.print_blank()

                self.print_iteration(relative_residual_norm, xvs)

            if self.timing:
                tid = iteration + 1
                self.timing_dict['reduced_space'][tid] += tm.time() - timing_t0
                timing_t0 = tm.time()

            # check convergence
            self.check_convergence(relative_residual_norm)

            if self.is_converged:
                break

            # update trial vectors
            if self.rank == mpi_master():
                new_trials_ger, new_trials_ung = self.setup_trials(
                    residuals, precond, bger, bung)

                residuals.clear()

                assert_msg_critical(
                    new_trials_ger.any() or new_trials_ung.any(),
                    'LinearResponseSolver: unable to add new trial vector')

                if new_trials_ger is None or not new_trials_ger.any():
                    new_trials_ger = np.zeros((new_trials_ung.shape[0], 0))
                if new_trials_ung is None or not new_trials_ung.any():
                    new_trials_ung = np.zeros((new_trials_ger.shape[0], 0))

                bger = np.append(bger, new_trials_ger, axis=1)
                bung = np.append(bung, new_trials_ung, axis=1)

            if self.timing:
                tid = iteration + 1
                self.timing_dict['reduced_space'][tid] += tm.time() - timing_t0
                timing_t0 = tm.time()

            new_e2bger = adc_drv.compute_sigma(
                new_trials_ger, np.zeros(new_trials_ger.shape[1]), epsilon,
                auxiliary_matrices, mo_indices, mo_integrals)
            new_e2bung = adc_drv.compute_sigma(
                new_trials_ung, np.zeros(new_trials_ung.shape[1]), epsilon,
                auxiliary_matrices, mo_indices, mo_integrals)

            if self.rank == mpi_master():
                e2bger = np.append(e2bger, new_e2bger, axis=1)
                e2bung = np.append(e2bung, new_e2bung, axis=1)

            if self.timing:
                tid = iteration + 1
                self.timing_dict['new_trials'][tid] += tm.time() - timing_t0
                timing_t0 = tm.time()

        # converged?
        if self.rank == mpi_master():
            self.print_convergence()

            assert_msg_critical(
                self.is_converged,
                'LinearResponseSolver.compute: failed to converge')

            if self.timing:
                self.print_timing()

        if self.profiling:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(20)
            if self.rank == mpi_master():
                for line in s.getvalue().split(os.linesep):
                    self.ostream.print_info(line)

        # calculate response functions
        a_rhs = get_rhs(self.a_operator, self.a_components, molecule, basis,
                        scf_tensors, self.rank, self.comm)

        if self.rank == mpi_master():
            va = {op: v for op, v in zip(self.a_components, a_rhs)}
            rsp_funcs = {}
            for aop in self.a_components:
                for bop, w in solutions:
                    rsp_funcs[(aop, bop,
                               w)] = -np.dot(va[aop], solutions[(bop, w)])
            return {
                'response_functions': rsp_funcs,
                'solutions': solutions,
            }
        else:
            return {}

    def print_header(self):
        """
        Prints linear response solver setup header to output stream.
        """

        self.ostream.print_blank()
        self.ostream.print_header("Linear Response Solver Setup")
        self.ostream.print_header(30 * "=")
        self.ostream.print_blank()

        str_width = 60

        cur_str = "Max. Number of Iterations       : " + str(self.max_iter)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = "Convergence Threshold           : " + \
            "{:.1e}".format(self.conv_thresh)
        self.ostream.print_header(cur_str.ljust(str_width))

        cur_str = "ERI Screening Scheme            : " + get_qq_type(
            self.qq_type)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = "ERI Screening Threshold         : " + \
            "{:.1e}".format(self.eri_thresh)
        self.ostream.print_header(cur_str.ljust(str_width))

        self.ostream.print_blank()
        self.ostream.flush()

    def print_iteration(self, relative_residual_norm, xvs):
        """
        Prints information of the iteration.

        :param relative_residual_norm:
            Relative residual norms.
        :param xvs:
            A list of tuples containing operator component, frequency, and
            property.
        """

        width = 92
        output_header = '*** Iteration:   {} '.format(self.cur_iter + 1)
        output_header += '* Residuals (Max,Min): '
        output_header += '{:.2e} and {:.2e}'.format(
            max(relative_residual_norm.values()),
            min(relative_residual_norm.values()))
        self.ostream.print_header(output_header.ljust(width))
        self.ostream.print_blank()
        for op, freq, xv in xvs:
            ops_label = '<<{};{}>>_{:.4f}'.format(op, op, freq)
            rel_res = relative_residual_norm[(op, freq)]
            output_iter = '{:<15s}: {:15.8f} '.format(ops_label, -xv)
            output_iter += 'Residual Norm: {:.8f}'.format(rel_res)
            self.ostream.print_header(output_iter.ljust(width))
        self.ostream.print_blank()
        self.ostream.flush()

    def print_convergence(self):
        """
        Prints information after convergence.
        """

        width = 92
        output_conv = '*** '
        if self.is_converged:
            output_conv += 'Linear response converged'
        else:
            output_conv += 'Linear response NOT converged'
        output_conv += ' in {:d} iterations. '.format(self.cur_iter + 1)
        output_conv += 'Time: {:.2f} sec'.format(tm.time() - self.start_time)
        self.ostream.print_header(output_conv.ljust(width))
        self.ostream.print_blank()

    def check_convergence(self, relative_residual_norm):
        """
        Checks convergence.

        :param relative_residual_norm:
            Relative residual norms.
        """

        if self.rank == mpi_master():
            max_residual = max(relative_residual_norm.values())
            if max_residual < self.conv_thresh:
                self.is_converged = True

        self.is_converged = self.comm.bcast(self.is_converged,
                                            root=mpi_master())

    def initial_guess(self, v1, freqs, precond):
        """
        Creating initial guess for the linear response solver.

        :param v1:
            The dictionary containing (operator, frequency) as keys and
            right-hand sides as values.
        :param freq:
            The frequencies.
        :param precond:
            The preconditioner.

        :return:
            The initial guess.
        """

        ig = {}
        for (op, w), grad in v1.items():
            gradger, gradung = self.decomp_grad(grad)

            grad = np.array([gradger, gradung]).flatten()
            gn = np.linalg.norm(grad) * np.sqrt(2.0)

            if gn < self.small_thresh:
                ig[(op, w)] = np.zeros(grad.shape[0])
            else:
                ig[(op, w)] = self.preconditioning(precond[w], grad)

        return ig

    def decomp_grad(self, grad):
        """
        Decomposes gradient into gerade and ungerade parts.

        :param grad:
            The gradient.

        :return:
            A tuple containing gerade and ungerade parts of gradient.
        """

        assert_msg_critical(
            len(grad.shape) == 1, 'decomp_grad: Expecting a 1D array')

        assert_msg_critical(grad.shape[0] % 2 == 0,
                            'decomp_grad: size of array should be even')

        half_size = grad.shape[0] // 2

        grad_T = np.zeros(grad.shape)
        grad_T[:half_size] = grad[half_size:]
        grad_T[half_size:] = grad[:half_size]

        ger = 0.5 * (grad + grad_T)[:half_size]
        ung = 0.5 * (grad - grad_T)[:half_size]

        return ger.T, ung.T

    def get_precond(self, orb_ene, nocc, norb, w):
        """
        Constructs the preconditioner matrix.

        :param orb_ene:
            The orbital energies.
        :param nocc:
            The number of doubly occupied orbitals.
        :param norb:
            The number of orbitals.
        :param w:
            The frequency.

        :return:
            The preconditioner matrix.
        """

        # spawning needed components

        ediag, sdiag = construct_ed_sd_half(orb_ene, nocc, norb)

        ediag_sq = ediag**2
        sdiag_sq = sdiag**2
        w_sq = w**2

        # constructing matrix block diagonals

        pa_diag = ediag / (ediag_sq - w_sq * sdiag_sq)
        pb_diag = (w * sdiag) / (ediag_sq - w_sq * sdiag_sq)

        precond = np.array([pa_diag, pb_diag])

        return precond

    def preconditioning(self, precond, v_in):
        """
        Creates trial vectors out of residuals and the preconditioner matrix.

        :param precond:
            The preconditioner matrix.
        :param v_in:
            The input trial vectors.

        :return:
            The trail vectors after preconditioning.
        """

        pa, pb = precond[0], precond[1]

        v_in_rg, v_in_ru = self.decomp_trials(v_in)

        v_out_rg = pa * v_in_rg + pb * v_in_ru
        v_out_ru = pb * v_in_rg + pa * v_in_ru

        v_out = np.array([v_out_rg, v_out_ru]).flatten()

        return v_out

    def setup_trials(self,
                     vectors,
                     precond=None,
                     bger=None,
                     bung=None,
                     renormalize=True):
        """
        Computes orthonormalized trial vectors.

        :param vectors:
            The set of vectors.
        :param precond:
            The preconditioner.
        :param bger:
            The gerade subspace.
        :param bung:
            The ungerade subspace.
        :param renormalize:
            The flag for normalization.

        :return:
            The orthonormalized gerade and ungerade trial vectors.
        """

        trials = []

        for (op, freq) in vectors:
            vec = vectors[(op, freq)]

            if precond is not None:
                v = self.preconditioning(precond[freq], vec)
            else:
                v = vec

            if np.linalg.norm(v) * np.sqrt(2.0) > self.small_thresh:
                trials.append(v)

        new_trials = np.array(trials).T

        # decomposing the full space trial vectors...

        new_ger, new_ung = self.decomp_trials(new_trials)

        if bger is not None and bger.any():
            new_ger_proj = np.matmul(bger, 2.0 * np.matmul(bger.T, new_ger))
            new_ger = new_ger - new_ger_proj

        if bung is not None and bung.any():
            new_ung_proj = np.matmul(bung, 2.0 * np.matmul(bung.T, new_ung))
            new_ung = new_ung - new_ung_proj

        if new_ger.any() and renormalize:
            new_ger = remove_linear_dependence_half(new_ger, self.lindep_thresh)
            new_ger = orthogonalize_gram_schmidt_half(new_ger)
            new_ger = normalize_half(new_ger)

        if new_ung.any() and renormalize:
            new_ung = remove_linear_dependence_half(new_ung, self.lindep_thresh)
            new_ung = orthogonalize_gram_schmidt_half(new_ung)
            new_ung = normalize_half(new_ung)

        return new_ger, new_ung

    def decomp_trials(self, vecs):
        """
        Decomposes trial vectors into gerade and ungerade parts.

        :param vecs:
            The trial vectors.

        :return:
            A tuple containing gerade and ungerade parts of the trial vectors.
        """

        assert_msg_critical(vecs.shape[0] % 2 == 0,
                            'decomp_trials: shape[0] of array should be even')

        ger, ung = None, None
        half_rows = vecs.shape[0] // 2

        if len(vecs.shape) == 1:
            ger = vecs[:half_rows]
            ung = vecs[half_rows:]

        elif len(vecs.shape) == 2:
            ger = vecs[:half_rows, :]
            ung = vecs[half_rows:, :]

        return ger, ung

    def print_timing(self):
        """
        Prints timing for the linear response eigensolver.
        """

        width = 92

        valstr = 'Timing (in sec):'
        self.ostream.print_header(valstr.ljust(width))
        self.ostream.print_header(('-' * len(valstr)).ljust(width))

        valstr = '{:<15s} {:>15s} {:>18s}'.format('', 'ReducedSpace',
                                                  'NewTrialVectors')
        self.ostream.print_header(valstr.ljust(width))

        for i, (a, b) in enumerate(
                zip(self.timing_dict['reduced_space'],
                    self.timing_dict['new_trials'])):
            if i == 0:
                title = 'Initial guess'
            else:
                title = 'Iteration {:<5d}'.format(i)
            valstr = '{:<15s} {:15.3f} {:18.3f}'.format(title, a, b)
            self.ostream.print_header(valstr.ljust(width))

        valstr = '---------'
        self.ostream.print_header(valstr.ljust(width))

        valstr = '{:<15s} {:15.3f} {:18.3f}'.format(
            'Sum', sum(self.timing_dict['reduced_space']),
            sum(self.timing_dict['new_trials']))
        self.ostream.print_header(valstr.ljust(width))

        self.ostream.print_blank()

    def print_rsp_functions(self, rsp_funcs):

        width = 92

        title = 'Response Functions at Given Frequencies'
        self.ostream.print_header(title.ljust(width))
        self.ostream.print_header(('=' * len(title)).ljust(width))
        self.ostream.print_blank()

        for w in self.frequencies:
            title = '{:<7s} {:<7s} {:>10s} {:>15s}'.format(
                'Dipole', 'Dipole', 'Frequency', 'Value')
            self.ostream.print_header(title.ljust(width))
            self.ostream.print_header(('-' * len(title)).ljust(width))

            for a in self.a_components:
                for b in self.b_components:
                    prop = rsp_funcs[(a, b, w)]
                    ops_label = '<<{:>3s}  ;  {:<3s}>> {:10.4f}'.format(
                        a.lower(), b.lower(), w)
                    output = '{:<15s} {:15.8f}'.format(ops_label, prop)
                    self.ostream.print_header(output.ljust(width))
            self.ostream.print_blank()

        self.ostream.print_blank()
        self.ostream.flush()
