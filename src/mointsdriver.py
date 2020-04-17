import numpy as np
import time as tm

from veloxchem import AODensityMatrix
from veloxchem import AOFockMatrix
from veloxchem import ElectronRepulsionIntegralsDriver
from veloxchem import SubCommunicators
from veloxchem import mpi_master
from veloxchem import denmat
from veloxchem.veloxchemlib import fockmat
from veloxchem import get_qq_scheme


class MOIntegralsDriver:
    """
    Implements MO integrals driver.

    :param comm:
        The MPI communicator.
    :param ostream:
        The output stream.

    Instance variable
        - comm: The MPI communicator.
        - rank: The MPI rank.
        - nodes: Number of MPI processes.
        - qq_type: The electron repulsion integrals screening scheme.
        - eri_thresh: The electron repulsion integrals screening threshold.
        - batch_size: The number of Fock matrices in each batch.
        - ostream: The output stream.
    """

    def __init__(self, comm, ostream):
        """
        Initializes MO integrals driver.
        """

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

        # screening scheme and batch size for Fock build
        self.qq_type = 'QQ_DEN'
        self.eri_thresh = 1.0e-12
        self.batch_size = 100

    def update_settings(self, settings):
        """
        Updates settings in MO integrals driver.

        :param settings:
            The dictionary of MO integrals settings.
        """

        if 'qq_type' in settings:
            self.qq_type = settings['qq_type'].upper()
        if 'eri_thresh' in settings:
            self.eri_thresh = float(settings['eri_thresh'])
        if 'batch_size' in settings:
            self.batch_size = int(settings['batch_size'])

    def compute(self, molecule, basis, scf_tensors, blocks=['oo', 'vv', 'ov']):
        """
        Performs MO integral calculation.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param scf_tensors:
            The dictionary of tensors from converged SCF wavefunction.
        :param blocks:
            The MO integral blocks (oo, vv, ov).
        :return:
            A dictionary of indices and MO integrals.
        """

        if self.rank == mpi_master():
            self.print_header()

        # subcommunicators

        grps = [p for p in range(self.nodes)]
        subcomm = SubCommunicators(self.comm, grps)
        local_comm = subcomm.local_comm
        cross_comm = subcomm.cross_comm

        cross_rank = cross_comm.Get_rank()
        cross_nodes = cross_comm.Get_size()

        local_master = (local_comm.Get_rank() == mpi_master())
        global_master = (self.rank == mpi_master())

        # screening data

        eri_drv = ElectronRepulsionIntegralsDriver(local_comm)
        screening = eri_drv.compute(get_qq_scheme(self.qq_type),
                                    self.eri_thresh, molecule, basis)

        # prepare molecular orbitals

        if local_master:
            if global_master:
                mo = scf_tensors['C']
            else:
                mo = None
            mo = cross_comm.bcast(mo, root=mpi_master())

            nocc = molecule.number_of_alpha_electrons()
            norb = mo.shape[1]
            nvir = norb - nocc

            mo_occ = mo[:, :nocc]
            mo_vir = mo[:, nocc:]

        # compute OO blocks: (OO|VV), (OV|OV), (OO|OV)
        #                     **       *  *     *  *
        if 'oo' in blocks:

            start_time = tm.time()

            if local_master:
                # only computes ij pairs where i <= j
                mo_ints_ids = [
                    (i, j) for i in range(nocc) for j in range(i + 1, nocc)
                ] + [(i, i) for i in range(nocc)]

                oo_indices = mo_ints_ids[cross_rank::cross_nodes]
                oo_count = len(oo_indices)

                chem_oovv_J = []
                chem_ovov_K = []
                chem_ooov_K = []

                num_batches = oo_count // self.batch_size
                if oo_count % self.batch_size != 0:
                    num_batches += 1
            else:
                num_batches = None

            num_batches = local_comm.bcast(num_batches, root=mpi_master())

            valstr = 'Processing Fock builds for the OO block... '
            self.ostream.print_info(valstr)

            for batch_ind in range(num_batches):

                valstr = '  batch {}/{}'.format(batch_ind + 1, num_batches)
                self.ostream.print_info(valstr)
                self.ostream.flush()

                dens = self.form_densities(oo_indices, batch_ind, nocc, nocc,
                                           mo_occ, mo_occ, local_comm)

                fock = self.comp_fock(dens, fockmat.rgenj, molecule, basis,
                                      screening, eri_drv, local_comm)

                if local_master:
                    for i in range(fock.number_of_fock_matrices()):
                        f_ao = fock.alpha_to_numpy(i)
                        f_vv = np.linalg.multi_dot([mo_vir.T, f_ao, mo_vir])
                        chem_oovv_J.append(f_vv)
                        pair = oo_indices[i + batch_ind * self.batch_size]
                        if pair[0] != pair[1]:
                            chem_oovv_J.append(f_vv.T)

                fock = self.comp_fock(dens, fockmat.rgenk, molecule, basis,
                                      screening, eri_drv, local_comm)

                if local_master:
                    for i in range(fock.number_of_fock_matrices()):
                        f_ao = fock.alpha_to_numpy(i)
                        f_vv = np.linalg.multi_dot([mo_vir.T, f_ao, mo_vir])
                        f_ov = np.linalg.multi_dot([mo_occ.T, f_ao, mo_vir])
                        chem_ovov_K.append(f_vv)
                        chem_ooov_K.append(f_ov)
                        pair = oo_indices[i + batch_ind * self.batch_size]
                        if pair[0] != pair[1]:
                            f_vo = np.linalg.multi_dot([mo_vir.T, f_ao, mo_occ])
                            chem_ovov_K.append(f_vv.T)
                            chem_ooov_K.append(f_vo.T)

            if local_master:
                dt = tm.time() - start_time
                dt = cross_comm.gather(dt, root=mpi_master())
                load_imb = 0.0
                if global_master:
                    load_imb = 1.0 - min(dt) / max(dt)

            # make full ij pairs
            oo_indices_full = []
            for i, j in oo_indices:
                oo_indices_full.append((i, j))
                if i != j:
                    oo_indices_full.append((j, i))
            oo_indices = oo_indices_full

            self.ostream.print_blank()
            valstr = 'Integrals transformation for the OO block done in '
            valstr += '{:.2f} sec.'.format(tm.time() - start_time)
            valstr += ' Load imb.: {:.1f} %'.format(load_imb * 100.0)
            self.ostream.print_info(valstr)
            self.ostream.print_blank()

        # compute VV blocks: (VO|VO), (VO|VV)
        #                     *  *     *  *

        if 'vv' in blocks:

            start_time = tm.time()

            if local_master:
                # only computes ab pairs where a <= b
                mo_ints_ids = [
                    (a, b) for a in range(nvir) for b in range(a + 1, nvir)
                ] + [(a, a) for a in range(nvir)]

                vv_indices = mo_ints_ids[cross_rank::cross_nodes]
                vv_count = len(vv_indices)

                chem_vovo_K = []
                chem_vovv_K = []

                num_batches = vv_count // self.batch_size
                if vv_count % self.batch_size != 0:
                    num_batches += 1
            else:
                num_batches = None

            num_batches = local_comm.bcast(num_batches, root=mpi_master())

            valstr = 'Processing Fock builds for the VV block... '
            self.ostream.print_info(valstr)

            for batch_ind in range(num_batches):

                valstr = '  batch {}/{}'.format(batch_ind + 1, num_batches)
                self.ostream.print_info(valstr)
                self.ostream.flush()

                dens = self.form_densities(vv_indices, batch_ind, nvir, nvir,
                                           mo_vir, mo_vir, local_comm)

                fock = self.comp_fock(dens, fockmat.rgenk, molecule, basis,
                                      screening, eri_drv, local_comm)

                if local_master:
                    for i in range(fock.number_of_fock_matrices()):
                        f_ao = fock.alpha_to_numpy(i)
                        f_oo = np.linalg.multi_dot([mo_occ.T, f_ao, mo_occ])
                        f_ov = np.linalg.multi_dot([mo_occ.T, f_ao, mo_vir])
                        chem_vovo_K.append(f_oo)
                        chem_vovv_K.append(f_ov)
                        pair = vv_indices[i + batch_ind * self.batch_size]
                        if pair[0] != pair[1]:
                            f_vo = np.linalg.multi_dot([mo_vir.T, f_ao, mo_occ])
                            chem_vovo_K.append(f_oo.T)
                            chem_vovv_K.append(f_vo.T)

            if local_master:
                dt = tm.time() - start_time
                dt = cross_comm.gather(dt, root=mpi_master())
                load_imb = 0.0
                if global_master:
                    load_imb = 1.0 - min(dt) / max(dt)

            # make full ab pairs
            vv_indices_full = []
            for a, b in vv_indices:
                vv_indices_full.append((a, b))
                if a != b:
                    vv_indices_full.append((b, a))
            vv_indices = vv_indices_full

            self.ostream.print_blank()
            valstr = 'Integrals transformation for the VV block done in '
            valstr += '{:.2f} sec.'.format(tm.time() - start_time)
            valstr += ' Load imb.: {:.1f} %'.format(load_imb * 100.0)
            self.ostream.print_info(valstr)
            self.ostream.print_blank()

        # compute OV blocks: (OV|OO), (OV|VV), (OO|VO), (OV|VV)
        #                     **       **       *  *     *  *

        if 'ov' in blocks:

            start_time = tm.time()

            if local_master:
                mo_ints_ids = [(i, a) for i in range(nocc) for a in range(nvir)]

                ov_indices = mo_ints_ids[cross_rank::cross_nodes]
                ov_count = len(ov_indices)

                chem_ovoo_J = []
                chem_ovvv_J = []
                chem_oovo_K = []
                chem_ovvv_K = []

                num_batches = ov_count // self.batch_size
                if ov_count % self.batch_size != 0:
                    num_batches += 1
            else:
                num_batches = None

            num_batches = local_comm.bcast(num_batches, root=mpi_master())

            valstr = 'Processing Fock builds for the OV block... '
            self.ostream.print_info(valstr)

            for batch_ind in range(num_batches):

                valstr = '  batch {}/{}'.format(batch_ind + 1, num_batches)
                self.ostream.print_info(valstr)
                self.ostream.flush()

                dens = self.form_densities(ov_indices, batch_ind, nocc, nvir,
                                           mo_occ, mo_vir, local_comm)

                fock = self.comp_fock(dens, fockmat.rgenj, molecule, basis,
                                      screening, eri_drv, local_comm)

                if local_master:
                    for i in range(fock.number_of_fock_matrices()):
                        f_ao = fock.alpha_to_numpy(i)
                        f_oo = np.linalg.multi_dot([mo_occ.T, f_ao, mo_occ])
                        f_vv = np.linalg.multi_dot([mo_vir.T, f_ao, mo_vir])
                        chem_ovoo_J.append(f_oo)
                        chem_ovvv_J.append(f_vv)

                fock = self.comp_fock(dens, fockmat.rgenk, molecule, basis,
                                      screening, eri_drv, local_comm)

                if local_master:
                    for i in range(fock.number_of_fock_matrices()):
                        f_ao = fock.alpha_to_numpy(i)
                        f_oo = np.linalg.multi_dot([mo_occ.T, f_ao, mo_occ])
                        f_vv = np.linalg.multi_dot([mo_vir.T, f_ao, mo_vir])
                        chem_oovo_K.append(f_oo)
                        chem_ovvv_K.append(f_vv)

            if local_master:
                dt = tm.time() - start_time
                dt = cross_comm.gather(dt, root=mpi_master())
                load_imb = 0.0
                if global_master:
                    load_imb = 1.0 - min(dt) / max(dt)

            self.ostream.print_blank()
            valstr = 'Integrals transformation for the OV block done in '
            valstr += '{:.2f} sec.'.format(tm.time() - start_time)
            valstr += ' Load imb.: {:.1f} %'.format(load_imb * 100.0)
            self.ostream.print_info(valstr)
            self.ostream.print_blank()

        # indices and integrals

        indices = {}

        if 'oo' in blocks:
            indices['oo'] = oo_indices
        if 'vv' in blocks:
            indices['vv'] = vv_indices
        if 'ov' in blocks:
            indices['ov'] = ov_indices

        integrals = {}

        if 'oo' in blocks:
            integrals['chem_oovv_J'] = chem_oovv_J
            integrals['chem_ovov_K'] = chem_ovov_K
            integrals['chem_ooov_K'] = chem_ooov_K
        if 'vv' in blocks:
            integrals['chem_vovo_K'] = chem_vovo_K
            integrals['chem_vovv_K'] = chem_vovv_K
        if 'ov' in blocks:
            integrals['chem_ovoo_J'] = chem_ovoo_J
            integrals['chem_ovvv_J'] = chem_ovvv_J
            integrals['chem_oovo_K'] = chem_oovo_K
            integrals['chem_ovvv_K'] = chem_ovvv_K

        return indices, integrals

    def form_densities(self, indices, batch_ind, n_1, n_2, mo_1, mo_2,
                       local_comm):

        if local_comm.Get_rank() == mpi_master():
            batch_start = batch_ind * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(indices))
            batch_ids = indices[batch_start:batch_end]

            dks = []
            for x, y in batch_ids:
                mo_xy = np.zeros((n_1, n_2))
                mo_xy[x, y] = 1.0
                dks.append(np.linalg.multi_dot([mo_1, mo_xy, mo_2.T]))
            dens = AODensityMatrix(dks, denmat.rest)
        else:
            dens = AODensityMatrix()

        dens.broadcast(local_comm.Get_rank(), local_comm)

        return dens

    def comp_fock(self, dens, fock_type, molecule, basis, screening, eri_drv,
                  local_comm):

        fock = AOFockMatrix(dens)
        for i in range(fock.number_of_fock_matrices()):
            fock.set_fock_type(fock_type, i)

        eri_drv.compute(fock, dens, molecule, basis, screening)
        fock.reduce_sum(local_comm.Get_rank(), local_comm.Get_size(),
                        local_comm)

        return fock

    def print_header(self):
        """
        Prints MO integrals driver setup header to output stream.
        """

        title = 'MO Integrals Driver Setup'
        self.ostream.print_header(title)
        self.ostream.print_header('=' * (len(title) + 2))
        self.ostream.print_blank()

        str_width = 60
        cur_str = 'ERI screening scheme        : {:s}'.format(self.qq_type)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = 'ERI Screening Threshold     : {:.1e}'.format(self.eri_thresh)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = 'Batch Size of Fock Matrices : {:d}'.format(self.batch_size)
        self.ostream.print_header(cur_str.ljust(str_width))
        self.ostream.print_blank()
        self.ostream.flush()
