from veloxchem import AODensityMatrix
from veloxchem import AOFockMatrix
from veloxchem import ElectronRepulsionIntegralsDriver
from veloxchem import mpi_master
from veloxchem import denmat
from veloxchem.veloxchemlib import fockmat
from veloxchem import SubCommunicators
from veloxchem import get_qq_scheme
from veloxchem import assert_msg_critical
import numpy as np
import time as tm


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
        - comm_size: The size of each subcommunicator.
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

        # screening scheme and batch size for Fock build
        self.qq_type = 'QQ_DEN'
        self.eri_thresh = 1.0e-12
        self.batch_size = 100

        # size of subcommunicator
        self.comm_size = 1

        # output stream
        self.ostream = ostream

    def update_settings(self, settings):
        """
        Updates settings in MO integrals driver.

        :param settings:
            The dictionary of MO integrals settings.
        """

        if 'qq_type' in settings:
            self.qq_type = settings['qq_type']
        if 'eri_thresh' in settings:
            self.eri_thresh = float(settings['eri_thresh'])

        if 'batch_size' in settings:
            self.batch_size = int(settings['batch_size'])
        if 'comm_size' in settings:
            self.comm_size = int(settings['comm_size'])
            if self.nodes % self.comm_size != 0:
                self.comm_size = 1

    def compute(self, molecule, basis, scf_tensors):
        """
        Performs MO integral calculation.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param scf_tensors:
            The dictionary of tensors from converged SCF wavefunction.
        """

        # subcommunicators

        if self.rank == mpi_master():
            assert_msg_critical(
                self.nodes % self.comm_size == 0,
                'MO integral driver: invalid size of subcommunicator')

        grps = [p // self.comm_size for p in range(self.nodes)]
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

        # compute OO blocks: <OV|OV>, <OO|VV>, <OO|OV>

        start_time = tm.time()

        if local_master:
            # only computes ij pairs where i <= j
            mo_ints_ids = [
                (i, j) for i in range(nocc) for j in range(i + 1, nocc)
            ] + [(i, i) for i in range(nocc)]

            oo_indices = mo_ints_ids[cross_rank::cross_nodes]
            oo_count = len(oo_indices)

            ovov = []
            oovv = []
            ooov = []

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

            if local_master:
                batch_start = batch_ind * self.batch_size
                batch_end = min(batch_start + self.batch_size, oo_count)
                batch_ids = oo_indices[batch_start:batch_end]

                dks = []
                for i, j in batch_ids:
                    mo_ij = np.zeros((nocc, nocc))
                    mo_ij[i, j] = 1.0
                    dks.append(np.linalg.multi_dot([mo_occ, mo_ij, mo_occ.T]))
                dens = AODensityMatrix(dks, denmat.rest)
            else:
                dens = AODensityMatrix()

            dens.broadcast(local_comm.Get_rank(), local_comm)

            fock = AOFockMatrix(dens)
            for i in range(fock.number_of_fock_matrices()):
                fock.set_fock_type(fockmat.rgenj, i)

            eri_drv.compute(fock, dens, molecule, basis, screening)
            fock.reduce_sum(local_comm.Get_rank(), local_comm.Get_size(),
                            local_comm)

            if local_master:
                for i in range(fock.number_of_fock_matrices()):
                    f_ao = fock.alpha_to_numpy(i)
                    f_vv = np.linalg.multi_dot([mo_vir.T, f_ao, mo_vir])
                    ovov.append(f_vv)
                    pair = oo_indices[i + batch_start]
                    if pair[0] != pair[1]:
                        ovov.append(f_vv.T)

            fock = AOFockMatrix(dens)
            for i in range(fock.number_of_fock_matrices()):
                fock.set_fock_type(fockmat.rgenk, i)

            eri_drv.compute(fock, dens, molecule, basis, screening)
            fock.reduce_sum(local_comm.Get_rank(), local_comm.Get_size(),
                            local_comm)

            if local_master:
                for i in range(fock.number_of_fock_matrices()):
                    f_ao = fock.alpha_to_numpy(i)
                    f_vv = np.linalg.multi_dot([mo_vir.T, f_ao, mo_vir])
                    f_ov = np.linalg.multi_dot([mo_occ.T, f_ao, mo_vir])
                    oovv.append(f_vv)
                    ooov.append(f_ov)
                    pair = oo_indices[i + batch_start]
                    if pair[0] != pair[1]:
                        f_vo = np.linalg.multi_dot([mo_vir.T, f_ao, mo_occ])
                        oovv.append(f_vv.T)
                        ooov.append(f_vo.T)

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

        # compute VV blocks: <VV|OO>, <VV|OV>

        start_time = tm.time()

        if local_master:
            # only computes ab pairs where a <= b
            mo_ints_ids = [
                (a, b) for a in range(nvir) for b in range(a + 1, nvir)
            ] + [(a, a) for a in range(nvir)]

            vv_indices = mo_ints_ids[cross_rank::cross_nodes]
            vv_count = len(vv_indices)

            vvoo = []
            vvov = []

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

            if local_master:
                batch_start = batch_ind * self.batch_size
                batch_end = min(batch_start + self.batch_size, vv_count)
                batch_ids = vv_indices[batch_start:batch_end]

                dks = []
                for a, b in batch_ids:
                    mo_ab = np.zeros((nvir, nvir))
                    mo_ab[a, b] = 1.0
                    dks.append(np.linalg.multi_dot([mo_vir, mo_ab, mo_vir.T]))
                dens = AODensityMatrix(dks, denmat.rest)
            else:
                dens = AODensityMatrix()

            dens.broadcast(local_comm.Get_rank(), local_comm)

            fock = AOFockMatrix(dens)
            for i in range(fock.number_of_fock_matrices()):
                fock.set_fock_type(fockmat.rgenk, i)

            eri_drv.compute(fock, dens, molecule, basis, screening)
            fock.reduce_sum(local_comm.Get_rank(), local_comm.Get_size(),
                            local_comm)

            if local_master:
                for i in range(fock.number_of_fock_matrices()):
                    f_ao = fock.alpha_to_numpy(i)
                    f_oo = np.linalg.multi_dot([mo_occ.T, f_ao, mo_occ])
                    f_ov = np.linalg.multi_dot([mo_occ.T, f_ao, mo_vir])
                    vvoo.append(f_oo)
                    vvov.append(f_ov)
                    pair = vv_indices[i + batch_start]
                    if pair[0] != pair[1]:
                        f_vo = np.linalg.multi_dot([mo_vir.T, f_ao, mo_occ])
                        vvoo.append(f_oo.T)
                        vvov.append(f_vo.T)

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

        return {
            'oo_indices': oo_indices,
            'ovov': ovov,
            'oovv': oovv,
            'ooov': ooov,
            'vv_indices': vv_indices,
            'vvoo': vvoo,
            'vvov': vvov,
        }
