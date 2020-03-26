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
            mo_ints_ids = [(i, j) for i in range(nocc) for j in range(nocc)]

            ave, res = divmod(len(mo_ints_ids), cross_nodes)
            count = [ave + 1 if i < res else ave for i in range(cross_nodes)]
            displ = [sum(count[:i]) for i in range(cross_nodes)]

            mo_ints_start = displ[cross_rank]
            mo_ints_end = mo_ints_start + count[cross_rank]

            oo_indices = mo_ints_ids[mo_ints_start:mo_ints_end]
            ovov = np.zeros((count[cross_rank], nvir * nvir))
            oovv = np.zeros((count[cross_rank], nvir * nvir))
            ooov = np.zeros((count[cross_rank], nocc * nvir))

            num_batches = count[cross_rank] // self.batch_size
            if count[cross_rank] % self.batch_size != 0:
                num_batches += 1
        else:
            num_batches = None

        num_batches = local_comm.bcast(num_batches, root=mpi_master())

        for batch_ind in range(num_batches):

            if local_master:
                batch_start = mo_ints_start + batch_ind * self.batch_size
                batch_end = min(batch_start + self.batch_size, mo_ints_end)
                batch_ids = mo_ints_ids[batch_start:batch_end]

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
                    ovov[i + batch_ind * self.batch_size, :] = f_vv.reshape(
                        nvir * nvir)[:]

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
                    oovv[i + batch_ind * self.batch_size, :] = f_vv.reshape(
                        nvir * nvir)[:]
                    ooov[i + batch_ind * self.batch_size, :] = f_ov.reshape(
                        nocc * nvir)[:]

        valstr = 'Integrals transformation for the OO block done in '
        valstr += '{:.2f} sec.'.format(tm.time() - start_time)
        self.ostream.print_info(valstr)
        self.ostream.print_blank()

        # compute VV blocks: <VV|OO>, <VV|OV>

        start_time = tm.time()

        if local_master:
            mo_ints_ids = [(a, b) for a in range(nvir) for b in range(nvir)]

            ave, res = divmod(len(mo_ints_ids), cross_nodes)
            count = [ave + 1 if i < res else ave for i in range(cross_nodes)]
            displ = [sum(count[:i]) for i in range(cross_nodes)]

            mo_ints_start = displ[cross_rank]
            mo_ints_end = mo_ints_start + count[cross_rank]

            vv_indices = mo_ints_ids[mo_ints_start:mo_ints_end]
            vvoo = np.zeros((count[cross_rank], nocc * nocc))
            vvov = np.zeros((count[cross_rank], nocc * nvir))

            num_batches = count[cross_rank] // self.batch_size
            if count[cross_rank] % self.batch_size != 0:
                num_batches += 1
        else:
            num_batches = None

        num_batches = local_comm.bcast(num_batches, root=mpi_master())

        for batch_ind in range(num_batches):

            if local_master:
                batch_start = mo_ints_start + batch_ind * self.batch_size
                batch_end = min(batch_start + self.batch_size, mo_ints_end)
                batch_ids = mo_ints_ids[batch_start:batch_end]

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
                    vvoo[i + batch_ind * self.batch_size, :] = f_oo.reshape(
                        nocc * nocc)[:]
                    vvov[i + batch_ind * self.batch_size, :] = f_ov.reshape(
                        nocc * nvir)[:]

        valstr = 'Integrals transformation for the VV block done in '
        valstr += '{:.2f} sec.'.format(tm.time() - start_time)
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
