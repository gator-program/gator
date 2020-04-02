import numpy as np
import time as tm
from veloxchem import AODensityMatrix
from veloxchem import AOFockMatrix
from veloxchem import MolecularOrbitals
from veloxchem import ElectronRepulsionIntegralsDriver
from veloxchem import mpi_master
from veloxchem import denmat
from veloxchem.veloxchemlib import fockmat
from veloxchem import molorb
from veloxchem import MOIntegralsDriver
from veloxchem import SubCommunicators
from veloxchem import get_qq_scheme
from veloxchem import get_qq_type
from veloxchem import assert_msg_critical


class Mp2Driver:
    """
    Implements MP2 driver.

    :param comm:
        The MPI communicator.
    :param ostream:
        The output stream.

    Instance variable
        - e_mp2: The MP2 correlation energy.
        - comm: The MPI communicator.
        - rank: The MPI rank.
        - nodes: Number of MPI processes.
        - qq_type: The electron repulsion integrals screening scheme.
        - eri_thresh: The electron repulsion integrals screening threshold.
        - batch_size: The number of Fock matrices in each batch.
        - comm_size: The size of each subcommunicator.
        - ostream: The output stream.
        - conventional: The flag for using conventional (in-memory) AO-to-MO
          integral transformation.
    """

    def __init__(self, comm, ostream):
        """
        Initializes MP2 driver.
        """

        self.e_mp2 = None

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

        # use conventional (in-memory) AO-to-MO integral transformation?
        self.conventional = False

    def update_settings(self, mp2_dict, scf_drv=None):
        """
        Updates settings in MP2 driver.

        :param mp2_dict:
            The dictionary of MP2 settings.
        :param scf_drv:
            The scf driver.
        """

        if 'qq_type' in mp2_dict:
            self.qq_type = mp2_dict['qq_type'].upper()
        elif scf_drv is not None:
            # inherit from SCF
            self.qq_type = scf_drv.qq_type

        if 'eri_thresh' in mp2_dict:
            self.eri_thresh = float(mp2_dict['eri_thresh'])
        elif scf_drv is not None:
            # inherit from SCF
            self.eri_thresh = scf_drv.eri_thresh

        if 'batch_size' in mp2_dict:
            self.batch_size = int(mp2_dict['batch_size'])
        if 'comm_size' in mp2_dict:
            self.comm_size = int(mp2_dict['comm_size'])
            if self.nodes % self.comm_size != 0:
                self.comm_size = 1

        if 'conventional' in mp2_dict:
            key = mp2_dict['conventional'].lower()
            self.conventional = True if key in ['yes', 'y'] else False

    def compute(self, molecule, ao_basis, scf_tensors):
        """
        Performs MP2 calculation.

        :param molecule:
            The molecule.
        :param ao_basis:
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

        if self.conventional:
            self.compute_conventional(molecule, ao_basis, mol_orbs)
        else:
            self.compute_distributed(molecule, ao_basis, mol_orbs)

    def compute_conventional(self, molecule, ao_basis, mol_orbs):
        """
        Performs conventional MP2 calculation.

        :param molecule:
            The molecule.
        :param ao_basis:
            The AO basis set.
        :param mol_orbs:
            The molecular orbitals.
        """

        moints_drv = MOIntegralsDriver(self.comm, self.ostream)

        if self.rank == mpi_master():

            orb_ene = mol_orbs.ea_to_numpy()
            nocc = molecule.number_of_alpha_electrons()
            eocc = orb_ene[:nocc]
            evir = orb_ene[nocc:]
            eab = evir.reshape(-1, 1) + evir

            self.e_mp2 = 0.0
            oovv = moints_drv.compute_in_mem(molecule, ao_basis, mol_orbs,
                                             "OOVV")
            for i in range(oovv.shape[0]):
                for j in range(oovv.shape[1]):
                    ij = oovv[i, j, :, :]
                    ij_antisym = ij - ij.T
                    denom = eocc[i] + eocc[j] - eab
                    self.e_mp2 += np.sum(ij * (ij + ij_antisym) / denom)

            mp2_str = '*** MP2 correlation energy: %20.12f a.u.' % self.e_mp2
            self.ostream.print_header(mp2_str.ljust(92))
            self.ostream.print_blank()

    def compute_distributed(self, molecule, basis, mol_orbs):
        """
        Performs MP2 calculation via distributed Fock builds.

        :param molecule:
            The molecule.
        :param basis:
            The AO basis set.
        :param mol_orbs:
            The molecular orbitals.
        """

        # subcommunicators

        if self.rank == mpi_master():
            assert_msg_critical(self.nodes % self.comm_size == 0,
                                'MP2 driver: invalid size of subcommunicator')

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

        # prepare MO integrals

        if local_master:
            e_mp2 = 0.0
            mol_orbs.broadcast(cross_comm.Get_rank(), cross_comm)
            nocc = molecule.number_of_alpha_electrons()

            mo = mol_orbs.alpha_to_numpy()
            mo_occ = mo[:, :nocc]
            mo_vir = mo[:, nocc:]

            orb_ene = mol_orbs.ea_to_numpy()
            evir = orb_ene[nocc:]
            eab = evir.reshape(-1, 1) + evir

            mo_ints_ids = [
                (i, j) for i in range(nocc) for j in range(i + 1, nocc)
            ] + [(i, i) for i in range(nocc)]
            self.print_header(len(mo_ints_ids))

            local_indices = mo_ints_ids[cross_rank::cross_nodes]
            local_count = len(local_indices)

            num_batches = local_count // self.batch_size
            if local_count % self.batch_size != 0:
                num_batches += 1
        else:
            num_batches = None

        num_batches = local_comm.bcast(num_batches, root=mpi_master())

        # compute MO integrals in batches

        valstr = 'Processing Fock builds for MP2... '
        self.ostream.print_info(valstr)

        start_time = tm.time()

        for batch_ind in range(num_batches):

            valstr = '  batch {}/{}'.format(batch_ind + 1, num_batches)
            self.ostream.print_info(valstr)
            self.ostream.flush()

            if local_master:

                batch_start = batch_ind * self.batch_size
                batch_end = min(batch_start + self.batch_size, local_count)
                batch_ids = local_indices[batch_start:batch_end]

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
                fock.set_fock_type(fockmat.rgenk, i)

            eri_drv.compute(fock, dens, molecule, basis, screening)
            fock.reduce_sum(local_comm.Get_rank(), local_comm.Get_size(),
                            local_comm)

            if local_master:
                for ind, (i, j) in enumerate(batch_ids):
                    f_ao = fock.alpha_to_numpy(ind)
                    f_vv = np.linalg.multi_dot([mo_vir.T, f_ao, mo_vir])
                    eijab = orb_ene[i] + orb_ene[j] - eab

                    ab = f_vv
                    e_mp2 += np.sum(ab * (2.0 * ab - ab.T) / eijab)

                    if i != j:
                        ba = f_vv.T
                        e_mp2 += np.sum(ba * (2.0 * ba - ba.T) / eijab)

        if local_master:
            dt = tm.time() - start_time
            dt = cross_comm.gather(dt, root=mpi_master())
            load_imb = 0.0
            if global_master:
                load_imb = 1.0 - min(dt) / max(dt)

        self.ostream.print_blank()
        valstr = 'Fock builds for MP2 done in '
        valstr += '{:.2f} sec.'.format(tm.time() - start_time)
        valstr += ' Load imb.: {:.1f} %'.format(load_imb * 100.0)
        self.ostream.print_info(valstr)
        self.ostream.print_blank()

        if local_master:
            e_mp2 = cross_comm.reduce(e_mp2, root=mpi_master())

        if global_master:
            self.e_mp2 = e_mp2
            mp2_str = '*** MP2 correlation energy: {:20.12f} a.u. '.format(
                self.e_mp2)
            self.ostream.print_header(mp2_str.ljust(80))
            self.ostream.print_blank()
            self.ostream.flush()

    def print_header(self, num_matrices):
        """
        Prints header for the MP2 driver.

        :param num_matrices:
            The number of Fock matrices to be computed.
        """

        self.ostream.print_blank()
        self.ostream.print_header("MP2 Driver Setup")
        self.ostream.print_header(18 * "=")
        self.ostream.print_blank()

        str_width = 60
        cur_str = "Number of Fock Matrices      : " + str(num_matrices)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = "Size of Fock Matrices Batch  : " + str(self.batch_size)
        self.ostream.print_header(cur_str.ljust(str_width))

        cur_str = "Number of Subcommunicators   : "
        cur_str += str(self.nodes // self.comm_size)
        self.ostream.print_header(cur_str.ljust(str_width))

        cur_str = "ERI Screening Scheme         : " + get_qq_type(self.qq_type)
        self.ostream.print_header(cur_str.ljust(str_width))
        cur_str = "ERI Screening Threshold      : " + \
            "{:.1e}".format(self.eri_thresh)
        self.ostream.print_header(cur_str.ljust(str_width))
        self.ostream.print_blank()
        self.ostream.flush()
