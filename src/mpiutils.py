from mpi4py import MPI
from veloxchem import assert_msg_critical
from veloxchem import mpi_initialized
from veloxchem import mpi_master
import sys
import os


def mpi_sanity_check(argv):

    assert_msg_critical(mpi_initialized(), "MPI: Initialized")

    if len(argv) <= 1 or argv[1] in ['-h', '--help']:
        info_txt = [
            '',
            '=================   GATOR   =================',
            '',
            'Usage:',
            '    python3 -m gator input_file [output_file]',
            '',
        ]
        if MPI.COMM_WORLD.Get_rank() == mpi_master():
            print(os.linesep.join(info_txt), file=sys.stdout)
        sys.exit(0)
