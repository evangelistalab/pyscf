#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Sheng Guo
#          Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib

class DSRG_MRPT2(lib.StreamObject):
    '''
    DSRG-MRPT2

    Attributes:
        root : int or list of ints (default: 0)
            To control which state to compute if multiple roots or state-average
            wfn were calculated in CASCI/CASSCF. If list of ints, then state-averaged
            DSRG-MRPT2 is performed of the given list of states.
        s : float (default: 0.5)
            The flow parameter, which controls the extent to which 
            the Hamiltonian is block-diagonalized.
        relax : str (default: 'none')
            Reference relaxation method. Options: 'none', 'once', 'twice', 'iterate'.

    Examples:

    >>> mf = gto.M('N 0 0 0; N 0 0 1.4', basis='6-31g').apply(scf.RHF).run()
    >>> mc = mcscf.CASCI(mf, 4, 4).run()
    >>> DSRG_MRPT2(mc, s=0.5).kernel()
    -0.15708345625685638
    '''
    def __init__(self, mc, root=0, s=0.5, relax='none'):
        self.mc = mc
        self.root = root
        self.flow_param = s
        self.relax = relax

        self.ncore = mc.ncore
        self.nact  = mc.ncas
        self.nvirt = mc.mol.nao - self.nact - self.ncore
        self.flow_param = s

        self.e_corr = None
        self.semi_canonicalized = False

        

    def kernel(self):
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    mol = gto.M(
        atom = '''
    N 0 0 0
    N 0 1.4 0
    ''',
        basis = '6-31g', spin=0, charge=0, symmetry=False
    )

    rhf = scf.RHF(mol)
    rhf.kernel()
    casci = mcscf.CASCI(rhf, 4, 8)
    casci.kernel()
    e_dsrg_mrpt2 = DSRG_MRPT2(casci).kernel()
    assert numpy.isclose(e_dsrg_mrpt2, -0.15708345625685638)
