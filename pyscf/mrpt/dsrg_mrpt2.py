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
import tempfile
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import fci
from pyscf.mcscf import mc_ao2mo
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

def F_T1(mc, dms, eris):
    pass

def F_T2(mc, dms, eris):
    pass

def V_T1(mc, dms, eris):
    pass

def V_T2(mc, dms, eris):
    pass

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
        self.nelecas = mc.nelecas # Tuple of (nalpha, nbeta)
        self.nvirt = mc.mol.nao - self.nact - self.ncore
        self.flow_param = s

        self.e_corr = None

    def load_ci(self, root=None):
        if root is None:
            root = self.root
        
        if self.fcisolver.nroots == 1:
            return self.ci
        else:
            return self.ci[root]
        
    def semi_canonicalize(self):
        pass

    def compute_t1(self):
        pass

    def compute_t2(self):
        pass

    def renormalize_F(self):
        pass

    def renormalize_V(self):
        pass

    def kernel(self):
        if isinstance(self.verbose, logger.Logger):
            log = self.verbose
        else:
            log = logger.Logger(self.stdout, self.verbose)
        time0 = (logger.process_clock(), logger.perf_counter())

        self.semi_canonicalize()

        dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf',
                                               self.load_ci(), self.load_ci(), self.nact, self.nelecas)

        dms = {'1': dm1, '2': dm2, '3': dm3}
        time1 = log.timer('RDM generation', *time0)

        eris = NotImplemented
        time1 = log.timer('integral transformation', *time1)

        self.compute_t1()
        self.renormalize_F()
        self.compute_t2()
        self.renormalize_V()

        e_F_T1 = F_T1(self, dms, eris)
        logger.note(self, "<[F,T1]>  ,   E = %.14f", e_F_T1)
        time1 = log.timer('<[F,T1]>', *time1)
        e_F_T2 = F_T2(self, dms, eris)
        logger.note(self, "<[F,T2]>  ,   E = %.14f", e_F_T2)
        time1 = log.timer('<[F,T2]>', *time1)
        e_V_T1 = V_T1(self, dms, eris)
        logger.note(self, "<[V,T1]>  ,   E = %.14f", e_V_T1)
        time1 = log.timer('<[V,T1]>', *time1)
        e_V_T2 = V_T2(self, dms, eris)
        logger.note(self, "<[V,T2]>  ,   E = %.14f", e_V_T2)
        time1 = log.timer('<[V,T2]>', *time1)

        self.e_corr = e_F_T1 + e_F_T2 + e_V_T1 + e_V_T2
        logger.note(self, "DSRG-MRPT2 correlation energy = %.14f", self.e_corr)
        log.timer('DSRG-MRPT2', *time0)

        return self.e_corr

# register DSRG_MRPT2 in MCSCF
# [todo]: is this so that we can access fcisolver options?
from pyscf.mcscf import casci
casci.CASCI.DSRG_MRPT2 = DSRG_MRPT2


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
    casci = mcscf.CASCI(rhf, 6, 6)
    casci.kernel()
    e_dsrg_mrpt2 = DSRG_MRPT2(casci).kernel()
    assert numpy.isclose(e_dsrg_mrpt2, -0.127274453305632)
