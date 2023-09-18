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

import unittest
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import fci
from pyscf.mrpt import dsrg_mrpt2

def setUpModule():
    global mol, mf, mc
    mol = gto.M(
        atom = '''
        N 0 0 0
        N 0 1.4 0
        ''',
        basis = '6-31g', spin=0, charge=0, verbose=5, output='/dev/null'
    )
    mf = scf.RHF(mol)
    mf.kernel()
    mc = mcscf.CASCI(mf, 6, 6)
    mc.fcisolver.conv_tol = 1e-15
    mc.kernel()

def tearDownModule():
    global mol, mf, mc
    mol.stdout.close()
    del mol, mf, mc

class KnownValues(unittest.TestCase):
    def test_energy(self):
        e = dsrg_mrpt2.DSRG_MRPT2(mc).kernel()
        self.assertAlmostEqual(e, -0.127274453305632, delta=1.0e-6)

if __name__ == "__main__":
    print("Full Tests for DSRG-MRPT2")
    unittest.main()