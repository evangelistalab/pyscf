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
# Authors:
#          Shuhang Li <shuhangli98@gmail.com>
#          Zijun Zhao <zdj519@gmail.com>
#

import numpy as np
import tempfile
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import fci
from pyscf.mcscf import mc_ao2mo
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

MACHEPS = 1e-9
TAYLOR_THRES = 1e-3

def taylor_exp(z):
    '''
    Taylor expansion of (1-exp(-z^2))/z for small z.
    '''
    n = int(0.5 * (15.0 / TAYLOR_THRES + 1)) + 1
    if (n > 0):
        value = z
        tmp = z
        for x in range(n-1):
            tmp *= -1.0 * z * z / (x + 2)
            value += tmp

        return value
    else:
        return 0.0
    
def regularized_denominator(x, s):
    '''
    Returns (1-exp(-s*x^2))/x
    '''
    z = np.sqrt(s) * x
    if abs(z) <= MACHEPS:
        return taylor_exp(z) * np.sqrt(s)
    else:
        return (1. - np.exp(-s * x**2)) / x
    
def get_SF_RDM(ci_vec, norb, nelec):
    '''
    Returns the spin-free active space 1-/2-/3-RDM.
    Reordered 2-rdm <p\dagger r\dagger s q> in Pyscf is stored as: dm2[pqrs]
    Forte stores it as rdm[prqs]
    '''
    dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', ci_vec, ci_vec, norb, nelec)
    dm1, dm2, dm3 = fci.rdm.reorder_dm123(dm1, dm2, dm3)
    G1 = np.einsum("pq -> qp", dm1)
    G2 = np.einsum("pqrs -> prqs", dm2)
    G3 = np.einsum("pqrstu -> prtqsu", dm3)
    return G1, G2, G3

def get_SF_cu2(G1, G2):
    '''
    Returns the spin-free active space 2-body cumulant.
    '''
    L2 = G2.copy() 
    L2 -= np.einsum("pr, qs->pqrs", G1, G1)
    L2 += 0.5 * np.einsum("ps, qr->pqrs", G1, G1)
    return L2
    
def get_SF_cu3(G1, G2, G3): 
    '''
    Returns the spin-free active space 3-body cumulant.
    '''
    L3 = G3.copy() # PQRSTU
    L3 -= (np.einsum("ps,qrtu -> pqrstu", G1, G2) + np.einsum("qt,prsu -> pqrstu", G1, G2) + np.einsum("ru,pqst -> pqrstu", G1, G2))
    L3 += 0.5 * (np.einsum("pt,qrsu -> pqrstu", G1, G2) + np.einsum("pu,qrts -> pqrstu", G1, G2) + np.einsum("qs,prtu -> pqrstu", G1, G2) + np.einsum("qu,prst -> pqrstu", G1, G2) + np.einsum("rs,pqut -> pqrstu", G1, G2) + np.einsum("rt,pqsu -> pqrstu", G1, G2))
    L3 += 2 * np.einsum("ps, qt, ru -> pqrstu", G1, G1, G1)
    L3 -= (np.einsum("ps, qu, rt -> pqrstu", G1, G1, G1) + np.einsum("pu, qt, rs -> pqrstu", G1, G1, G1) + np.einsum("pt, qs, ru -> pqrstu", G1, G1, G1))
    L3 += 0.5 * (np.einsum("pt, qu, rs -> pqrstu", G1, G1, G1) + np.einsum("pu, qs, rt -> pqrstu", G1, G1, G1))
    return L3

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
        
        if (not mc.converged): raise RuntimeError('MCSCF not converged or not performed.')

        self.nao = mc.mol.nao
        self.ncore = mc.ncore
        self.nact  = mc.ncas
        self.nelecas = mc.nelecas # Tuple of (nalpha, nbeta)
                
        self.nvirt = self.nao - self.nact - self.ncore
        self.flow_param = s
        
        self.nhole = self.ncore + self.nact
        self.npart = self.nact + self.nvirt
        
        self.core = slice(0, self.ncore)
        self.active = slice(self.ncore, self.ncore + self.nact)
        self.virt = slice(self.ncore + self.nact, mc.mol.nao)
        self.hole = slice(0, self.ncore + self.nact)
        self.part = slice(self.ncore, mc.mol.nao)
        
        self.hc = self.core
        self.ha = self.active
        self.pa = slice(0,self.nact)
        self.pv = slice(self.nact, self.nact + self.nvirt)
        
        rhf_eri_ao = self.mc.mol.intor('int2e_sph', aosym='s1') # Chemist's notation (\mu\nu|\lambda\rho)
        self.rhf_eri_mo = ao2mo.incore.full(rhf_eri_ao, self.mc.mo_coeff, False) # (pq|rs)

        self.e_corr = None
        
    def ao2mo(self, mo_coeff): # frozen core should be added later        
        rhf_hcore_ao = self.mol.intor_symmetric('int1e_kin') + self.mol.intor_symmetric('int1e_nuc')
        self.rhf_hcore_mo = np.einsum('pi,pq,qj->ij', mo_coeff, rhf_hcore_ao, mo_coeff)
        
        rhf_eri_ao = self.mol.intor('int2e_sph', aosym='s1') # Chemist's notation (\mu\nu|\lambda\rho)
        self.rhf_eri_mo = ao2mo.incore.full(rhf_eri_ao, mo_coeff, False) # (pq|rs)
    
    def semi_canonicalize(self):
        '''
        Within this function, we generate semicanonicalizer, RDMs, cumulant, F, and V.
        '''
        
        _fock_canon = np.einsum("pi, pq, qj->ij", self.mc.mo_coeff, self.mc.get_fock(), self.mc.mo_coeff, dtype='float64', optimize='optimal')
        
        self.semicanonicalizer = np.zeros((self.nao, self.nao), dtype='float64')
        _, self.semicanonicalizer[self.core,self.core] = np.linalg.eigh(_fock_canon[self.core,self.core])
        _, self.semicanonicalizer[self.active,self.active] = np.linalg.eigh(_fock_canon[self.active,self.active])
        _, self.semicanonicalizer[self.virt,self.virt] = np.linalg.eigh(_fock_canon[self.virt,self.virt])
        
        # RDMs in semi-canonical basis.
        _G1_canon, _G2_canon, _G3_canon = get_SF_RDM(self.mc.ci, self.nact, self.nelecas)
        _G1_semi_canon = np.einsum("pi,pq,qj->ij", self.semicanonicalizer[self.active,self.active], _G1_canon, self.semicanonicalizer[self.active,self.active], dtype='float64', optimize='optimal')
        _G2_semi_canon = np.einsum("pi,qj,rk,sl,pqrs->ijkl", self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], _G2_canon, dtype='float64', optimize='optimal')
        _G3_semi_canon = np.einsum("pi,qj,rk,sl,tm,un,pqrstu->ijklmn", self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], _G3_canon, dtype='float64', optimize='optimal')
        self.Eta = 2. * np.identity(self.nact) - _G1_semi_canon
        self.L1 = _G1_semi_canon.copy()
        self.L2 = get_SF_cu2(_G1_semi_canon, _G2_semi_canon)
        self.L3 = get_SF_cu3(_G1_semi_canon, _G2_semi_canon, _G3_semi_canon)
        del _G1_canon, _G2_canon, _G3_canon, _G1_semi_canon, _G2_semi_canon, _G3_semi_canon
        
        self.fock = np.einsum("pi,pq,qj->ij", self.semicanonicalizer, _fock_canon, self.semicanonicalizer, dtype='float64', optimize='optimal')
        
        _tmp = self.rhf_eri_mo[self.part, self.hole, self.part, self.hole].copy()
        _tmp = np.einsum("aibj->abij", _tmp, dtype='float64')
        
        self.V = np.einsum("pi,qj,pqrs,rk,sl->ijkl", self.semicanonicalizer[self.part, self.part], self.semicanonicalizer[self.part, self.part], _tmp, self.semicanonicalizer[self.hole, self.hole], self.semicanonicalizer[self.hole, self.hole], dtype='float64', optimize='optimal') 
        self.e_orb = np.diagonal(self.fock)
        del _tmp

    def compute_T2(self):
        self.T2 = np.einsum("abij->ijab", self.V.copy())
        for i in range(self.nhole):
            for j in range(self.nhole):
                for k in range(self.npart):
                    for l in range(self.npart):
                        denom = np.float64(self.e_orb[i] + self.e_orb[j] - self.e_orb[self.ncore+k] - self.e_orb[self.ncore+l])
                        self.T2[i, j, k, l] *= np.float64(regularized_denominator(denom, self.flow_param))
                        
        self.T2[self.ha,self.ha,self.pa,self.pa] = 0
        
        self.S = 2.0 * self.T2.copy() - np.einsum("ijab->ijba", self.T2.copy(), dtype='float64', optimize='optimal')   # 2 * J - K             
    
    def compute_T1(self):
        self.T1 = np.zeros((self.nhole, self.npart), dtype='float64')
        self.T1 = self.fock[self.hole,self.part].copy()
        self.T1  += 0.5 * np.einsum('ivaw,wu,uv->ia', self.S[:, self.ha, :, self.pa], self.fock[self.active,self.active], self.L1, dtype='float64', optimize='optimal')
        self.T1 -= 0.5 * np.einsum('iwau,vw,uv->ia', self.S[:, self.ha, :, self.pa], self.fock[self.active,self.active], self.L1, dtype='float64', optimize='optimal')
        
        for i in range(self.nhole):
            for k in range(self.npart):
                denom_t1 = np.float64(self.e_orb[i] - self.e_orb[self.ncore+k])
                self.T1[i, k] *= np.float64(regularized_denominator(denom_t1, self.flow_param))
        
        self.T1[self.ha,self.pa] = 0
        
    def renormalize_V(self):
        self.V_tilde = np.zeros((self.npart, self.npart, self.nhole, self.nhole), dtype='float64')
        self.V_tilde = self.V.copy()
        for k in range(self.npart):
            for l in range(self.npart):
                for i in range(self.nhole):
                    for j in range(self.nhole):
                        denom = np.float64(self.e_orb[i] + self.e_orb[j] - self.e_orb[self.ncore+k] - self.e_orb[self.ncore+l])
                        self.V_tilde[k, l, i, j] *= np.float64(1. + np.exp(-self.flow_param * (denom)**2))
    
    def renormalize_F(self):
        _tmp = np.zeros((self.npart, self.nhole), dtype='float64')
        _tmp = self.fock[self.part,self.hole].copy()
        _tmp += 0.5 * np.einsum("ivaw, wu, uv->ai", self.S[:, self.ha, :, self.pa], self.fock[self.active,self.active], self.L1, dtype='float64', optimize='optimal')
        _tmp -= 0.5 * np.einsum("iwau, vw, uv->ai", self.S[:, self.ha, :, self.pa], self.fock[self.active,self.active], self.L1, dtype='float64', optimize='optimal')
        
        for i in range(self.nhole):
            for k in range(self.npart):
                denom_f = np.float64(self.e_orb[i] - self.e_orb[self.ncore+k])
                _tmp[k, i] *= np.float64(np.exp(-self.flow_param*(denom_f)**2))

        self.F_tilde = np.zeros((self.npart, self.nhole), dtype='float64')
        self.F_tilde = self.fock[self.part,self.hole].copy()
        self.F_tilde += _tmp
        del _tmp
        
    def H1_T1_C0(self):
        E = 0.0
        E = 2. * np.einsum("am,ma->", self.F_tilde[:, self.hc], self.T1[self.hc, :], dtype='float64', optimize='optimal')
        temp = np.einsum("ev,ue->uv", self.F_tilde[self.pv, self.ha], self.T1[self.ha, self.pv], dtype='float64', optimize='optimal') 
        temp -= np.einsum("um,mv->uv", self.F_tilde[self.pa, self.hc], self.T1[self.hc, self.pa], dtype='float64', optimize='optimal')
        E += np.einsum("vu,uv->", self.L1, temp, dtype='float64', optimize='optimal')
        return E
    
    def H1_T2_C0(self):
        E = 0.0
        temp = np.einsum("ex,uvey->uvxy", self.F_tilde[self.pv, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], dtype='float64', optimize='optimal')
        temp -= np.einsum("vm,umxy->uvxy", self.F_tilde[self.pa, self.hc], self.T2[self.ha, self. hc, self.pa, self.pa], dtype='float64', optimize='optimal')
        E = np.einsum("xyuv,uvxy->", self.L2, temp, dtype='float64', optimize='optimal')
        return E
    
    def H2_T1_C0(self):
        E = 0.0
        temp = np.einsum("evxy,ue->uvxy", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.T1[self.ha, self.pv], dtype='float64', optimize='optimal')
        temp -= np.einsum("uvmy,mx->uvxy", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.T1[self.hc, self.pa], dtype='float64', optimize='optimal')
        E = np.einsum("xyuv,uvxy->", self.L2, temp, dtype='float64')
        return E
    
    def H2_T2_C0(self):
        E = np.einsum("efmn,mnef->", self.V_tilde[self.pv, self.pv, self.hc, self.hc], self.S[self.hc, self.hc, self.pv, self.pv], dtype='float64', optimize='optimal')
        E += np.einsum("efmu,mvef,uv->", self.V_tilde[self.pv, self.pv, self.hc, self.ha], self.S[self.hc, self.ha, self.pv, self.pv], self.L1, dtype='float64', optimize='optimal')
        E += np.einsum("vemn,mnue,uv->", self.V_tilde[self.pa, self.pv, self.hc, self.hc], self.S[self.hc, self.hc, self.pa, self.pv], self.Eta, dtype='float64', optimize='optimal')
        E += self.H2_T2_C0_T2small()
        return E
    
    def H2_T2_C0_T2small(self):
    #  Note the following blocks should be available in memory.
    #  H2: vvaa, aacc, avca, avac, vaaa, aaca
    #  T2: aavv, ccaa, caav, acav, aava, caaa
    #  S2: aavv, ccaa, caav, acav, aava, caaa
        E = 0.0
        # [H2, T2] L1 from aavv
        E += 0.25 * np.einsum("efxu,yvef,uv,xy -> ", self.V_tilde[self.pv, self.pv, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pv], self.L1, self.L1, dtype='float64', optimize='optimal')
        # [H2, T2] L1 from ccaa
        E += 0.25 * np.einsum("vymn,mnux,uv,xy -> ", self.V_tilde[self.pa, self.pa, self.hc, self.hc], self.S[self.hc, self.hc, self.pa, self.pa], self.Eta, self.Eta, dtype='float64', optimize='optimal')
        # [H2, T2] L1 from caav
        temp = 0.5 * np.einsum("vemx,myue-> uxyv", self.V_tilde[self.pa, self.pv, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pv], dtype='float64', optimize='optimal')
        temp += 0.5 * np.einsum("vexm,ymue-> uxyv", self.V_tilde[self.pa, self.pv, self.ha, self.hc], self.S[self.ha, self.hc, self.pa, self.pv], dtype='float64', optimize='optimal')
        E += np.einsum("uxyv, uv, xy -> ", temp, self.Eta, self.L1, dtype='float64', optimize='optimal')
        # [H2, T2] L1 from caaa and aaav
        temp = 0.25 * np.einsum("evwx, zyeu, wz -> uxyv", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pa], self.L1, dtype='float64', optimize='optimal')
        temp += 0.25 * np.einsum("vzmx, myuw, wz -> uxyv", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pa], self.Eta, dtype='float64', optimize='optimal')
        E += np.einsum("uxyv, uv, xy ->", temp, self.Eta, self.L1, dtype='float64', optimize='optimal')
        
        # <[Hbar2, T2]> C_4 (C_2)^2
        # HH
        temp = 0.5 * np.einsum("uvmn, mnxy -> uvxy", self.V_tilde[self.pa, self.pa, self.hc, self.hc], self.T2[self.hc, self.hc, self. pa, self.pa], dtype='float64', optimize='optimal')
        temp += 0.5 * np.einsum("uvmw, mzxy, wz -> uvxy", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.L1, dtype='float64', optimize='optimal')
        
        # PP
        temp += 0.5 * np.einsum("efxy, uvef -> uvxy", self.V_tilde[self.pv, self.pv, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pv], dtype='float64', optimize='optimal')
        temp += 0.5 * np.einsum("ezxy, uvew, wz -> uvxy", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.Eta, dtype='float64', optimize='optimal')
        
        # HP
        temp += np.einsum("uexm, vmye -> uvxy", self.V_tilde[self.pa, self.pv, self.ha, self.hc], self.S[self.ha, self.hc, self.pa, self.pv], dtype='float64', optimize='optimal')
        temp -= np.einsum("uemx, vmye -> uvxy", self.V_tilde[self.pa, self.pv, self.hc, self.ha], self.T2[self.ha, self.hc, self.pa, self.pv], dtype='float64', optimize='optimal')
        temp -= np.einsum("vemx, muye -> uvxy", self.V_tilde[self.pa, self.pv, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pv], dtype='float64', optimize='optimal')
        
        # HP with Gamma1
        temp += 0.5 * np.einsum("euwx, zvey, wz -> uvxy", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pa], self.L1, dtype='float64', optimize='optimal')
        temp -= 0.5 * np.einsum("euxw, zvey, wz -> uvxy", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.L1, dtype='float64', optimize='optimal')
        temp -= 0.5 * np.einsum("evxw, uzey, wz -> uvxy", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.L1, dtype='float64', optimize='optimal')
        
        # HP with Eta1
        temp += 0.5 * np.einsum("wumx, mvzy, wz -> uvxy", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pa], self.Eta, dtype='float64', optimize='optimal')
        temp -= 0.5 * np.einsum("uwmx, mvzy, wz -> uvxy", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.Eta, dtype='float64', optimize='optimal')
        temp -= 0.5 * np.einsum("vwmx, muyz, wz -> uvxy", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.Eta, dtype='float64', optimize='optimal')
        
        E += np.einsum("uvxy, uvxy ->", temp, self.L2)
        
        #
        E += np.einsum("ewxy, uvez, xyzuwv ->", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.L3, dtype='float64', optimize='optimal')
        E -= np.einsum("uvmz, mwxy, xyzuwv ->", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.L3, dtype='float64', optimize='optimal')
        
        return E
    
    def kernel(self):
        self.semi_canonicalize()
        
        self.compute_T2()
        self.compute_T1()
        self.renormalize_V()
        self.renormalize_F()

        self.e_h1_t1 = self.H1_T1_C0()
        self.e_h1_t2 = self.H1_T2_C0()
        self.e_h2_t1 = self.H2_T1_C0()
        self.e_h2_t2 = self.H2_T2_C0()
        self.e_corr = self.e_h1_t1 + self.e_h1_t2 + self.e_h2_t1 + self.e_h2_t2
        self.e_tot = self.mc.e_tot + self.e_corr

        return self.e_corr

# register DSRG_MRPT2 in MCSCF
# [todo]: is this so that we can access fcisolver options?
from pyscf.mcscf import casci
casci.CASCI.DSRG_MRPT2 = DSRG_MRPT2


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    if (False):
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
        assert np.isclose(e_dsrg_mrpt2, -0.127274453305632)

        mol = gto.M(
            verbose = 2,
            atom = '''
        H 0 0 0
        F 0 0 1.5
        ''',
            basis = 'sto-3g', spin=0, charge=0
        )
        rhf = scf.RHF(mol)
        rhf.kernel()
        casci = mcscf.CASCI(rhf, 4, 6)
        casci.kernel()
        dsrg = DSRG_MRPT2(casci)
        e_dsrg_mrpt2 = dsrg.kernel()
        print(e_dsrg_mrpt2)
        print(dsrg.e_tot)

    mol = gto.M(
        verbose = 2,
        atom = '''
    O 0 0 0
    O 0 0 1.251
    ''',
        basis = 'cc-pvdz', spin=0, charge=0
    )
    mf = scf.RHF(mol)
    mf.kernel()
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fix_spin_(ss=0) # we want the singlet state, not the Ms=0 triplet state
    mc.kernel() 
    dsrg = DSRG_MRPT2(mc)
    e_dsrg_mrpt2 = dsrg.kernel()
    print(e_dsrg_mrpt2)
    print(dsrg.e_tot)