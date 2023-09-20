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
Taylor_threshold = 1e-3

def Taylor_Exp(Z, Taylor_threshold):
    n = int(0.5 * (15.0 / Taylor_threshold + 1)) + 1
    if (n > 0):
        value = Z
        tmp = Z
        for x in range(n-1):
            tmp *= -1.0 * Z * Z / (x + 2)
            value += tmp

        return value
    else:
        return 0.0
    
def regularized_denominator(x, s):
    z = np.sqrt(s) * x
    if abs(z) <= MACHEPS:
        return Taylor_Exp(z, Taylor_threshold) * np.sqrt(s)
    return (1. - np.exp(-s * x**2)) / x

def regularized_denominator(x, s):
    z = np.sqrt(s) * x
    if abs(z) <= MACHEPS:
        return Taylor_Exp(z, Taylor_threshold) * np.sqrt(s)
    return (1. - np.exp(-s * x**2)) / x
    
def get_SF_RDM(ci_vec, nelec, norb):
# Reordered 2-rdm <p\dagger r\dagger s q> in Pyscf is stored as: dm2[pqrs]
# Forte store it as rdm[prqs]
    dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', ci_vec, ci_vec, norb, nelec)
    dm1, dm2, dm3 = fci.rdm.reorder_dm123(dm1, dm2, dm3)
    G1 = np.einsum("pq -> qp", dm1)
    G2 = np.einsum("pqrs -> prqs", dm2)
    G3 = np.einsum("pqrstu -> prtqsu", dm3)
    return G1, G2, G3

def get_SF_cu2(G1, G2):
    L2 = G2.copy() 
    L2 -= np.einsum("pr, qs->pqrs", G1, G1)
    L2 += 0.5 * np.einsum("ps, qr->pqrs", G1, G1)
    return L2
    
def get_SF_cu3(G1, G2, G3): 
    L3 = G3.copy() # PQRSTU
    L3 -= (np.einsum("ps,qrtu -> pqrstu", G1, G2) + np.einsum("qt,prsu -> pqrstu", G1, G2) + np.einsum("ru,pqst -> pqrstu", G1, G2))
    L3 += 0.5 * (np.einsum("pt,qrsu -> pqrstu", G1, G2) + np.einsum("pu,qrts -> pqrstu", G1, G2) + np.einsum("qs,prtu -> pqrstu", G1, G2) + np.einsum("qu,prst -> pqrstu", G1, G2) + np.einsum("rs,pqut -> pqrstu", G1, G2) + np.einsum("rt,pqsu -> pqrstu", G1, G2))
    L3 += 2 * np.einsum("ps, qt, ru -> pqrstu", G1, G1, G1)
    L3 -= (np.einsum("ps, qu, rt -> pqrstu", G1, G1, G1) + np.einsum("pu, qt, rs -> pqrstu", G1, G1, G1) + np.einsum("pt, qs, ru -> pqrstu", G1, G1, G1))
    L3 += 0.5 * (np.einsum("pt, qu, rs -> pqrstu", G1, G1, G1) + np.einsum("pu, qs, rt -> pqrstu", G1, G1, G1))
    return L3

# def F_T1(mc, dms, eris):
#     pass

# def F_T2(mc, dms, eris):
#     pass

# def V_T1(mc, dms, eris):
#     pass

# def V_T2(mc, dms, eris):
#     pass

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
        
        self.res = mc.kernel()

        self.ncore = mc.ncore
        self.nact  = mc.ncas
        self.nelecas = mc.nelecas # Tuple of (nalpha, nbeta)
        
        if (type(self.nelecas) is tuple):
            self.nelec = 0
            for i in self.nelecas:
                self.nelec += i
        else:
            self.nelec = self.nelecas
        
        self.nvirt = mc.mol.nao - self.nact - self.ncore
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
        
        rhf_eri_ao = self.mol.intor('int2e_sph', aosym='s1') # Chemist's notation (\mu\nu|\lambda\rho)
        self.rhf_eri_mo = ao2mo.incore.full(rhf_eri_ao, self.res[3], False) # (pq|rs)

        self.e_corr = None

    def load_ci(self, root=None):
        if root is None:
            root = self.root
        
        if self.fcisolver.nroots == 1:
            return self.ci
        else:
            return self.ci[root]
        
    def semi_canonicalize(self):
        # Within this function, we generate semicanonicalizer, RDMs, cumulant, F, and V.
        
        F = np.einsum("pi, pq, qj->ij", self.res[3], self.mc.get_fock(), self.res[3], dtype='float64', optimize='optimal')
        
        self.semicanonicalizer = np.zeros((self.nao, self.nao), dtype='float64')
        _, self.semicanonicalizer[self.core,self.core] = np.linalg.eigh(F[self.core,self.core])
        _, self.semicanonicalizer[self.active,self.active] = np.linalg.eigh(F[self.active,self.active])
        _, self.semicanonicalizer[self.virt,self.virt] = np.linalg.eigh(F[self.virt,self.virt])
        
        # RDMs in semi-canonical basis.
        G1, G2, G3 = get_SF_RDM(self.res[2], self.nelec, self.nact)
        self.G1 = np.einsum("pi, pq, qj->ij", self.semicanonicalizer[self.active,self.active], G1, self.semicanonicalizer[self.active,self.active], dtype='float64', optimize='optimal')
        self.G2 = np.einsum("pi, qj, rk, sl, pqrs->ijkl", self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], G2, dtype='float64', optimize='optimal')
        self.G3 = np.einsum("pi, qj, rk, sl, tm, un, pqrstu->ijklmn", self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], G3, dtype='float64', optimize='optimal')
        self.Eta = 2. * np.identity(self.nact) - self.G1
        self.L1 = self.G1.copy() # Remove this.
        self.L2 = get_SF_cu2(self.G1, self.G2)
        self.L3 = get_SF_cu3(self.G1, self.G2, self.G3)
        
        self.F = np.einsum("pi, pq, qj->ij", self.semicanonicalizer, F, self.semicanonicalizer, dtype='float64', optimize='optimal')
        
        tmp= self.rhf_eri_mo[self.part, self.hole, self.part, self.hole].copy()
        tmp= np.einsum("aibj->abij", tmp, dtype='float64')
        
        self.V = np.einsum("pi, qj, pqrs, rk, sl->ijkl", self.semicanonicalizer[self.part, self.part], self.semicanonicalizer[self.part, self.part], tmp, self.semicanonicalizer[self.hole, self.hole], self.semicanonicalizer[self.hole, self.hole], dtype='float64', optimize='optimal') 
        self.e_orb = np.diagonal(self.F)

    def compute_T2(self):
        self.T2 = np.einsum("abij -> ijab", self.V.copy())
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
        self.T1 = self.F[self.hole,self.part].copy()
        self.T1  += 0.5 * np.einsum('ivaw, wu, uv->ia', self.S[:, self.ha, :, self.pa], self.F[self.active,self.active], self.G1, dtype='float64', optimize='optimal')
        self.T1 -= 0.5 * np.einsum('iwau, vw, uv->ia', self.S[:, self.ha, :, self.pa], self.F[self.active,self.active], self.G1, dtype='float64', optimize='optimal')
        
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
        tmp_f = np.zeros((self.npart, self.nhole), dtype='float64')
        tmp_f = self.F[self.part,self.hole].copy()
        tmp_f += 0.5 * np.einsum("ivaw, wu, uv->ai", self.S[:, self.ha, :, self.pa], self.F[self.active,self.active], self.G1, dtype='float64', optimize='optimal')
        tmp_f -= 0.5 * np.einsum("iwau, vw, uv->ai", self.S[:, self.ha, :, self.pa], self.F[self.active,self.active], self.G1, dtype='float64', optimize='optimal')
        
        for i in range(self.nhole):
            for k in range(self.npart):
                denom_f = np.float64(self.e_orb[i] - self.e_orb[self.ncore+k])
                tmp_f[k, i] *= np.float64(np.exp(-self.flow_param*(denom_f)**2))
                   
        
        self.F_tilde = np.zeros((self.npart, self.nhole), dtype='float64')
        self.F_tilde = self.F[self.part,self.hole].copy()
        self.F_tilde += tmp_f
        
    def H1_T1_C0(self):
        E = 0.0
        E = 2. * np.einsum("am, ma ->", self.F_tilde[:, self.hc], self.T1[self.hc, :], dtype='float64', optimize='optimal')
        temp = np.einsum("ev, ue -> uv", self.F_tilde[self.pv, self.ha], self.T1[self.ha, self.pv], dtype='float64', optimize='optimal') 
        temp -= np.einsum("um, mv -> uv", self.F_tilde[self.pa, self.hc], self.T1[self.hc, self.pa], dtype='float64', optimize='optimal')
        E += np.einsum("vu,uv ->", self.G1, temp, dtype='float64', optimize='optimal')
        print(E)
        return E
    
    def H1_T2_C0(self):
        E = 0.0
        temp = np.einsum("ex, uvey -> uvxy", self.F_tilde[self.pv, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], dtype='float64', optimize='optimal')
        temp -= np.einsum("vm, umxy -> uvxy", self.F_tilde[self.pa, self.hc], self.T2[self.ha, self. hc, self.pa, self.pa], dtype='float64', optimize='optimal')
        E = np.einsum("xyuv, uvxy -> ", self.L2, temp, dtype='float64', optimize='optimal')
        print(E)
        return E
    
    def H2_T1_C0(self):
        E = 0.0
        temp = np.einsum("evxy, ue -> uvxy", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.T1[self.ha, self.pv], dtype='float64', optimize='optimal')
        temp -= np.einsum("uvmy, mx -> uvxy", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.T1[self.hc, self.pa], dtype='float64', optimize='optimal')
        E = np.einsum("xyuv, uvxy -> ", self.L2, temp, dtype='float64')
        print(E)
        return E
    
    def H2_T2_C0(self):
        Eout = np.zeros(3, dtype='float64')
        E = np.einsum("efmn, mnef ->", self.V_tilde[self.pv, self.pv, self.hc, self.hc], self.S[self.hc, self.hc, self.pv, self.pv], dtype='float64', optimize='optimal')
        E += np.einsum("efmu, mvef, uv -> ", self.V_tilde[self.pv, self.pv, self.hc, self.ha], self.S[self.hc, self.ha, self.pv, self.pv], self.L1, dtype='float64', optimize='optimal')
        E += np.einsum("vemn, mnue, uv -> ", self.V_tilde[self.pa, self.pv, self.hc, self.hc], self.S[self.hc, self.hc, self.pa, self.pv], self.Eta, dtype='float64', optimize='optimal')
        Eout[0] += E
        Esmall = self.H2_T2_C0_T2small()
        for i in range(3):
            E += Esmall[i]
            Eout[i] += Esmall[i]
        print(E)
        print(Eout)
        return E, Eout
    
    def H2_T2_C0_T2small(self):
    #  Note the following blocks should be available in memory.
    #  H2: vvaa, aacc, avca, avac, vaaa, aaca
    #  T2: aavv, ccaa, caav, acav, aava, caaa
    #  S2: aavv, ccaa, caav, acav, aava, caaa
        Esmall = np.zeros(3, dtype='float64')
        # [H2, T2] L1 from aavv
        Esmall[0] += 0.25 * np.einsum("efxu, yvef, uv, xy -> ", self.V_tilde[self.pv, self.pv, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pv], self.L1, self.L1, dtype='float64', optimize='optimal')
        # [H2, T2] L1 from ccaa
        Esmall[0] += 0.25 * np.einsum("vymn, mnux, uv, xy -> ", self.V_tilde[self.pa, self.pa, self.hc, self.hc], self.S[self.hc, self.hc, self.pa, self.pa], self.Eta, self.Eta, dtype='float64', optimize='optimal')
        # [H2, T2] L1 from caav
        temp = 0.5 * np.einsum("vemx, myue -> uxyv", self.V_tilde[self.pa, self.pv, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pv], dtype='float64', optimize='optimal')
        temp += 0.5 * np.einsum("vexm, ymue -> uxyv", self.V_tilde[self.pa, self.pv, self.ha, self.hc], self.S[self.ha, self.hc, self.pa, self.pv], dtype='float64', optimize='optimal')
        Esmall[0] += np.einsum("uxyv, uv, xy -> ", temp, self.Eta, self.L1, dtype='float64', optimize='optimal')
        # [H2, T2] L1 from caaa and aaav
        temp = 0.25 * np.einsum("evwx, zyeu, wz -> uxyv", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pa], self.L1, dtype='float64', optimize='optimal')
        temp += 0.25 * np.einsum("vzmx, myuw, wz -> uxyv", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pa], self.Eta, dtype='float64', optimize='optimal')
        Esmall[0] += np.einsum("uxyv, uv, xy ->", temp, self.Eta, self.L1, dtype='float64', optimize='optimal')
        
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
        
        Esmall[1] += np.einsum("uvxy, uvxy ->", temp, self.L2)
        
        #
        Esmall[2] += np.einsum("ewxy, uvez, xyzuwv ->", self.V_tilde[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.L3, dtype='float64', optimize='optimal')
        Esmall[2] -= np.einsum("uvmz, mwxy, xyzuwv ->", self.V_tilde[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.L3, dtype='float64', optimize='optimal')
        
        return Esmall
    
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

        self.compute_T2()
        self.compute_T1()
        self.renormalize_V()
        self.renormalize_F()

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
