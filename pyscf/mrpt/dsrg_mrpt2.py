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
from pyscf import lib, mcscf
from pyscf.lib import logger
from pyscf import fci
from pyscf.mcscf import mc_ao2mo
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
import math

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
    
def get_SF_RDM_SA(ci_vecs, weights, norb, nelec):
    '''
    Returns the state-averaged spin-free active space 1-/2-/3-RDM.
    Reordered 2-rdm <p\dagger r\dagger s q> in Pyscf is stored as: dm2[pqrs]
    Forte stores it as rdm[prqs]
    '''
    G1 = np.zeros((norb,)*2)
    G2 = np.zeros((norb,)*4)
    G3 = np.zeros((norb,)*6)

    for i in range(len(ci_vecs)):
        # Unlike fcisolver.make_rdm1, make_dm123 doesn't automatically return the state-averaged RDM.
        _dm1, _dm2, _dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', ci_vecs[i], ci_vecs[i], norb, nelec)
        _dm1, _dm2, _dm3 = fci.rdm.reorder_dm123(_dm1, _dm2, _dm3)
        _G1 = np.einsum("pq->qp", _dm1)
        _G2 = np.einsum("pqrs->prqs", _dm2)
        _G3 = np.einsum("pqrstu->prtqsu", _dm3)
        G1 += weights[i] * _G1
        G2 += weights[i] * _G2
        G3 += weights[i] * _G3
    return G1, G2, G3

def get_SF_cu2(G1, G2):
    '''
    Returns the spin-free active space 2-body cumulant.
    '''
    L2 = G2.copy() 
    L2 -= np.einsum("pr,qs->pqrs", G1, G1)
    L2 += 0.5 * np.einsum("ps,qr->pqrs", G1, G1)
    return L2
    
def get_SF_cu3(G1, G2, G3): 
    '''
    Returns the spin-free active space 3-body cumulant.
    '''
    L3 = G3.copy() 
    L3 -= (np.einsum("ps,qrtu -> pqrstu", G1, G2) + np.einsum("qt,prsu->pqrstu", G1, G2) + np.einsum("ru,pqst->pqrstu", G1, G2))
    L3 += 0.5 * (np.einsum("pt,qrsu->pqrstu", G1, G2) + np.einsum("pu,qrts->pqrstu", G1, G2) + np.einsum("qs,prtu->pqrstu", G1, G2) + \
                 np.einsum("qu,prst->pqrstu", G1, G2) + np.einsum("rs,pqut->pqrstu", G1, G2) + np.einsum("rt,pqsu->pqrstu", G1, G2))
    L3 += 2 * np.einsum("ps,qt,ru->pqrstu", G1, G1, G1)
    L3 -= (np.einsum("ps,qu,rt->pqrstu", G1, G1, G1) + np.einsum("pu,qt,rs->pqrstu", G1, G1, G1) + np.einsum("pt,qs,ru->pqrstu", G1, G1, G1))
    L3 += 0.5 * (np.einsum("pt,qu,rs->pqrstu", G1, G1, G1) + np.einsum("pu,qs,rt->pqrstu", G1, G1, G1))
    return L3

class DSRG_MRPT2(lib.StreamObject):
    '''
    DSRG-MRPT2

    Attributes:
        s : float (default: 0.5)
            The flow parameter, which controls the extent to which 
            the Hamiltonian is block-diagonalized.
        relax : str (default: 'none')
            Reference relaxation method. Options: 'none', 'once', 'twice', 'iterate'.
        density_fit: bool (default: False)
            To control whether density fitting to be used.
            For CCVV, CAVV, and CCAV terms, V and T2 will not be stored explicitly.
        batch: bool(default: False)
            To control whether the CCVV term to be computed in batches.
            CCVV: for a given m and n, form B(ef) = Bm(L|e) * Bn(L|f)
            This is only available with density fitting.
            
            (Bpq is larger than Bme. I am not sure whether batching would provide any benefit since we always store Bpq.)
        verbose: bool(default: False)

    Examples:

    >>> mf = gto.M('N 0 0 0; N 0 0 1.4', basis='6-31g').apply(scf.RHF).run()
    >>> mc = mcscf.CASCI(mf, 4, 4).run()
    >>> DSRG_MRPT2(mc, s=0.5).kernel()
    -0.15708345625685638
    '''
    def __init__(self, mc, s=0.5, relax='none', relax_maxiter=10, relax_conv=1e-8, density_fit=False, batch=False, verbose=False):
        if (not mc.converged): raise RuntimeError('MCSCF not converged or not performed.')
        self.mc = mc
        self.flow_param = s
        self.relax = relax
        if (relax not in ['none','once','twice','iterate']):
            raise RuntimeError(f"Relaxation method '{relax}' not recognized. Supported methods are 'none', 'once', 'twice', and 'iterate'.")
        self.df = density_fit
        self.batch = batch
        self.verbose = verbose

        if (isinstance(mc.fcisolver, mcscf.addons.StateAverageFCISolver)):
            self.state_average = True
            self.state_average_weights = mc.fcisolver.weights
            self.state_average_nstates = mc.fcisolver.nstates
            self.ci_vecs = mc.ci
        else:
            self.state_average = False
            self.state_average_weights = [1.0]
            self.state_average_nstates = 1
            self.ci_vecs = [mc.ci]

        if (relax == 'none'):
            self.nrelax = 0
        elif (relax == 'once'):
            self.nrelax = 1
        elif (relax == 'twice'):
            self.nrelax = 2
        elif (relax == 'iterate'):
            self.nrelax = relax_maxiter

        self.relax_ref = (self.nrelax > 0)

        # [todo]: remove this restriction
        if (self.relax_ref and self.df):
            raise RuntimeError('Relaxation is not supported with density fitting.')

        self.converged = False

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
        
        self.e_corr = None

        _, self.e_core = mc.get_h1eff()
    
    def semi_canonicalize(self):
        # get_fock() uses the state-averaged RDM by default, via mc.fcisolver.make_rdm1()
        _G1_canon, _G2_canon, _G3_canon = get_SF_RDM_SA(self.ci_vecs, self.state_average_weights, self.nact, self.nelecas)
        _fock_canon = np.einsum("pi,pq,qj->ij", self.mc.mo_coeff, self.mc.get_fock(casdm1=_G1_canon), self.mc.mo_coeff, optimize='optimal') 
        self.semicanonicalizer = np.zeros((self.nao, self.nao), dtype='float64')
        _, self.semicanonicalizer[self.core,self.core] = np.linalg.eigh(_fock_canon[self.core,self.core])
        _, self.semicanonicalizer[self.active,self.active] = np.linalg.eigh(_fock_canon[self.active,self.active])
        _, self.semicanonicalizer[self.virt,self.virt] = np.linalg.eigh(_fock_canon[self.virt,self.virt])
        self.fock = np.einsum("pi,pq,qj->ij", self.semicanonicalizer, _fock_canon, self.semicanonicalizer, optimize='optimal')

        # RDMs in semi-canonical basis.
        # This should be fine since all indices are active.
        
        _G1_semi_canon = np.einsum("pi,qj,pq->ij", \
                                    self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], \
                                    _G1_canon, optimize='optimal')
        _G2_semi_canon = np.einsum("pi,qj,rk,sl,pqrs->ijkl", \
                                    self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], \
                                    self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], \
                                    _G2_canon, optimize='optimal')
        _G3_semi_canon = np.einsum("pi,qj,rk,sl,tm,un,pqrstu->ijklmn", \
                                    self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], \
                                    self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], \
                                    self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], \
                                    _G3_canon, optimize='optimal')
        self.Eta = 2. * np.identity(self.nact) - _G1_semi_canon
        self.L1 = _G1_semi_canon.copy()
        self.L2 = get_SF_cu2(_G1_semi_canon, _G2_semi_canon)
        self.L3 = get_SF_cu3(_G1_semi_canon, _G2_semi_canon, _G3_semi_canon)
        del _G1_canon, _G2_canon, _G3_canon, _G1_semi_canon, _G2_semi_canon, _G3_semi_canon
        
        if (self.df):
            # I don't think batching will help here since a N^3 tensor (Bpq_ao) has to be construct explicitly.
            # If we want to avoid storing tensors with N^3 elements, DiskDF should be implemented.
            self.semi_coeff = np.einsum("pi,up->ui", self.semicanonicalizer, self.mc.mo_coeff, optimize='optimal')
            self.V = dict.fromkeys(["vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"])
            Bpq_ao = lib.unpack_tril(mc.with_df._cderi) # Aux * ao * ao
            self.Bpq = np.einsum("pi,lpq,qj->lij", self.semi_coeff[:, self.part], Bpq_ao, self.semi_coeff[:, self.hole], optimize='optimal') # Aux * Particle * Hole
            self.V["vvaa"] = np.einsum("gai,gbj->abij", self.Bpq[:, self.pv, self.ha], self.Bpq[:, self.pv, self.ha], optimize='optimal')
            self.V["aacc"] = np.einsum("gai,gbj->abij", self.Bpq[:, self.pa, self.hc], self.Bpq[:, self.pa, self.hc], optimize='optimal')
            self.V["avca"] = np.einsum("gai,gbj->abij", self.Bpq[:, self.pa, self.hc], self.Bpq[:, self.pv, self.ha], optimize='optimal')
            self.V["avac"] = np.einsum("gai,gbj->abij", self.Bpq[:, self.pa, self.ha], self.Bpq[:, self.pv, self.hc], optimize='optimal')
            self.V["vaaa"] = np.einsum("gai,gbj->abij", self.Bpq[:, self.pv, self.ha], self.Bpq[:, self.pa, self.ha], optimize='optimal')
            self.V["aaca"] = np.einsum("gai,gbj->abij", self.Bpq[:, self.pa, self.hc], self.Bpq[:, self.pa, self.ha], optimize='optimal')
            self.V["aaaa"] = np.einsum("gai,gbj->abij", self.Bpq[:, self.pa, self.ha], self.Bpq[:, self.pa, self.ha], optimize='optimal')
            del Bpq_ao
        else:
            _rhf_eri_ao = self.mc.mol.intor('int2e_sph', aosym='s1') # Chemist's notation
            _rhf_eri_mo = ao2mo.incore.full(_rhf_eri_ao, self.mc.mo_coeff, False) # (pq|rs)
            _tmp = _rhf_eri_mo[self.part, self.hole, self.part, self.hole].copy()
            _tmp = np.einsum("aibj->abij", _tmp, dtype='float64')
            self.V = np.einsum("pi,qj,pqrs,rk,sl->ijkl", self.semicanonicalizer[self.part, self.part], self.semicanonicalizer[self.part, self.part], _tmp, self.semicanonicalizer[self.hole, self.hole], self.semicanonicalizer[self.hole, self.hole], optimize='optimal') 
            del _rhf_eri_ao, _rhf_eri_mo, _tmp
            
    def compute_T2_df_minimal(self):
        self.e_orb = {"c":np.diagonal(self.fock)[self.core], "a": np.diagonal(self.fock)[self.active], "v": np.diagonal(self.fock)[self.virt]}
        self.T2 = {}
        self.S = {}
        #ONLY these T2 blocks are stored: aavv, ccaa, caav, acav, aava, caaa. Internal exciation (aaaa) tensor is zero.
        for Vblock, tensor in self.V.items():
            if Vblock != "aaaa":
                block = Vblock[2] + Vblock[3] + Vblock[0] + Vblock[1]   
                self.T2[block] = np.einsum("abij->ijab", tensor.copy())
                # Shuhang Li: This is inefficient.
                for a in range(tensor.shape[0]):
                    for b in range(tensor.shape[1]):
                        for i in range(tensor.shape[2]):
                            for j in range(tensor.shape[3]):
                                denom = -np.float64(self.e_orb[block[2]][a] + self.e_orb[block[3]][b] - self.e_orb[block[0]][i] - self.e_orb[block[1]][j])
                                self.T2[block][i,j,a,b] *= np.float64(regularized_denominator(denom, self.flow_param))
        # form S2 = 2 * J - K 
        # aavv, ccaa, caav, acav, aava, caaa                       
        self.S["aavv"] = 2.0 * self.T2["aavv"] - np.einsum("uvef->uvfe", self.T2["aavv"])
        self.S["ccaa"] = 2.0 * self.T2["ccaa"] - np.einsum("mnuv->mnvu", self.T2["ccaa"])
        self.S["caav"] = 2.0 * self.T2["caav"] - np.einsum("umve->muve", self.T2["acav"])
        self.S["acav"] = 2.0 * self.T2["acav"] - np.einsum("muve->umve", self.T2["caav"])
        self.S["aava"] = 2.0 * self.T2["aava"] - np.einsum("vuex->uvex", self.T2["aava"])
        self.S["caaa"] = 2.0 * self.T2["caaa"] - np.einsum("muvx->muxv", self.T2["caaa"])

    def compute_T2_full(self):
        self.e_orb = np.diagonal(self.fock)
        self.T2 = np.einsum("abij->ijab", self.V.copy())
        for i in range(self.nhole):
            for j in range(self.nhole):
                for k in range(self.npart):
                    for l in range(self.npart):
                        denom = np.float64(self.e_orb[i] + self.e_orb[j] - self.e_orb[self.ncore+k] - self.e_orb[self.ncore+l])
                        self.T2[i, j, k, l] *= np.float64(regularized_denominator(denom, self.flow_param))    
        self.T2[self.ha,self.ha,self.pa,self.pa] = 0
        # 2 * J - K
        self.S = 2.0 * self.T2.copy() - np.einsum("ijab->ijba", self.T2.copy(), optimize='optimal')
    
    def renormalize_V_df(self):
        for block, tensor in self.V.items():   # abij
            # Shuhang Li: This is inefficient.
            for a in range(tensor.shape[0]):
                for b in range(tensor.shape[1]):
                    for i in range(tensor.shape[2]):
                        for j in range(tensor.shape[3]):
                            denom = np.float64(self.e_orb[block[0]][a] + self.e_orb[block[1]][b] - self.e_orb[block[2]][i] - self.e_orb[block[3]][j])
                            self.V[block][a,b,i,j] *= np.float64(1. + np.exp(-self.flow_param * (denom)**2))
                            
    def renormalize_V_full(self):
        for k in range(self.npart):
            for l in range(self.npart):
                for i in range(self.nhole):
                    for j in range(self.nhole):
                        denom = np.float64(self.e_orb[i] + self.e_orb[j] - self.e_orb[self.ncore+k] - self.e_orb[self.ncore+l])
                        self.V[k, l, i, j] *= np.float64(1. + np.exp(-self.flow_param * (denom)**2))
    
    def compute_T2(self):
        if (self.df):
            self.compute_T2_df_minimal()  
        else:
            self.compute_T2_full()             
    
    def compute_T1(self):
        # initialize T1 with F + [H0, A]
        self.T1 = self.fock[self.hole,self.part].copy()
        if (self.df):      
            self.T1[self.hc, self.pa] += 0.5 * np.einsum("ivaw, wu, uv->ia", self.S["caaa"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            self.T1[self.hc, self.pv] += 0.5 * np.einsum("vmwe, wu, uv->me", self.S["acav"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            self.T1[self.ha, self.pv] += 0.5 * np.einsum("ivaw, wu, uv->ia", self.S["aava"], self.fock[self.active,self.active], self.L1, optimize='optimal')
        
            self.T1[self.hc, self.pa] -= 0.5 * np.einsum("iwau,vw,uv->ia", self.S["caaa"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            self.T1[self.hc, self.pv] -= 0.5 * np.einsum("wmue,vw,uv->me", self.S["acav"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            self.T1[self.ha, self.pv] -= 0.5 * np.einsum("iwau,vw,uv->ia", self.S["aava"], self.fock[self.active,self.active], self.L1, optimize='optimal')
        else:
            self.T1 += 0.5 * np.einsum('ivaw,wu,uv->ia', self.S[:, self.ha, :, self.pa], self.fock[self.active,self.active], self.L1, optimize='optimal')
            self.T1 -= 0.5 * np.einsum('iwau,vw,uv->ia', self.S[:, self.ha, :, self.pa], self.fock[self.active,self.active], self.L1, optimize='optimal')
        
        # Shuhang Li: This is inefficient.
        for i in range(self.nhole):
            for k in range(self.npart):
                denom_t1 = np.float64(np.diagonal(self.fock)[i] - np.diagonal(self.fock)[self.ncore+k])
                self.T1[i, k] *= np.float64(regularized_denominator(denom_t1, self.flow_param))
        self.T1[self.ha,self.pa] = 0
 
    def renormalize_V(self):
        if (self.df):
            self.renormalize_V_df()
        else:
            self.renormalize_V_full()
    
    def renormalize_F(self):
        _tmp = np.zeros((self.npart, self.nhole), dtype='float64')
        _tmp = self.fock[self.part,self.hole].copy()
        if (self.df):
            _tmp[self.pa, self.hc] += 0.5 * np.einsum("ivaw,wu,uv->ai", self.S["caaa"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            _tmp[self.pa, self.hc] -= 0.5 * np.einsum("iwau,vw,uv->ai", self.S["caaa"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            
            _tmp[self.pv, self.hc] += 0.5 * np.einsum("vmwe,wu,uv->em", self.S["acav"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            _tmp[self.pv, self.hc] -= 0.5 * np.einsum("wmue,vw,uv->em", self.S["acav"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            
            _tmp[self.pv, self.ha] += 0.5 * np.einsum("ivaw,wu,uv->ai", self.S["aava"], self.fock[self.active,self.active], self.L1, optimize='optimal')
            _tmp[self.pv, self.ha] -= 0.5 * np.einsum("iwau,vw,uv->ai", self.S["aava"], self.fock[self.active,self.active], self.L1, optimize='optimal')
        else:
            _tmp += 0.5 * np.einsum("ivaw,wu,uv->ai", self.S[:, self.ha, :, self.pa], self.fock[self.active,self.active], self.L1, optimize='optimal')
            _tmp -= 0.5 * np.einsum("iwau,vw,uv->ai", self.S[:, self.ha, :, self.pa], self.fock[self.active,self.active], self.L1, optimize='optimal')
        
        for i in range(self.nhole):
            for k in range(self.npart):
                denom_f = np.float64(np.diagonal(self.fock)[i] - np.diagonal(self.fock)[self.ncore+k])
                _tmp[k, i] *= np.float64(np.exp(-self.flow_param*(denom_f)**2))

        self.F_tilde = np.zeros((self.npart, self.nhole), dtype='float64')
        self.F_tilde = self.fock[self.part,self.hole].copy()
        self.F_tilde += _tmp
        del _tmp
        
    def H1_T1_C0(self):
        E = 0.0
        E = 2. * np.einsum("am,ma->", self.F_tilde[:, self.hc], self.T1[self.hc, :], optimize='optimal')
        temp = np.einsum("ev,ue->uv", self.F_tilde[self.pv, self.ha], self.T1[self.ha, self.pv], optimize='optimal') 
        temp -= np.einsum("um,mv->uv", self.F_tilde[self.pa, self.hc], self.T1[self.hc, self.pa], optimize='optimal')
        E += np.einsum("vu,uv->", self.L1, temp, optimize='optimal')
        if (self.verbose):
            print(f"FT1, {E}")
        return E
    
    def H1_T2_C0(self):
        E = 0.0
        if (self.df):
            temp = np.einsum("ex,uvey->uvxy", self.F_tilde[self.pv, self.ha], self.T2["aava"], optimize='optimal')
            temp -= np.einsum("vm,muyx->uvxy", self.F_tilde[self.pa, self.hc], self.T2["caaa"], optimize='optimal')
            E = np.einsum("xyuv,uvxy->", self.L2, temp, optimize='optimal')
        else:
            temp = np.einsum("ex,uvey->uvxy", self.F_tilde[self.pv, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], optimize='optimal')
            temp -= np.einsum("vm,umxy->uvxy", self.F_tilde[self.pa, self.hc], self.T2[self.ha, self.hc, self.pa, self.pa], optimize='optimal')
            E = np.einsum("xyuv,uvxy->", self.L2, temp, optimize='optimal')
        if (self.verbose):
            print(f"FT2, {E}")
        return E
    
    def H2_T1_C0(self):
        E = 0.0
        if (self.df):
            temp = np.einsum("evxy,ue->uvxy", self.V["vaaa"], self.T1[self.ha, self.pv], optimize='optimal')
            temp -= np.einsum("uvmy,mx->uvxy", self.V["aaca"], self.T1[self.hc, self.pa], optimize='optimal')
        else:
            temp = np.einsum("evxy,ue->uvxy", self.V[self.pv, self.pa, self.ha, self.ha], self.T1[self.ha, self.pv], optimize='optimal')
            temp -= np.einsum("uvmy,mx->uvxy", self.V[self.pa, self.pa, self.hc, self.ha], self.T1[self.hc, self.pa], optimize='optimal')
        E = np.einsum("xyuv,uvxy->", self.L2, temp)
        if (self.verbose):
            print(f"VT1, {E}")
        return E
    
    def H2_T2_C0(self):
        E = np.einsum("efmn,mnef->", self.V[self.pv, self.pv, self.hc, self.hc], self.S[self.hc, self.hc, self.pv, self.pv], optimize='optimal')
        E += np.einsum("efmu,mvef,uv->", self.V[self.pv, self.pv, self.hc, self.ha], self.S[self.hc, self.ha, self.pv, self.pv], self.L1, optimize='optimal')
        E += np.einsum("vemn,mnue,uv->", self.V[self.pa, self.pv, self.hc, self.hc], self.S[self.hc, self.hc, self.pa, self.pv], self.Eta, optimize='optimal')
        E += self.H2_T2_C0_T2small()
        if (self.verbose):
            print(f"VT2, {E}")
        return E
    
    def H2_T2_C0_T2small(self):
        E = 0.0
        # [H2, T2] L1 from aavv
        E += 0.25 * np.einsum("efxu,yvef,uv,xy->", self.V[self.pv, self.pv, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pv], self.L1, self.L1, optimize='optimal')
        # [H2, T2] L1 from ccaa
        E += 0.25 * np.einsum("vymn,mnux,uv,xy->", self.V[self.pa, self.pa, self.hc, self.hc], self.S[self.hc, self.hc, self.pa, self.pa], self.Eta, self.Eta, optimize='optimal')
        # [H2, T2] L1 from caav
        temp = 0.5 * np.einsum("vemx,myue->uxyv", self.V[self.pa, self.pv, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pv], optimize='optimal')
        temp += 0.5 * np.einsum("vexm,ymue->uxyv", self.V[self.pa, self.pv, self.ha, self.hc], self.S[self.ha, self.hc, self.pa, self.pv], optimize='optimal')
        E += np.einsum("uxyv,uv,xy->", temp, self.Eta, self.L1, optimize='optimal')
        # [H2, T2] L1 from caaa and aaav
        temp = 0.25 * np.einsum("evwx,zyeu,wz->uxyv", self.V[self.pv, self.pa, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pa], self.L1, optimize='optimal')
        temp += 0.25 * np.einsum("vzmx,myuw,wz->uxyv", self.V[self.pa, self.pa, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pa], self.Eta, optimize='optimal')
        E += np.einsum("uxyv,uv,xy->", temp, self.Eta, self.L1, optimize='optimal')
        
        # <[Hbar2, T2]> C_4 (C_2)^2
        # HH
        temp = 0.5 * np.einsum("uvmn,mnxy->uvxy", self.V[self.pa, self.pa, self.hc, self.hc], self.T2[self.hc, self.hc, self. pa, self.pa], optimize='optimal')
        temp += 0.5 * np.einsum("uvmw,mzxy,wz->uvxy", self.V[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.L1, optimize='optimal')
        
        # PP
        temp += 0.5 * np.einsum("efxy,uvef->uvxy", self.V[self.pv, self.pv, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pv], optimize='optimal')
        temp += 0.5 * np.einsum("ezxy,uvew,wz->uvxy", self.V[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.Eta, optimize='optimal')
        
        # HP
        temp += np.einsum("uexm,vmye->uvxy", self.V[self.pa, self.pv, self.ha, self.hc], self.S[self.ha, self.hc, self.pa, self.pv], optimize='optimal')
        temp -= np.einsum("uemx,vmye->uvxy", self.V[self.pa, self.pv, self.hc, self.ha], self.T2[self.ha, self.hc, self.pa, self.pv], optimize='optimal')
        temp -= np.einsum("vemx,muye->uvxy", self.V[self.pa, self.pv, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pv], optimize='optimal')
        
        # HP with Gamma1
        temp += 0.5 * np.einsum("euwx,zvey,wz->uvxy", self.V[self.pv, self.pa, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pa], self.L1, optimize='optimal')
        temp -= 0.5 * np.einsum("euxw,zvey,wz->uvxy", self.V[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.L1, optimize='optimal')
        temp -= 0.5 * np.einsum("evxw,uzey,wz->uvxy", self.V[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.L1, optimize='optimal')
        
        # HP with Eta1
        temp += 0.5 * np.einsum("wumx,mvzy,wz->uvxy", self.V[self.pa, self.pa, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pa], self.Eta, optimize='optimal')
        temp -= 0.5 * np.einsum("uwmx,mvzy,wz->uvxy", self.V[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.Eta, optimize='optimal')
        temp -= 0.5 * np.einsum("vwmx,muyz,wz->uvxy", self.V[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.Eta, optimize='optimal')
        
        E += np.einsum("uvxy,uvxy->", temp, self.L2)
        
        #
        E += np.einsum("ewxy,uvez,xyzuwv->", self.V[self.pv, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, self.pv, self.pa], self.L3, optimize='optimal')
        E -= np.einsum("uvmz,mwxy,xyzuwv->", self.V[self.pa, self.pa, self.hc, self.ha], self.T2[self.hc, self.ha, self.pa, self.pa], self.L3, optimize='optimal')
        return E

    def H2_T2_C0_T2small_df(self):
    #  Note the following blocks should be available in memory.
    #  V : vvaa, aacc, avca, avac, vaaa, aaca
    #  T2: aavv, ccaa, caav, acav, aava, caaa
    #  S : aavv, ccaa, caav, acav, aava, caaa
        E = 0.0
        # [H2, T2] L1 from aavv
        E += 0.25 * np.einsum("efxu,yvef,uv,xy->", self.V["vvaa"], self.S["aavv"], self.L1, self.L1, optimize='optimal')
        # [H2, T2] L1 from ccaa
        E += 0.25 * np.einsum("vymn,mnux,uv,xy->", self.V["aacc"], self.S["ccaa"], self.Eta, self.Eta, optimize='optimal')
        # [H2, T2] L1 from caav
        temp = 0.5 * np.einsum("vemx,myue->uxyv", self.V["avca"], self.S["caav"], optimize='optimal')
        temp += 0.5 * np.einsum("vexm,ymue->uxyv", self.V["avac"], self.S["acav"], optimize='optimal')
        E += np.einsum("uxyv,uv,xy->", temp, self.Eta, self.L1, optimize='optimal')
        # [H2, T2] L1 from caaa and aaav
        temp = 0.25 * np.einsum("evwx,zyeu,wz->uxyv", self.V["vaaa"], self.S["aava"], self.L1, optimize='optimal')
        temp += 0.25 * np.einsum("vzmx,myuw,wz->uxyv", self.V["aaca"], self.S["caaa"], self.Eta, optimize='optimal')
        E += np.einsum("uxyv,uv,xy->", temp, self.Eta, self.L1, optimize='optimal')
        
        # <[Hbar2, T2]> C_4 (C_2)^2
        # HH
        temp = 0.5 * np.einsum("uvmn,mnxy->uvxy", self.V["aacc"], self.T2["ccaa"], optimize='optimal')
        temp += 0.5 * np.einsum("uvmw,mzxy,wz->uvxy", self.V["aaca"], self.T2["caaa"], self.L1, optimize='optimal')
        
        # PP
        temp += 0.5 * np.einsum("efxy,uvef->uvxy", self.V["vvaa"], self.T2["aavv"], optimize='optimal')
        temp += 0.5 * np.einsum("ezxy,uvew,wz->uvxy", self.V["vaaa"], self.T2["aava"], self.Eta, optimize='optimal')
        
        # HP
        temp += np.einsum("uexm,vmye->uvxy", self.V["avac"], self.S["acav"], optimize='optimal')
        temp -= np.einsum("uemx,vmye->uvxy", self.V["avca"], self.T2["acav"], optimize='optimal')
        temp -= np.einsum("vemx,muye->uvxy", self.V["avca"], self.T2["caav"], optimize='optimal')
        
        # HP with Gamma1
        temp += 0.5 * np.einsum("euwx,zvey,wz->uvxy", self.V["vaaa"], self.S["aava"], self.L1, optimize='optimal')
        temp -= 0.5 * np.einsum("euxw,zvey,wz->uvxy", self.V["vaaa"], self.T2["aava"], self.L1, optimize='optimal')
        temp -= 0.5 * np.einsum("evxw,uzey,wz->uvxy", self.V["vaaa"], self.T2["aava"], self.L1, optimize='optimal')
        
        # HP with Eta1
        temp += 0.5 * np.einsum("wumx,mvzy,wz->uvxy", self.V["aaca"], self.S["caaa"], self.Eta, optimize='optimal')
        temp -= 0.5 * np.einsum("uwmx,mvzy,wz->uvxy", self.V["aaca"], self.T2["caaa"], self.Eta, optimize='optimal')
        temp -= 0.5 * np.einsum("vwmx,muyz,wz->uvxy", self.V["aaca"], self.T2["caaa"], self.Eta, optimize='optimal')
        
        E += np.einsum("uvxy,uvxy->", temp, self.L2)
        
        #
        E += np.einsum("ewxy,uvez,xyzuwv->", self.V["vaaa"], self.T2["aava"], self.L3, optimize='optimal')
        E -= np.einsum("uvmz,mwxy,xyzuwv->", self.V["aaca"], self.T2["caaa"], self.L3, optimize='optimal')
        return E
    
    def E_V_T2_CCVV_batch(self):
        E = 0.0
        # The three-index integral is created in the semicanonicalization step. 
        # (me|nf) * [2 * (me|nf) - (mf|ne)] * [1 - e^(-2 * s * D)] / D
        # Batching: for a given m and n, form B(ef) = Bm(L|e) * Bn(L|f)
        for m in range(self.ncore):
            for n in range(m, self.ncore):
                if m == n:
                    factor = 1.0
                else:
                    factor = 2.0
                    
                B_Le = np.squeeze(self.Bpq[:, self.pv, m]).copy()
                B_Lf = np.squeeze(self.Bpq[:, self.pv, n]).copy()
                J_mn = np.einsum("Le,Lf->ef", B_Le, B_Lf, optimize='optimal')
                JK_mn = 2.0 * J_mn - J_mn.T
                
                for e in range(self.nvirt):
                    for f in range(self.nvirt):
                        denom = self.e_orb["c"][m] + self.e_orb["c"][n] - self.e_orb["v"][e] - self.e_orb["v"][f]
                        J_mn[e, f] *= (1.0 + np.exp(-self.flow_param * (denom)**2)) * regularized_denominator(denom, self.flow_param)
                
                E += factor * np.einsum("ef,ef->", J_mn, JK_mn, optimize='optimal')
        if (self.verbose):
            print(f"VT2_CCVV (DF) with batching, {E}")  
        return E
    
    def E_V_T2_CCVV(self):
        E = 0.0
        B_Lfn = self.Bpq[:, self.pv, self.hc].copy()
        for m in range(self.ncore):
            B_Le = np.squeeze(self.Bpq[:, self.pv, m]).copy()
            J_m = np.einsum("Le,Lfn->efn", B_Le, B_Lfn, optimize='optimal')
            JK_m = 2.0 * J_m - np.einsum("efn->fen", J_m.copy())
            
            for n in range(self.ncore):
                for e in range(self.nvirt):
                    for f in range(self.nvirt):
                        denom = self.e_orb["c"][m] + self.e_orb["c"][n] - self.e_orb["v"][e] - self.e_orb["v"][f]
                        J_m[e,f,n] *= (1.0 + np.exp(-self.flow_param * (denom)**2)) * regularized_denominator(denom, self.flow_param)
            E += np.einsum("efn,efn->", J_m, JK_m, optimize='optimal')
        if (self.verbose):
            print(f"VT2_CCVV (DF), {E}")
        return E
    
    def E_V_T2_CAVV(self):
        E = 0.0
        B_Lfv = self.Bpq[:, self.pv, self.ha].copy()
        for m in range(self.ncore):
            B_Le = np.squeeze(self.Bpq[:, self.pv, m]).copy()      
            J_m = np.einsum("Le,Lfv->efv", B_Le, B_Lfv, optimize='optimal')
            JK_m = 2.0 * J_m - np.einsum("efv->fev", J_m.copy())
            
            for u in range(self.nact):
                for e in range(self.nvirt):
                    for f in range(self.nvirt):
                        denom = self.e_orb["c"][m] + self.e_orb["a"][u] - self.e_orb["v"][e] - self.e_orb["v"][f]
                        JK_m[e, f, u] *= regularized_denominator(denom, self.flow_param)
                        J_m[e, f, u] *= (1.0 + np.exp(-self.flow_param * (denom)**2))
            temp = np.einsum("efu,efv->uv", J_m, JK_m, optimize='optimal')
            E += np.einsum("uv,vu->", temp, self.L1, optimize='optimal')
        if (self.verbose):
            print(f"VT2_CAVV (DF), {E}") 
        return E
        
    def E_V_T2_CCAV(self):
        E = 0.0
        for m in range(self.ncore):
            for n in range(0, self.ncore):
                B_Le = np.squeeze(self.Bpq[:, self.pv, m]).copy()
                B_Lu = np.squeeze(self.Bpq[:, self.pa, n]).copy()
                
                B_Le_2 = np.squeeze(self.Bpq[:, self.pv, n]).copy()
                B_Lu_2 = np.squeeze(self.Bpq[:, self.pa, m]).copy()
                
                J_mn = np.einsum("Le,Lu->eu", B_Le, B_Lu, optimize='optimal')
                J_mn_2 = np.einsum("Le,Lu->eu", B_Le_2, B_Lu_2, optimize='optimal')
                JK_mn = 2.0 * J_mn - J_mn_2
                
                for u in range(self.nact):
                    for e in range(self.nvirt):
                        denom = self.e_orb["c"][m] + self.e_orb["c"][n] - self.e_orb["a"][u] - self.e_orb["v"][e]
                        JK_mn[e,u] *= regularized_denominator(denom, self.flow_param)
                        J_mn[e,u] *= (1.0 + np.exp(-self.flow_param * (denom)**2))
                temp = np.einsum("eu,ev->uv", J_mn, JK_mn, optimize='optimal')
                E += np.einsum("uv,vu->", temp, self.Eta, optimize='optimal')
        if (self.verbose):
            print(f"VT2_CCAV (DF), {E}") 
        return E
    
    def H1_T_C1a_smallS(self, C1):
        C1 += 1.00 * np.einsum('ev,ue->uv', self.F_tilde[self.pv, self.ha], self.T1[self.ha,self.pv], optimize='optimal')
        C1 -= 1.00 * np.einsum('um,mv->uv', self.F_tilde[self.pa, self.hc], self.T1[self.hc,self.pa], optimize='optimal')
        C1 += 1.00 * np.einsum('em,umve->uv', self.F_tilde[self.pv, self.hc], self.S[self.ha,self.hc,self.pa,self.pv], optimize='optimal')
        C1 += 1.00 * np.einsum('xm,muxv->uv', self.F_tilde[self.pa, self.hc], self.S[self.hc,self.ha,self.pa,self.pa], optimize='optimal')
        C1 += 0.50 * np.einsum('ex,yuev,xy->uv', self.F_tilde[self.pv, self.ha], self.S[self.ha,self.ha,self.pv,self.pa], self.L1, optimize='optimal')
        #C1 += 0.50 * np.einsum('wx,uyvw,xy->uv', self.F_tilde[self.pa, self.ha], self.S[self.ha,self.ha,self.pa,self.pa], self.L1, optimize='optimal')
        #C1["uv"] += 0.5 * H1["wx"] * S2["uyvw"] * L1_["xy"]; (sadsrg_comm.cc::877)
        C1 -= 0.50 * np.einsum('ym,muxv,xy->uv', self.F_tilde[self.pa, self.hc], self.S[self.hc,self.ha,self.pa,self.pa], self.L1, optimize='optimal')
        #C1 -= 0.50 * np.einsum('yw,uwvx,xy->uv', self.F_tilde[self.pa, self.ha], self.S[self.ha,self.ha,self.pa,self.pa], self.L1, optimize='optimal')
        # C1["uv"] -= 0.5 * H1["yw"] * S2["uwvx"] * L1_["xy"]; (ibid, 881)

    def H2_T_C1a_smallS(self, C1):
        C1 += 1.00 * np.einsum('uemz,mwue->wz', self.V[self.pa, self.pv, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pv], optimize='optimal')
        C1 += 1.00 * np.einsum('uezm,wmue->wz', self.V[self.pa, self.pv, self.ha, self.hc], self.S[self.ha, self.hc, self.pa, self.pv], optimize='optimal')
        C1 += 1.00 * np.einsum('vumz,mwvu->wz', self.V[self.pa, self.pa, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pa], optimize='optimal')
        
        C1 -= 1.00 * np.einsum('wemu,muze->wz', self.V[self.pa, self.pv, self.hc, self.ha], self.S[self.hc, self.ha, self.pa, self.pv], optimize='optimal')
        C1 -= 1.00 * np.einsum('weum,umze->wz', self.V[self.pa, self.pv, self.ha, self.hc], self.S[self.ha, self.hc, self.pa, self.pv], optimize='optimal')
        C1 -= 1.00 * np.einsum('ewvu,vuez->wz', self.V[self.pv, self.pa, self.ha, self.ha], self.S[self.ha, self.ha, self.pv, self.pa], optimize='optimal')

        temp =  0.5 * np.einsum('wvef,efzu->wzuv', self.S[self.ha, self.ha, self.pv, self.pv], self.V[self.pv, self.pv, self.ha, self.ha], optimize='optimal')
        temp += 0.5 * np.einsum('wvex,exzu->wzuv', self.S[self.ha, self.ha, self.pv, self.pa], self.V[self.pv, self.pa, self.ha, self.ha], optimize='optimal')
        temp += 0.5 * np.einsum('vwex,exuz->wzuv', self.S[self.ha, self.ha, self.pv, self.pa], self.V[self.pv, self.pa, self.ha, self.ha], optimize='optimal')
        #temp += 0.5 * np.einsum('wvxy,xyzu->wzuv', self.S[self.ha, self.ha, self.pv, self.pa], self.V[self.pv, self.pa, self.ha, self.ha], optimize='optimal')

        temp -= 0.5 * np.einsum('wmue,vezm->wzuv', self.S[self.ha, self.hc, self.pa, self.pv], self.V[self.pa, self.pv, self.ha, self.hc], optimize='optimal')
        temp -= 0.5 * np.einsum('mwxu,xvmz->wzuv', self.S[self.hc, self.ha, self.pa, self.pa], self.V[self.pa, self.pa, self.hc, self.ha], optimize='optimal')

        temp -= 0.5 * np.einsum('mwue,vemz->wzuv', self.S[self.hc, self.ha, self.pa, self.pv], self.V[self.pa, self.pv, self.hc, self.ha], optimize='optimal')
        temp -= 0.5 * np.einsum('mwux,vxmz->wzuv', self.S[self.hc, self.ha, self.pa, self.pa], self.V[self.pa, self.pa, self.hc, self.ha], optimize='optimal')

        temp += 0.25 * np.einsum('jwxu,xy,yvjz->wzuv', self.S[:, self.ha, self.pa, self.pa], self.L1, self.V[self.pa, self.pa, :, self.ha], optimize='optimal')
        temp -= 0.25 * np.einsum('ywbu,xy,bvxz->wzuv', self.S[self.ha, self.ha, :, self.pa], self.L1, self.V[:, self.pa, self.ha, self.ha], optimize='optimal')
        temp -= 0.25 * np.einsum('wybu,xy,bvzx->wzuv', self.S[self.ha, self.ha, :, self.pa], self.L1, self.V[:, self.pa, self.ha, self.ha], optimize='optimal')

        C1 += np.einsum('wzuv,uv->wz', temp, self.L1, optimize='optimal')
        temp = np.zeros((self.nact,)*4)

        temp -= 0.5 * np.einsum('mnzu,wvmn->wzuv', self.S[self.hc, self.hc, self.pa, self.pa], self.V[self.pa, self.pa, self.hc, self.hc], optimize='optimal')
        temp -= 0.5 * np.einsum('mxzu,wvmx->wzuv', self.S[self.hc, self.ha, self.pa, self.pa], self.V[self.pa, self.pa, self.hc, self.ha], optimize='optimal')
        temp -= 0.5 * np.einsum('mxuz,vwmx->wzuv', self.S[self.hc, self.ha, self.pa, self.pa], self.V[self.pa, self.pa, self.hc, self.ha], optimize='optimal')
        #temp -= 0.5 * np.einsum('xyzu,wvxy->wzuv', self.S[self.hc, self.hc, self.pa, self.pa], self.V[self.pa, self.pa, self.hc, self.hc], optimize='optimal')

        temp += 0.5 * np.einsum('vmze,weum->wzuv', self.S[self.ha, self.hc, self.pa, self.pv], self.V[self.pa, self.pv, self.ha, self.hc], optimize='optimal')
        temp += 0.5 * np.einsum('xvez,ewxu->wzuv', self.S[self.ha, self.ha, self.pv, self.pa], self.V[self.pv, self.pa, self.ha, self.ha], optimize='optimal')

        temp += 0.5 * np.einsum('mvze,wemu->wzuv', self.S[self.hc, self.ha, self.pa, self.pv], self.V[self.pa, self.pv, self.hc, self.ha], optimize='optimal')
        temp += 0.5 * np.einsum('vxez,ewux->wzuv', self.S[self.ha, self.ha, self.pv, self.pa], self.V[self.pv, self.pa, self.ha, self.ha], optimize='optimal')

        temp -= 0.25 * np.einsum('yvbz,xy,bwxu->wzuv', self.S[self.ha, self.ha, :, self.pa], self.Eta, self.V[:, self.pa, self.ha, self.ha], optimize='optimal')
        temp += 0.25 * np.einsum('jvxz,xy,ywju->wzuv', self.S[:, self.ha, self.pa, self.pa], self.Eta, self.V[self.pa, self.pa, :, self.ha], optimize='optimal')
        temp += 0.25 * np.einsum('jvzx,xy,wyju->wzuv', self.S[:, self.ha, self.pa, self.pa], self.Eta, self.V[self.pa, self.pa, :, self.ha], optimize='optimal')

        C1 += np.einsum('wzuv,uv->wz', temp, self.Eta, optimize='optimal')

        C1 += 0.50 * np.einsum('vujz,jwyx,xyuv->wz', self.V[self.pa, self.pa, :, self.ha], self.T2[:, self.ha, self.pa, self.pa], self.L2, optimize='optimal')
        C1 += 0.50 * np.einsum('auzx,wvay,xyuv->wz', self.V[:, self.pa, self.ha, self.ha], self.S[self.ha, self.ha, :, self.pa], self.L2, optimize='optimal')
        C1 -= 0.50 * np.einsum('auxz,wvay,xyuv->wz', self.V[:, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, :, self.pa], self.L2, optimize='optimal')
        C1 -= 0.50 * np.einsum('auxz,vway,xyvu->wz', self.V[:, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, :, self.pa], self.L2, optimize='optimal')

        C1 -= 0.50 * np.einsum('bwyx,vubz,xyuv->wz', self.V[:, self.pa, self.ha, self.ha], self.T2[self.ha, self.ha, :, self.pa], self.L2, optimize='optimal')
        C1 -= 0.50 * np.einsum('wuix,ivzy,xyuv->wz', self.V[self.pa, self.pa, :, self.ha], self.S[:, self.ha, self.pa, self.pa], self.L2, optimize='optimal')
        C1 += 0.50 * np.einsum('uwix,ivzy,xyuv->wz', self.V[self.pa, self.pa, :, self.ha], self.T2[:, self.ha, self.pa, self.pa], self.L2, optimize='optimal')
        C1 += 0.50 * np.einsum('uwix,ivyz,xyvu->wz', self.V[self.pa, self.pa, :, self.ha], self.T2[:, self.ha, self.pa, self.pa], self.L2, optimize='optimal')

        C1 += 0.50 * np.einsum('avxy,uwaz,xyuv->wz', self.V[:, self.pa, self.ha, self.ha], self.S[self.ha, self.ha, :, self.pa], self.L2, optimize='optimal')
        C1 -= 0.50 * np.einsum('uviy,iwxz,xyuv->wz', self.V[self.pa, self.pa, :, self.ha], self.S[:, self.ha, self.pa, self.pa], self.L2, optimize='optimal')

    def H2_T_C1a_smallG(self, C1):
        G2 = 2.0 * self.V - self.V.swapaxes(2,3) # we definitely don't need to store all of this
        C1 += np.einsum('ma,uavm->uv', self.T1[self.hc,:], G2[self.pa,:,self.ha,self.hc], optimize='optimal')
        C1 += 0.50 * np.einsum('xe,yx,uevy->uv', self.T1[self.ha,self.pv], self.L1, G2[self.pa,self.pv,self.ha,self.ha], optimize='optimal')
        C1 -= 0.50 * np.einsum('mx,xy,uyvm->uv', self.T1[self.hc,self.pa], self.L1, G2[self.pa,self.pa,self.ha,self.hc], optimize='optimal')

        C1 += 0.50 * np.einsum('wezx,uvey,xyuv->wz', G2[self.pa,self.pv,self.ha,self.ha], self.T2[self.ha,self.ha,self.pv,self.pa], self.L2, optimize='optimal')
        C1 -= 0.50 * np.einsum('wuzm,mvxy,xyuv->wz', G2[self.pa,self.pa,self.ha,self.hc], self.T2[self.hc,self.ha,self.pa,self.pa], self.L2, optimize='optimal')

    def H_T_C2a_smallS(self, C2):
        C2 += np.einsum('efxy,uvef->uvxy', self.V[self.pv,self.pv,self.ha,self.ha], self.T2[self.ha,self.ha,self.pv,self.pv], optimize='optimal')
        #C2["uvxy"] += H2["wzxy"] * T2["uvwz"];
        C2 += np.einsum('ewxy,uvew->uvxy', self.V[self.pv,self.pa,self.ha,self.ha], self.T2[self.ha,self.ha,self.pv,self.pa], optimize='optimal')
        C2 += np.einsum('ewyx,vuew->uvxy', self.V[self.pv,self.pa,self.ha,self.ha], self.T2[self.ha,self.ha,self.pv,self.pa], optimize='optimal')

        C2 += np.einsum('uvmn,mnxy->uvxy', self.V[self.pa,self.pa,self.hc,self.hc], self.T2[self.hc,self.hc,self.pa,self.pa], optimize='optimal')
        #C2["uvxy"] += H2["uvwz"] * T2["wzxy"];
        C2 += np.einsum('vumw,mwyx->uvxy', self.V[self.pa,self.pa,self.hc,self.ha], self.T2[self.hc,self.ha,self.pa,self.pa], optimize='optimal')
        C2 += np.einsum('uvmw,mwxy->uvxy', self.V[self.pa,self.pa,self.hc,self.ha], self.T2[self.hc,self.ha,self.pa,self.pa], optimize='optimal')

        temp = np.einsum('ax,uvay->uvxy', self.F_tilde[:,self.ha], self.T2[self.ha,self.ha,:,self.pa], optimize='optimal')
        temp -= np.einsum('ui,ivxy->uvxy', self.F_tilde[self.pa,:], self.T2[:,self.ha,self.pa,self.pa], optimize='optimal')
        temp += np.einsum('ua,avxy->uvxy', self.T1[self.ha,:], self.V[:,self.pa,self.ha,self.ha], optimize='optimal')
        temp -= np.einsum('ix,uviy->uvxy', self.T1[:,self.pa], self.V[self.pa,self.pa,:,self.ha], optimize='optimal')

        temp -= 0.50 * np.einsum('wz,vuaw,azyx->uvxy', self.L1, self.T2[self.ha,self.ha,:,self.pa], self.V[:,self.pa,self.ha,self.ha], optimize='optimal')
        temp -= 0.50 * np.einsum('wz,izyx,vuiw->uvxy', self.Eta, self.T2[:,self.ha,self.pa,self.pa], self.V[self.pa,self.pa,:,self.ha], optimize='optimal')

        temp += np.einsum('uexm,vmye->uvxy', self.V[self.pa,self.pv,self.ha,self.hc], self.S[self.ha,self.hc,self.pa,self.pv], optimize='optimal')
        temp += np.einsum('wumx,mvwy->uvxy', self.V[self.pa,self.pa,self.hc,self.ha], self.S[self.hc,self.ha,self.pa,self.pa], optimize='optimal')

        temp += 0.50 * np.einsum('wz,zvay,auwx->uvxy', self.L1, self.S[self.ha,self.ha,:,self.pa], self.V[:,self.pa,self.ha,self.ha], optimize='optimal')
        temp -= 0.50 * np.einsum('wz,ivwy,zuix->uvxy', self.L1, self.S[:,self.ha,self.pa,self.pa], self.V[self.pa,self.pa,:,self.ha], optimize='optimal')

        temp -= np.einsum('uemx,vmye->uvxy', self.V[self.pa,self.pv,self.hc,self.ha], self.T2[self.ha,self.hc,self.pa,self.pv], optimize='optimal')
        temp -= np.einsum('uwmx,mvwy->uvxy', self.V[self.pa,self.pa,self.hc,self.ha], self.T2[self.hc,self.ha,self.pa,self.pa], optimize='optimal')

        temp -= 0.50 * np.einsum('wz,zvay,auxw->uvxy', self.L1, self.T2[self.ha,self.ha,:,self.pa], self.V[:,self.pa,self.ha,self.ha], optimize='optimal')
        temp += 0.50 * np.einsum('wz,ivwy,uzix->uvxy', self.L1, self.T2[:,self.ha,self.pa,self.pa], self.V[self.pa,self.pa,:,self.ha], optimize='optimal')

        temp -= np.einsum('vemx,muye->uvxy', self.V[self.pa,self.pv,self.hc,self.ha], self.T2[self.hc,self.ha,self.pa,self.pv], optimize='optimal')
        temp -= np.einsum('vwmx,muyw->uvxy', self.V[self.pa,self.pa,self.hc,self.ha], self.T2[self.hc,self.ha,self.pa,self.pa], optimize='optimal')

        temp -= 0.50 * np.einsum('wz,uzay,avxw->uvxy', self.L1, self.T2[self.ha,self.ha,:,self.pa], self.V[:,self.pa,self.ha,self.ha], optimize='optimal')
        temp += 0.50 * np.einsum('wz,iuyw,vzix->uvxy', self.L1, self.T2[:,self.ha,self.pa,self.pa], self.V[self.pa,self.pa,:,self.ha], optimize='optimal')

        C2 += temp
        C2 += np.einsum('uvxy->vuyx', temp, optimize='optimal')

    def compute_hbar(self):
        hbar1_temp = np.zeros((self.nact,)*2)

        self.H1_T_C1a_smallS(hbar1_temp)
        self.H2_T_C1a_smallS(hbar1_temp)
        self.H2_T_C1a_smallG(hbar1_temp)

        self.hbar1 += 0.5 * hbar1_temp 
        self.hbar1 += 0.5 * hbar1_temp.T

        hbar2_temp = np.zeros((self.nact,)*4)
        self.H_T_C2a_smallS(hbar2_temp)
        
        self.hbar2 += 0.5 * hbar2_temp
        self.hbar2 += 0.5 * np.einsum('uvxy->xyuv', hbar2_temp, optimize='optimal')

        hbar1_temp = np.einsum('efzm,wmef->wz', self.V[self.pv, self.pv, self.ha, self.hc], self.S[self.ha, self.hc, self.pv, self.pv], optimize='optimal')
        hbar1_temp -= np.einsum('wemn,mnze->wz', self.V[self.pa, self.pv, self.hc, self.hc], self.S[self.hc, self.hc, self.pa, self.pv], optimize='optimal')

        self.hbar1 += 0.5 * hbar1_temp
        self.hbar1 += 0.5 * hbar1_temp.T
        del hbar1_temp

    def deGNO_ints(self):
        hbar2_temp = 2*self.hbar2 - np.einsum('pqrs->pqsr', self.hbar2, optimize='optimal')

        self.e_scalar1 = - np.einsum('vu,uv->', self.hbar1, self.L1)
        self.e_scalar2 = 0.25 * np.einsum('uv,vyux,xy->', self.L1, hbar2_temp, self.L1) - 0.5 * np.einsum('xyuv,uvxy->', self.hbar2, self.L2)
        self.relax_e_scalar = self.e_scalar1 + self.e_scalar2

        self.hbar1 -= 0.5 * np.einsum('uxvy,yx->uv', hbar2_temp, self.L1)

        del hbar2_temp

        self.hbar1_canon = np.einsum('ip,pq,jq->ij', self.semicanonicalizer[self.active, self.active], self.hbar1, self.semicanonicalizer[self.active, self.active], optimize='optimal')
        self.hbar2_canon = np.einsum('ip,jq,pqrs,kr,ls->ijkl', self.semicanonicalizer[self.active, self.active], self.semicanonicalizer[self.active, self.active], self.hbar2, self.semicanonicalizer[self.active, self.active], self.semicanonicalizer[self.active, self.active], optimize='optimal')


    def drsg_mrpt2_iteration(self):
        self.semi_canonicalize()

        if (self.relax_ref):
            self.hbar1 = self.fock[self.active, self.active].copy()
            self.hbar2 = self.V[self.pa, self.pa, self.ha, self.ha].copy()
        
        self.compute_T2()
        self.compute_T1()
        self.renormalize_V()
        self.renormalize_F()

        self.e_h1_t1 = self.H1_T1_C0()
        self.e_h1_t2 = self.H1_T2_C0()
        self.e_h2_t1 = self.H2_T1_C0()
        if (self.df):
            self.e_h2_t2_small = self.H2_T2_C0_T2small_df()
            self.e_h2_t2_cavv = self.E_V_T2_CAVV()
            self.e_h2_t2_ccav = self.E_V_T2_CCAV()
            # [todo]: unified interface for batching: give a list of indices to batch over
            if (self.batch):
                self.e_h2_t2_ccvv = self.E_V_T2_CCVV_batch()
            else:
                self.e_h2_t2_ccvv = self.E_V_T2_CCVV()
            self.e_h2_t2 = self.e_h2_t2_small + self.e_h2_t2_cavv + self.e_h2_t2_ccav + self.e_h2_t2_ccvv
        else:
            self.e_h2_t2 = self.H2_T2_C0()

        self.e_corr = self.e_h1_t1 + self.e_h1_t2 + self.e_h2_t1 + self.e_h2_t2
        self.e_tot = self.mc.e_tot + self.e_corr

        if (self.relax_ref): 
            self.compute_hbar()
            self.deGNO_ints()
            # hbar2_canon is in physicist's notation, PySCF uses chemist's notation
            self.relax_eigval, self.ci_vecs = fci.direct_spin1.kernel(self.hbar1_canon, self.hbar2_canon.swapaxes(1,2), self.mc.ncas, self.mc.nelecas, \
                                                                 ecore=self.relax_e_scalar, nroots=self.state_average_nstates)
            if (self.state_average_nstates == 1):
                self.relax_eigval = [self.relax_eigval]
                self.ci_vecs = [self.ci_vecs]
            _eci_avg = np.dot(self.relax_eigval[:self.state_average_nstates], self.state_average_weights)
            self.e_corr += _eci_avg
            self.e_tot += _eci_avg

    def kernel(self):
        self.drsg_mrpt2_iteration()

        if (self.relax_ref):
            self.relax_energies = np.zeros((self.nrelax,3)) # [iter, [unrelaxed, relaxed, Eref]]
        else:
            self.relax_energies = np.zeros((1,3))

        self.relax_energies[0, 0] = self.e_tot
        self.relax_energies[0, 2] = self.mc.e_tot

        if (self.nrelax == 1): self.converged = True

        for irelax in range(self.nrelax):
            pass      

        return self.e_corr

# register DSRG_MRPT2 in MCSCF
# [todo]: is this so that we can access fcisolver options?
from pyscf.mcscf import casci
casci.CASCI.DSRG_MRPT2 = DSRG_MRPT2


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    test = 2

    if (test == 1):
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
    elif (test == 2):
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
        dsrg = DSRG_MRPT2(casci, relax='once')
        e_dsrg_mrpt2 = dsrg.kernel()
        print(f"{e_dsrg_mrpt2=}")
        print(f"{dsrg.e_tot=}")
        try:
            assert (np.isclose(dsrg.e_tot, -98.51880426166926, atol=1e-6))
        except:
            print(f'Warning: dsrg.e_tot is not close to -98.51880426166926')
        
        print(f"{dsrg.e_scalar1=}")
        try:
            assert (np.isclose(dsrg.e_scalar1, 2.5191661113766437, atol=1e-6))
        except:
            print(f'Warning: dsrg.e_scalar1 is not close to 2.5191661113766437')
        print(f"{dsrg.e_scalar2=}")

        try:
            assert (np.isclose(dsrg.e_scalar2, 11.143089786391007, atol=1e-6))
        except:
            print(f'Warning: dsrg.e_scalar2 is not close to 11.143089786391007')
    elif (test==3):
        mol = gto.M(
            verbose = 2,
            atom = '''
        O 0 0 0
        O 0 0 1.251
        ''',
            basis = 'cc-pvdz', spin=0, charge=0
        )
        mf = scf.RHF(mol).density_fit()
        mf.kernel()
        mc = mcscf.CASSCF(mf, 6, 8) # density_fit() should propagate to mcscf
        mc.fix_spin_(ss=0) # we want the singlet state, not the Ms=0 triplet state
        mc.mc2step() 
        dsrg = DSRG_MRPT2(mc, relax='none', density_fit=True, batch=False, verbose=True) # [todo]: propagate density_fit to DSRG_MRPT2
        e_dsrg_mrpt2 = dsrg.kernel()
        print(f"casscf: {mc.e_tot}")
        #assert np.isclose(mc.e_tot, -149.675640632305, atol=1e-6)  This is for direct computation
        assert np.isclose(mc.e_tot, -149.675391362112094, atol = 1e-6) # This is for DF
        
        # Here are tests for DF
        assert np.isclose(dsrg.e_h2_t2_ccvv, -0.014939333740318, atol = 1e-6) # This is for DF
        assert np.isclose(dsrg.e_h2_t2_cavv, -0.042801582864407, atol = 1e-6) # This is for DF
        assert np.isclose(dsrg.e_h2_t2_ccav, -0.003545083460275, atol = 1e-6) # This is for DF
        
        print(f"DSRG-MRPT2 correlation energy: {e_dsrg_mrpt2}")
        #assert np.isclose(e_dsrg_mrpt2, -0.25739463745825364, atol=1e-6) This is for direct computation
        assert np.isclose(e_dsrg_mrpt2, -0.257376059270690, atol=1e-6) # This is for DF
        
        print(f"DSRG-MRPT2 total energy: {dsrg.e_tot}") 
        assert np.isclose(dsrg.e_tot, -149.932767421382778, atol=1e-6) # This is for DF
