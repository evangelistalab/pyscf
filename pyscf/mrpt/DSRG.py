import pyscf
from pyscf import mp, mcscf
import numpy as np
import scipy as sp
import time
import os, glob
import warnings
import itertools

MACHEPS = 1e-9
Taylor_threshold = 1e-3

def antisymmetrize_and_hermitize(T):
    # antisymmetrize the residual
    T += np.einsum("ijab->abij",T.conj()) # This is the Hermitized version (i.e., [H,A]), which should then be antisymmetrized
    temp = T.copy()
    T -= np.einsum("ijab->jiab", temp)
    T += np.einsum("ijab->jiba", temp)
    T -= np.einsum("ijab->ijba", temp)
    
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
    
def get_SF_RDM(ci_vec, nelec, norb):
    dm1, dm2, dm3 = pyscf.fci.rdm.make_dm123('FCI3pdm_kern_sf', ci_vec, ci_vec, norb, nelec)
    dm1, dm2, dm3 = pyscf.fci.rdm.reorder_dm123(dm1, dm2, dm3)
    G1 = np.einsum("pq -> qp", dm1)
    G2 = np.einsum("pqrs -> prqs", dm2)
    G3 = np.einsum("pqrstu -> prtqsu", dm3)
    return G1, G2, G3
# Pyscf store reordered rdms as: dm2[pqrs] = <p\dagger r\dagger s q> 
# reorder dm2[pqrs] -> dm[prsq] will give you normal rdm. 
# HOWEVER, Forte store <p\dagger r\dagger s q> as rdm[prqs]. Therefore, if you are following Forte code, you will make mistake.

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
    
class DSRG:
    def __init__(self, mol, verbose = True, density_fitting = False, decontract = False):
        # Do df or not?
        if (type(density_fitting) is bool):
            self.density_fitting = density_fitting
            self.df_basis = None
        elif(type(density_fitting) is str):
            self.density_fitting = True
            self.df_basis = density_fitting
        # Decontract basis?
        self.decontract = decontract
        if (self.decontract):
            self.mol, _ = mol.decontract_basis()
        else:
            self.mol = mol
        # Basic information
        self.nuclear_repulsion = self.mol.energy_nuc()
        self.nelec = sum(self.mol.nelec)
        self.nao = self.mol.nao # Number of basis function
        self.nocc = int( 0.5 * self.nelec) # This is somewhat wrong. Only even number of electrons are considered here.
        self.nuocc = self.nao - self.nocc
        self.verbose = verbose
    def ao2mo(self, mo_coeff): # frozen core should be added later
        t0 = time.time()
        print('Building integrals...')
        
        rhf_hcore_ao = self.mol.intor_symmetric('int1e_kin') + self.mol.intor_symmetric('int1e_nuc')
        self.rhf_hcore_mo = np.einsum('pi,pq,qj->ij', mo_coeff, rhf_hcore_ao, mo_coeff)
        
        rhf_eri_ao = self.mol.intor('int2e_sph', aosym='s1') # Chemist's notation (\mu\nu|\lambda\rho)
        self.rhf_eri_mo = pyscf.ao2mo.incore.full(rhf_eri_ao, mo_coeff, False) # (pq|rs)

        if (self.verbose):
            t1 = time.time()
            print(f'Integral build time:         {t1 - t0:15.7f} s')
            print('-'*47)
    
    def run_rhf(self, transform=False, debug=True):
        ti = time.time()
        if (self.verbose):
            print('='*47)
            print('{:^47}'.format('PySCF RHF interface'))
            print('='*47)
        if (self.density_fitting):
            if (self.verbose): print('{:#^47}'.format('Enabling density fitting!')) 
            self.rhf = pyscf.scf.RHF(self.mol).density_fit() if self.df_basis is None else pyscf.scf.RHF(self.mol).density_fit(self.df_basis)
        else:
            self.rhf = pyscf.scf.RHF(self.mol)
        
        self.rhf_energy = self.rhf.kernel()
        
        if (self.verbose): print(f"Non-relativistic RHF Energy: {self.rhf_energy:15.7f} Eh")
        
        t1 = time.time()
        
        print(f'PySCF RHF time:              {t1-ti:15.7f} s')
        print('-'*47)
        
        if (transform):
            self.ao2mo(self.rhf.mo_coeff)

            if (debug and self.verbose):
                self.rhf_e1 = 2 * np.einsum('ii->',self.rhf_hcore_mo[:self.nocc, :self.nocc])
                self.rhf_e2 = 2 * np.einsum('iijj->',self.rhf_eri_mo[:self.nocc, :self.nocc, :self.nocc, :self.nocc]) - np.einsum('ijji->',self.rhf_eri_mo[:self.nocc, :self.nocc, :self.nocc, :self.nocc])          
                self.rhf_e_rebuilt = self.rhf_e1 + self.rhf_e2 + self.nuclear_repulsion
                print(f"Rebuilt RHF Energy:          {self.rhf_e_rebuilt.real:15.7f} Eh")
                print(f"Error to PySCF:              {np.abs(self.rhf_e_rebuilt.real - self.rhf_energy):15.7f} Eh")
                print('-'*47)
            
            if (self.verbose):
                tf = time.time()
                print(f'RHF time:                    {tf - ti:15.7f} s')
                print('='*47)
    
    def run_casscf(self, cas, transform = True): # cas is (e, o)
        ti = time.time()
        if (self.verbose):
            print('='*47)
            print('{:^47}'.format('PySCF CASSCF interface'))
            print('='*47)
        
        self.casscf = pyscf.mcscf.CASSCF(self.rhf, cas[1], cas[0])
        self.casscf.conv_tol = 1e-8
        self.casscf.conv_tol_grad = 1e-7
        #self.casscf.max_stepsize = 0.03
        self.res = self.casscf.mc2step() 
        #self.res = self.casscf.kernel() 
        self.e_casscf = self.res[0]

        if (transform):
            self.ao2mo(self.res[3])

            if (self.verbose):
                tf = time.time()
                print(f'CASSCF time:                 {tf - ti:15.7f} s')
                print(f"CASSCF Energy:               {self.e_casscf.real:15.7f} Eh")
                print('='*47)
    
    def run_casci(self, cas, transform = True): # cas is (e, o)
        ti = time.time()
        if (self.verbose):
            print('='*47)
            print('{:^47}'.format('PySCF CASCI interface'))
            print('='*47)
        
        self.casscf = pyscf.mcscf.CASCI(self.rhf, cas[1], cas[0])
        self.casscf.conv_tol = 1e-8
        self.casscf.conv_tol_grad = 1e-7
        #self.casscf.max_stepsize = 0.03
        self.res = self.casscf.kernel() 
        self.e_casscf = self.res[0]

        if (transform):
            self.ao2mo(self.res[3])

            if (self.verbose):
                tf = time.time()
                print(f'CASCI time:                 {tf - ti:15.7f} s')
                print(f"CASCI Energy:               {self.e_casscf.real:15.7f} Eh")
                print('='*47)
            
                        
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
                        
    def dsrg_mrpt2(self, cas, s):
        self.ncore = int(0.5 * (self.nelec - cas[0]))
        self.nact = cas[1]
        self.nvirt = self.nao - self.ncore - self.nact
        self.nhole = self.ncore + self.nact
        self.npart = self.nact + self.nvirt
        
        self.core = slice(0, self.ncore)
        self.active = slice(self.ncore, self.ncore + self.nact)
        self.virt = slice(self.ncore + self.nact, self.nao)
        self.hole = slice(0, self.ncore + self.nact)
        self.part = slice(self.ncore, self.nao)
        
        self.hc = self.core
        self.ha = self.active
        self.pa = slice(0,self.nact)
        self.pv = slice(self.nact, self.nact + self.nvirt)
        
        #F in canonical basis
        F = np.einsum("pi, pq, qj->ij", self.res[3], self.casscf.get_fock(), self.res[3], dtype='float64', optimize='optimal')
        
        #Semicanonicalization
        self.semicanonicalizer = np.zeros((self.nao, self.nao), dtype='float64')
        _, self.semicanonicalizer[self.core,self.core] = np.linalg.eigh(F[self.core,self.core])
        _, self.semicanonicalizer[self.active,self.active] = np.linalg.eigh(F[self.active,self.active])
        _, self.semicanonicalizer[self.virt,self.virt] = np.linalg.eigh(F[self.virt,self.virt])
        

        # RDMs in semi-canonical basis.
        G1, G2, G3 = get_SF_RDM(self.res[2], *cas)
        self.G1 = np.einsum("pi, pq, qj->ij", self.semicanonicalizer[self.active,self.active], G1, self.semicanonicalizer[self.active,self.active], dtype='float64', optimize='optimal')
        self.G2 = np.einsum("pi, qj, rk, sl, pqrs->ijkl", self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], G2, dtype='float64', optimize='optimal')
        self.G3 = np.einsum("pi, qj, rk, sl, tm, un, pqrstu->ijklmn", self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], self.semicanonicalizer[self.active,self.active], G3, dtype='float64', optimize='optimal')
        self.Eta = 2. * np.identity(self.nact) - self.G1
        self.L1 = self.G1.copy() # Remove this.
        self.L2 = get_SF_cu2(self.G1, self.G2)
        self.L3 = get_SF_cu3(self.G1, self.G2, self.G3)
        
    
        # F and V in semi-canonical basis.
        self.F = np.einsum("pi, pq, qj->ij", self.semicanonicalizer, F, self.semicanonicalizer, dtype='float64', optimize='optimal')
        
        F_diag_active = self.F[self.active,self.active].copy()

        tmp= self.rhf_eri_mo[self.part, self.hole, self.part, self.hole].copy()
        tmp= np.einsum("aibj->abij", tmp, dtype='float64')
        
        self.V = np.einsum("pi, qj, pqrs, rk, sl->ijkl", self.semicanonicalizer[self.part, self.part], self.semicanonicalizer[self.part, self.part], tmp, self.semicanonicalizer[self.hole, self.hole], self.semicanonicalizer[self.hole, self.hole], dtype='float64', optimize='optimal') 
        
        # # T amplitudes
        e_orb = np.diagonal(self.F)

        self.T2 = np.einsum("abij -> ijab", self.V.copy())
        for i in range(self.nhole):
            for j in range(self.nhole):
                for k in range(self.npart):
                    for l in range(self.npart):
                        denom = np.float64(e_orb[i] + e_orb[j] - e_orb[self.ncore+k] - e_orb[self.ncore+l])
                        self.T2[i, j, k, l] *= np.float64(regularized_denominator(denom, s))
                            
        
        self.T2[self.ha,self.ha,self.pa,self.pa] = 0
         
        self.S = 2.0 * self.T2.copy() - np.einsum("ijab->ijba", self.T2.copy(), dtype='float64', optimize='optimal')
        
        
        self.T1 = np.zeros((self.nhole, self.npart), dtype='float64')
        self.T1 = self.F[self.hole,self.part].copy()
        self.T1  += 0.5 * np.einsum('ivaw, wu, uv->ia', self.S[:, self.ha, :, self.pa], F_diag_active, self.G1, dtype='float64', optimize='optimal')
        self.T1 -= 0.5 * np.einsum('iwau, vw, uv->ia', self.S[:, self.ha, :, self.pa], F_diag_active, self.G1, dtype='float64', optimize='optimal')
        
        for i in range(self.nhole):
            for k in range(self.npart):
                denom_t1 = np.float64(e_orb[i] - e_orb[self.ncore+k])
                self.T1[i, k] *= np.float64(regularized_denominator(denom_t1, s))
        
        self.T1[self.ha,self.pa] = 0
        
        
        # F_tilde and V_tilde
        self.V_tilde = np.zeros((self.npart, self.npart, self.nhole, self.nhole), dtype='float64')
        self.V_tilde = self.V.copy()
        for k in range(self.npart):
            for l in range(self.npart):
                for i in range(self.nhole):
                    for j in range(self.nhole):
                        denom = np.float64(e_orb[i] + e_orb[j] - e_orb[self.ncore+k] - e_orb[self.ncore+l])
                        self.V_tilde[k, l, i, j] *= np.float64(1. + np.exp(-s*(denom)**2))
                        
        
        tmp_f = np.zeros((self.npart, self.nhole), dtype='float64')
        tmp_f = self.F[self.part,self.hole].copy()
        tmp_f += 0.5 * np.einsum("ivaw, wu, uv->ai", self.S[:, self.ha, :, self.pa], F_diag_active, self.G1, dtype='float64', optimize='optimal')
        tmp_f -= 0.5 * np.einsum("iwau, vw, uv->ai", self.S[:, self.ha, :, self.pa], F_diag_active, self.G1, dtype='float64', optimize='optimal')
        
        for i in range(self.nhole):
            for k in range(self.npart):
                denom_f = np.float64(e_orb[i] - e_orb[self.ncore+k])
                tmp_f[k, i] *= np.float64(np.exp(-s*(denom_f)**2))
                   
        
        self.F_tilde = np.zeros((self.npart, self.nhole), dtype='float64')
        self.F_tilde = self.F[self.part,self.hole].copy()
        self.F_tilde += tmp_f

        # Energy 
        E1 = self.H1_T1_C0()
        E2 = self.H1_T2_C0()
        E3 = self.H2_T1_C0()
        E4, Eout = self.H2_T2_C0()
        self.e_dsrg = self.e_casscf + E1 + E2 + E3 + E4
        print(f"DSRG-MRPT2 Energy:               {self.e_dsrg:15.7f} Eh")
        print('='*47)
        
if (__name__=='__main__'):
    mol = pyscf.gto.M(
        verbose = 2,
        atom = '''
    H 0.0 0.0 0.0
    F 0.0 1.5 0.0
    ''',
        basis = 'cc-pvdz', spin=0, charge=0, symmetry=False
    )
    a = DSRG(mol, verbose=True, density_fitting=False, decontract=False)
    a.run_rhf(transform=False, debug=True)
    a.run_casscf(cas=(6,4), transform=True)
    a.dsrg_mrpt2(cas=(6,4), s = 1.0)