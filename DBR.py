# just the function - use DBR.ipynb for computations

import numpy as np

def layer_matrix(n,d,lam): 
    # This gives the Abelès matrix of a layer 
    # see https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics)
    delta = 2*np.pi *n*d/lam # the width in wavelength 
    eta = n # admittance (= n for normal incidence - complex in most general case)
    M = np.array([
        [np.cos(delta),1j*np.sin(delta)/eta],
        [1j*eta*np.sin(delta),np.cos(delta)]])
    return M

def stack_matrix(layers,lam):
    # layer is a list 
    # (E,H) continuous at interface - product in order from incident layer to last
    M = np.eye(2,dtype=complex)
    for n,d in layers:
        M = M @ layer_matrix(n,d,lam)
    return M

def reflectance(layers,n_out,n_sub,lam):
    # out is outside (air) - sub is substrate (GaN or gold?)
    eta_out = n_out # same as above
    eta_sub = n_sub
    M = stack_matrix(layers,lam)
    # BC for substrate : only transmitted so H = eta*E so (E,H) = t*(1,eta)
    E_out = M[0,0]+M[0,1]*eta_sub
    H_out = M[1,0]+M[1,1]*eta_sub
    r = (eta_out*E_out - H_out)/(eta_out*E_out + H_out) # solved system see notes
    R = abs(r)**2 
    T = 4*eta_out.real*eta_sub.real/abs(eta_out*E_out + H_out)**2 # same system
    return R,T,r

def build_DBR(n_high,n_low,n_pairs,lam0):
    # Just outputs list of layers (n_high faces incident - n_low faces substrate)
    d_high = lam0/(4*n_high)
    d_low  = lam0/(4*n_low)
    layers = []
    for _ in range(n_pairs):
        layers.append((n_high,d_high))
        layers.append((n_low,d_low))
    return layers

def build_microcavity(mirror_top,mirror_bot,n_cav,lam0,m=1):
    # top/bottom from build_DBR (both in (high,low) order so low faces cavity)
    d_cav = m*lam0/(2*n_cav) # m factor to have multiples of lam/2 width
    return mirror_top + [(n_cav,d_cav)] + mirror_bot[::-1]

def sweep(layers,n_out,n_sub,lams):
    # (R,T) for every wavelength in lams
    R = np.empty_like(lams)
    T = np.empty_like(lams)
    for i,lam in enumerate(lams):
        R[i],T[i],_ = reflectance(layers,n_out,n_sub,lam)
    return R,T

def stopband(lams,R,threshold = 0.95):
    # width of the high-R plateau : range where R stays above threshold
    above = np.where(R >= threshold)[0]
    if len(above) == 0:
        return None, None, 0.0
    lower,higher = lams[above[0]],lams[above[-1]]
    return lower,higher,higher-lower

def cavity_dip(layers,n_out,n_sub,lam0,span=2e-9,n_coarse=2001,n_fine=40001):
    # locate the cavity resonance (R minimum) near lam0 and measure it
    # coarse scan - zoom - FWHM
    lc = np.linspace(lam0*0.98,lam0*1.02,n_coarse)
    Rc,_ = sweep(layers,n_out,n_sub,lc)
    lam_c = lc[Rc.argmin()]
    lf = np.linspace(lam_c-span,lam_c+span,n_fine)
    Rf,_ = sweep(layers,n_out,n_sub,lf)
    k = Rf.argmin()
    lam_c = lf[k]
    plateau = Rf.max()
    half = Rf[k] + 0.5*(plateau-Rf[k])
    left  = lf[:k][Rf[:k] >= half][-1] # last plateau point left
    right = lf[k:][Rf[k:] >= half][0] # first plateau point right
    fwhm = right - left
    Q = lam_c/fwhm
    return lam_c,fwhm,Q,Rf[k]