import numpy as np

def findinvpoly(coeffs, radius):
    maxerr = np.inf
    N = 1
    while maxerr > 0.01:
        N += 1
        pol, err = findinvpoly2(coeffs, radius, N)
        maxerr = np.max(err)
    
    return pol, err, N


def findinvpoly2(coeffs, radius, N):
    theta = np.arange(-np.pi, 1.20, step=0.01)
    r = inv_fun(coeffs, theta, radius)
    ind = np.argwhere(r != np.inf)
    theta = theta[ind]
    r = r[ind]

    pol = np.polyfit(theta, r, N)
    err = np.abs(r - np.polyval(pol, theta))

    return pol, err, N


def inv_fun(coeffs, theta, radius):

    m = np.tan(theta)

    r = np.zeros(shape=m.shape)
    poly_coeff = np.flip(coeffs).copy()
    poly_coeff_tmp = poly_coeff.copy()
    for j in range(len(m)):
        poly_coeff_tmp[-1] = poly_coeff[-1] - m[j]
        rho_tmp = np.roots(poly_coeff_tmp)
        rho_ind = np.argwhere(np.imag(rho_tmp) == 0 and rho_tmp>0 and rho_tmp<radius)
        res = rho_tmp[rho_ind]
        if np.sum(res) == 0 or len(res)>1:
            r[j] = np.inf
        else:
            r[j] = res
    
    return r
