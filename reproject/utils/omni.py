import numpy as np
from multiprocessing import Pool

def findinvpoly(coeffs, radius):
    maxerr = np.inf
    N = 460
    while maxerr > 0.01:
        N += 1
        pol, err, _ = findinvpoly2(coeffs, radius, N)
        maxerr = np.max(err)
        print(N, maxerr)
        # break
    
    return pol, err, N


def findinvpoly2(coeffs, radius, N):
    theta = np.arange(-np.pi/2., 1.20, step=0.01)
    r = inv_fun(coeffs, theta, radius)
    theta = theta[r != np.inf]
    r = r[r != np.inf]
    
    pol = np.polyfit(theta, r, N)
    v = np.polyval(pol, theta)
    err = np.abs(r - v)

    return pol, err, N


def inv_fun(coeffs, theta, radius):

    m = np.tan(theta)

    r = np.zeros(shape=m.shape)
    poly_coeff = np.flip(coeffs).copy()
    poly_coeff_tmp = poly_coeff.copy()
    for j in range(len(m)):
        poly_coeff_tmp[-2] = poly_coeff[-2] - m[j]
        rho_tmp = np.roots(poly_coeff_tmp)
        rho_ind = np.argwhere(np.logical_and(np.imag(rho_tmp) == 0, rho_tmp>0, rho_tmp<radius))
        res = rho_tmp[rho_ind]
        if res.size == 0 or res.size > 1:
            r[j] = np.inf
        else:
            r[j] = np.real(res[0][0])
    
    return r


def correct_d_distortion(Xn, Yn, dist_params):
    # Distortion correction vectors
    k1, k2, k3, k4, k5 = dist_params
    r2 = Xn ** 2 + Yn ** 2
    rad_distortion = (1 + (k1 * r2) + (k2 * r2 * r2) + (k5 * r2 * r2 * r2))
    x_distortion = (2 * k3 * Xn * Yn) + k4 * (r2 + 2 * Xn * Xn)
    y_distortion = k3 * (r2 + 2 * Yn * Yn) + (2 * k4 * Xn * Yn)
    Xk = rad_distortion * Xn + x_distortion
    Yk = rad_distortion * Yn + y_distortion
    return Xk, Yk


def correct_k_distortion(Xk, Yk, K):
    fx, fy, cx, cy = K

    Xu = fx * Xk + cx
    Yu = fy * Yk + cy

    return Xu, Yu


def world_to_omni(Ts, world_coordinates, K_omni_params, D_omni, uw, uh):
    omni_coordinates = np.matmul(Ts, world_coordinates.T)
    X, Y = omni_coordinates[0, :] / omni_coordinates[2, :], omni_coordinates[1, :] / omni_coordinates[2, :]

    X, Y = correct_d_distortion(X, Y, D_omni)
    X, Y = correct_k_distortion(X, Y, K_omni_params)

    Y[Y < 0] = 0
    X[X < 0] = 0
    Y[np.round(Y) >= uh] = uh - 1
    X[np.round(X) >= uw] = uw - 1

    return X, Y


def calculate_rho(poly_temp):
    rho_temp = np.roots(poly_temp)
    res = np.logical_and(np.imag(rho_temp) == 0, rho_temp > 0)
    if np.sum(res) == 0:
        rho = np.nan
    elif np.sum(res) > 1:
        rho = np.min(rho_temp[res])
    else:
        rho = rho_temp[res][0]
    return rho


pool = Pool(4)


def omni3dtopixel(X, Y, Z, omni_params):
    eps = 1e-9
    X[X == 0] = eps
    Y[Y == 0] = eps

    r = np.sqrt((X*X) + (Y*Y))
    m = Z / r

    rho = np.zeros((len(m)))

    poly_coeff = np.flip(omni_params['Coeffs']).copy()
    poly_temp = [poly_coeff.copy() for j in range(len(m))]
    for j in range(len(m)):
        poly_temp[j][-2] = poly_coeff[-2] - m[j]

    rho = pool.map(calculate_rho, poly_temp)

    x = X / r * rho
    y = Y / r * rho
    return x, y


def world_to_omni_scaramuzza(Ts, world_coordinates, ocam_intrinsics, uw, uh):
    if len(world_coordinates) == 0:
        return [0], [0]

    omni_coordinates = np.matmul(Ts, world_coordinates.T)
    max_z = omni_coordinates[2, :].max()
    X, Y, Z = omni_coordinates[0, :], omni_coordinates[1, :], omni_coordinates[2, :]

    #indices = (Z < max_z - 0.05)

    #X = X[indices]
    #Y = Y[indices]
    #Z = Z[indices]

    x, y = omni3dtopixel(X, Y, Z, ocam_intrinsics)

    c, d, e = ocam_intrinsics['c'], ocam_intrinsics['d'], ocam_intrinsics['e']
    xc, yc = ocam_intrinsics['Centre']

    x = x * c + y * d + xc
    y = x * e + y + yc

    y[y < 0] = 0
    x[x < 0] = 0
    y[np.round(y) >= uh] = uh - 1
    x[np.round(x) >= uw] = uw - 1

    return x, y


def world_to_omni_scaramuzza_fast(Ts, world_coordinates, ocam_intrinsics, uw, uh):
    if len(world_coordinates) == 0:
        return [0], [0]
    
    omni_coordinates = np.matmul(Ts, world_coordinates.T)
    max_z = omni_coordinates[2, :].max()
    X, Y, Z = omni_coordinates[0, :], omni_coordinates[1, :], omni_coordinates[2, :]

    c, d, e = ocam_intrinsics['c'], ocam_intrinsics['d'], ocam_intrinsics['e']
    xc, yc = ocam_intrinsics['Centre']

    if not 'Poly' in ocam_intrinsics:
        ss = ocam_intrinsics['Coeffs']
        radius = np.sqrt((uw/2)**2+(uh/2)**2)
        ocam_intrinsics['Poly'], err, n = findinvpoly(ss, radius)

    pol = ocam_intrinsics['Poly']

    theta = np.zeros(shape=X.shape)
    norm = np.sqrt(X**2 + Y**2)

    norm[norm==0] = 1e-9  # eps

    theta = np.arctan(Z/norm)

    rho = np.polyval(pol, theta)

    x = X/norm*rho
    y = Y/norm*rho

    x = x * c + y * d + xc
    y = x * e + y + yc

    y[y < 0] = 0
    x[x < 0] = 0
    y[np.round(y) >= uh] = uh - 1
    x[np.round(x) >= uw] = uw - 1

    return x, y