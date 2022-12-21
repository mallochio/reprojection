"""
Script to load 'stereoParams' objects from MATLAB json-encoded files.
"""
import json
import numpy as np
from config import load_config as conf

config = conf.load_config()

def _load_calibrations(filename):
    """
    Loads stereo calibration parameters from JSON encoded file into dictionary
    :param filename: path to JSON-encoded MATLAB stereo calibration object file
    :return: dictionary containing parameters loaded from file
    """
    fd = open(filename, 'r')
    params = json.load(fd)
    fd.close()
    return params


def _parse_stereo_matrices(params):
    """
    Reads params dictionary and prepares important matrices
    :param params: parameters returned by load_calibrations()
    :return: dictionary containing numpy arrays of loaded matrices
    """
    dict = {}

    K1 = np.asarray(params["CameraParameters1"]["IntrinsicMatrix"]).transpose()
    Dk = np.asarray(params["CameraParameters1"]["RadialDistortion"])
    Dp = np.asarray(params["CameraParameters1"]["TangentialDistortion"])
    if len(Dk) == 3:
        D1 = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1], Dk[2]])
    else:
        D1 = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1]])
    dict["K1"] = K1
    dict["D1"] = D1
    dict['k1_size'] = tuple(params["CameraParameters1"]["ImageSize"])

    K2 = np.asarray(params["CameraParameters2"]["IntrinsicMatrix"]).transpose()
    Dk = np.asarray(params["CameraParameters2"]["RadialDistortion"])
    Dp = np.asarray(params["CameraParameters2"]["TangentialDistortion"])
    if len(Dk) == 3:
        D2 = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1], Dk[2]])
    else:
        D2 = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1]])
    dict["K2"] = K2
    dict["D2"] = D2
    dict['k2_size'] = tuple(params["CameraParameters2"]["ImageSize"])

    Rot = np.asarray(params["RotationOfCamera2"]).transpose()
    Trans = np.asarray(params["TranslationOfCamera2"])

    dict["rot"] = Rot
    dict["trans"] = Trans

    Fundamental = np.asarray(params["FundamentalMatrix"]).transpose()
    Essential = np.asarray(params["EssentialMatrix"]).transpose()

    dict["F"] = Fundamental
    dict["E"] = Essential

    fx, fy = tuple(params["CameraParameters1"]["FocalLength"])
    cx, cy = tuple(params["CameraParameters1"]["PrincipalPoint"])
    dict["k1_params"] = (fx, fy, cx, cy)

    fx, fy = tuple(params["CameraParameters2"]["FocalLength"])
    cx, cy = tuple(params["CameraParameters2"]["PrincipalPoint"])
    dict["k2_params"] = (fx, fy, cx, cy)

    RR1 = np.asarray(params["CameraParameters1"]["RotationMatrices"])
    dict["RR1"] = [RR1[:,:,i].T for i in range(RR1.shape[2])]

    RR2 = np.asarray(params["CameraParameters2"]["RotationMatrices"])
    dict["RR2"] = [RR2[:,:,i].T for i in range(RR2.shape[2])]

    tt1 = np.asarray(params["CameraParameters1"]["TranslationVectors"])
    dict["tt1"] = [tt1[i] for i in range(tt1.shape[0])]

    tt2 = np.asarray(params["CameraParameters2"]["TranslationVectors"])
    dict["tt2"] = [tt2[i] for i in range(tt2.shape[0])]

    return dict


def get_stereo_calibration_matrices(filename):
    params = _load_calibrations(filename)
    matrices = _parse_stereo_matrices(params)
    return matrices


def _parse_mono_matrices(params):
    dict = {}

    fx, fy = tuple(params["FocalLength"])
    cx, cy = tuple(params["PrincipalPoint"])
    dict["k_params"] = (fx, fy, cx, cy)

    dict['K'] = np.asarray(params['IntrinsicMatrix']).T

    Dk = np.asarray(params["RadialDistortion"])
    Dp = np.asarray(params["TangentialDistortion"])
    if len(Dk) == 3:
        D = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1], Dk[2]])
    else:
        D = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1]])
    dict['D'] = D

    RR = np.asarray(params["RotationMatrices"])
    dict["RR"] = [RR[:, :, i].T for i in range(RR.shape[2])]

    tt = np.asarray(params["TranslationVectors"])
    dict["tt"] = [tt[i] for i in range(tt.shape[0])]

    return dict


def get_mono_calibration_matrices(filename):
    params = _load_calibrations(filename)
    matrices = _parse_mono_matrices(params)
    return matrices


def _parse_omni_parameters_matlab(params):
    omni = {}

    c0, c2, c3, c4 = params['Intrinsics']['MappingCoefficients']
    omni['Coeffs'] = np.array([c0, 0., c2, c3, c4])
    cx, cy = params['Intrinsics']['DistortionCenter']
    omni['Centre'] = (cx, cy)
    m = np.asarray(params['Intrinsics']['StretchMatrix'])
    omni['c'] = m[0, 0]
    omni['d'] = m[0, 1]
    omni['e'] = m[1, 0]

    RR = np.asarray(params["RotationMatrices"])
    omni["RR"] = [RR[:, :, i].T for i in range(RR.shape[2])]

    tt = np.asarray(params["TranslationVectors"])
    omni["tt"] = [tt[i] for i in range(tt.shape[0])]

    return omni


def _parse_omni_parameters(params):
    pass
    # omni = {}
    # omni['R'] = np.asarray(params['R'])
    # omni['T'] = np.asarray(params['T'])
    # omni['K'] = np.asarray(params['intrinsics'])
    # omni['D'] = np.asarray(params['distortion'])
    # omni['xi'] = np.asarray(params['xi'])
    # return omni

def get_omni_calibration_matrices(ix):
    params = config['omni_params']
    return _parse_omni_parameters(params)

def get_omni_calibration_matrices_old(filename):
    dico = _load_calibrations(filename)
    params = _parse_omni_parameters(dico)
    return params


if __name__ == '__main__':
    get_omni_calibration_matrices('../out/omniParams.json')
