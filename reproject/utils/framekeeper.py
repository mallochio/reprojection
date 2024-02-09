from os import path
import re
import numpy as np
import os
import cv2
from utils import syncro
from utils import load_matlab_calibration as lomat
from glob import glob
from random import sample
from tqdm import tqdm


def _wrap_image(image, new_h, new_w):
    """
    Creates a wrapped image of new_h x new_w with image image in the centre.
    :param image: The image to centre
    :param new_h: height of the new bigger image
    :param new_w: width of the new bigger image
    :return: A wrapped image, i.e. one with the new dimensions and "image" centred on it.
    """
    dims = image.shape
    bigger = None
    if len(dims) == 3:
        bigger = np.zeros((new_h, new_w, dims[2]), dtype=image.dtype)
    elif len(dims) == 2:
        bigger = np.zeros((new_h, new_w), dtype=image.dtype)
    sh, sw = dims[0], dims[1]
    uh, uw = new_h, new_w
    bigger[int(uh / 2 - sh / 2):int(uh / 2 + sh / 2), int(uw / 2 - sw / 2):int(uw / 2 + sw / 2)] = image
    return bigger


def _calculate_median_background_depth(depth_images, base_dir, k_idx):
    """
    Calculates mask as the median of depth_images in stream
    :param depth_images: the depth images to calculate the median on
    :return: The median of all or a 500 sample of images
    """
    cachedfile = path.join(base_dir, 'median%d.png' % k_idx)
    if not path.exists(cachedfile):
        print('Sampling and median ...')
        if len(depth_images) > 500:
            list_images = sample(depth_images, 499)
        else:
            list_images = depth_images

        first = cv2.imread(depth_images[0], cv2.CV_16UC1)
        images = np.zeros(shape=(len(list_images), first.shape[0], first.shape[1]))
        for i, image_file in tqdm(enumerate(list_images)):
            image = np.float32(cv2.imread(image_file, cv2.CV_16UC1))
            if image is not None:
                images[i, :, :] = cv2.flip(image, 1)
            else:
                print("Warning: Some sampled depth images could not be loaded!")

        e_depth = np.median(images, axis=0)
        print('Caching to file ...')
        cv2.imwrite(cachedfile, np.uint16(e_depth))
        print('Done.')
    else:
        print('Loading cached file for median depth ...')
        e_depth = np.float32(cv2.imread(cachedfile, cv2.CV_16UC1))
        print('Done.')

    return e_depth


def _get_transformation_matrix(R1, T1, R2, T2):
    """
    Given the rotations and translations of two cameras w.r.t. a board plane,
    calculates the transformation between the two views.
    :param R1: Rotation of camera 1 (omni) wrt the board
    :param T1: Translation of camera 1 wrt the board
    :param R2: Rotation of camera 2 (infrared) wrt the board
    :param T2: Translation of camera 2 wrt the board
    :return: A Transformation matrix Ts, for reprojection of points.
    """
    meo = np.zeros((4, 4))
    meo[0:3, 0:3] = R1
    meo[:3, 3] = T1 / 1000.
    meo[3, :] = np.asarray([0, 0, 0, 1])

    mei = np.zeros((4, 4))
    mei[0:3, 0:3] = R2
    mei[:3, 3] = T2 / 1000.
    mei[3, :] = np.asarray([0, 0, 0, 1])

    Ts = np.matmul(meo, np.linalg.pinv(mei))
    return Ts


class FrameKeeper(object):
    """
    A frame keeper is a nice class to store frame collections and retrieve synchronized frames
    """
    def __init__(self, base_dir, capture_Hz):
        self._step_ms = 1000./(2*capture_Hz)  # TODO: Replace, and calculate from SLOWEST (lower FPS) camera.
        self._framesets = {}
        self._frame_timestamps = {}
        self._cvcodes = {}
        self._ts_diffs = {}
        self._ts_start = 0
        self.num_kinects = 0
        self.empty_depth = {}
        self.kinect_params = {}
        self.Ts = {}

        print("Loading recording under '%s' ..." % base_dir)
        # Load calibration + syncronisation data
        syncro_data = syncro.get_timestamp_differences(base_dir)
        for capture_name in syncro_data:
            print("Found capture '%s' ..." % capture_name)

            if 'omni' in capture_name:
                file_set = sorted(glob('%s/%s/*' % (base_dir, capture_name)))[1:]
                self._ts_start = syncro_data[capture_name]['timestamp']  # Omni is the lead camera.
                self._ts_diffs[capture_name] = syncro_data[capture_name]['diff_to_lead']
                self._add_frameset(capture_name, file_set, cv2.CV_8UC3)

            elif 'capture' in capture_name:
                # Estimate kinect index (k_idx) based on directory name
                k_idx = int(capture_name.replace('capture', ''))

                # Load the frame set for the depth images
                file_set = sorted(glob('%s/%s/depth/*' % (base_dir, capture_name)))[1:]
                self._ts_diffs[capture_name] = syncro_data[capture_name]['diff_to_lead']
                self._add_frameset(capture_name, file_set, cv2.CV_16UC1)

                # Load the frame set for the densepose masks
                file_set_dp = sorted(glob('%s/%s/depth-dp-masks/*' % (base_dir, capture_name)))[1:]
                if len(file_set_dp) > 0:
                    capture_mod_name = '_%s_rgb_densepose' % capture_name
                    self._ts_diffs[capture_mod_name] = syncro_data[capture_name]['diff_to_lead']
                    self._add_frameset(capture_mod_name, file_set_dp, cv2.CV_8UC3)

                # Load the kinect camera parameters
                self.kinect_params[k_idx] = lomat.get_mono_calibration_matrices('%s/k%dParams.json' % (base_dir, k_idx))

                # Empty depth images, necessary for movement mask calculation
                e_depth = _calculate_median_background_depth(file_set, base_dir, k_idx)
                e_depth = cv2.undistort(e_depth, self.kinect_params[k_idx]['K'], self.kinect_params[k_idx]['D'])
                self.empty_depth[k_idx] = np.float32(e_depth)
                self.num_kinects += 1

            else:
                print("Error: Malformed sync data file. Capture name '%s' unrecognised." % capture_name)
                exit(-1)
            print("Retrieved %d frames for capture '%s'." % (len(self._framesets[capture_name]), capture_name))
        print("Found %d kinect captures in total." % self.num_kinects)

        # Obtain transformation matrix for omni-kinect camera pair
        self.omni_params = {}

        if path.exists('%s/omni0Params.json' % base_dir):
            for k_idx in range(self.num_kinects):
                # Load the omnidirectional camera parameters
                self.omni_params[k_idx] = lomat.get_omni_calibration_matrices('%s/omni%dParams.json' % (base_dir, k_idx))
        else:
            print("WARNING! No separate rotation files for each Kinect (wrt Omni)")
            for k_idx in range(self.num_kinects):
                self.omni_params[k_idx] = lomat.get_omni_calibration_matrices('%s/omniParams.json' % base_dir)

        for k_idx in range(self.num_kinects):
            self.Ts[k_idx] = _get_transformation_matrix(self.omni_params[k_idx]['RR'][0],
                                                        self.omni_params[k_idx]['tt'][0],
                                                        self.kinect_params[k_idx]['RR'][0],
                                                        self.kinect_params[k_idx]['tt'][0])

    def get_step_ms(self):
        return self._step_ms

    def _add_frameset(self, name, fileset, cvcode):
        self._framesets[name] = fileset
        self._cvcodes[name] = cvcode
        self._cache_frameset(name)

    def _cache_frameset(self, name):
        self._frame_timestamps[name] = np.zeros(len(self._framesets[name]))
        for i, filepath in enumerate(self._framesets[name]):
            dirpath, filename = path.split(filepath)
            timestamp = int(re.split('\.', filename)[0])
            self._frame_timestamps[name][i] = timestamp / 1000.
        
    def _find_min_diff_ts_frame(self, name, timestamp):
        differences = np.abs(self._frame_timestamps[name] - timestamp)
        return np.argmin(differences)

    def get_span(self):
        ts0 = []
        tsf = []
        for name in self._frame_timestamps:
            ts0.append(self._frame_timestamps[name][0])
            tsf.append(self._frame_timestamps[name][-1])
        return max(ts0), min(tsf)

    def get_lead_span(self):
        ts0, tsf = self.get_span()
        timestamps = []
        for ts in self._frame_timestamps['omni']:
            if ts0 <= ts <= tsf:
               timestamps.append(ts)
        return timestamps

    def get_syncro_frames(self, timestamp, debug=False):
        frames = {}
        frame_files = {}
        for name in self._framesets:
            ts_diff = self._ts_diffs[name]
            frame_idx = self._find_min_diff_ts_frame(name, timestamp+ts_diff)
            filename = self._framesets[name][frame_idx]
            frame_files[name] = filename
            if 'rgb' in name or 'omni' in name:
                frames[name] = cv2.imread(filename)
            else:
                frames[name] = cv2.imread(filename, self._cvcodes[name])  # TODO could convert to float here.
            if debug:
                print('%-10s: %s' % (name, filename))
        if debug:
            print("-" * 30)

        return frames, frame_files
