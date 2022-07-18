"""
    Script to syncronize several video streams from several kinects and an omnidir. fisheye camera.
"""
# First of all, let this module find others.
import sys
sys.path.append('..')

from glob import glob
import cv2
import re
from utils import syncro
from config import load_config as conf
import os

# Reduce image size by set percent (to see all capture in one screen)
show_ratio=0.5

# Calculate milliseconds per frame.
ms_frame = (1000./30.)  # 2*Hz of recording.

# Load the configuration file, where the working directory is set.
config = conf.load_config()
work_dir = config['base_dir']

# Find Kinect captures (capture0, capture1, etc.)
folders = next(os.walk(work_dir))[1]
captures = sorted([os.path.join(work_dir, f) for f in folders if 'capture' in f])
print(f'Found {len(captures)} Kinect captures.')

# Kinect RGB images
kinect_rgb_files = [sorted(glob(f'{capture}/rgb/*.jpg')) for capture in captures]
for idx, k in enumerate(kinect_rgb_files):
    print(f'  Kinect {idx} captured {len(k)} RGB images')

# Omnicam RGB images
if 'omni' in folders:
    print('Found "omni" capture.')
    omni_rgb_files = sorted(glob(work_dir + '/omni/*.jpg'))
    print('  Omnicam captured %d RGB images' % len(omni_rgb_files))
else:
    print('ERROR: No "omni" folder found in working directory.')

# Find initial and final timestamps for omnicam:
omni_start = int(os.path.split(omni_rgb_files[0])[1].replace('.jpg', ''))
omni_end = int(os.path.split(omni_rgb_files[-1])[1].replace('.jpg', ''))

kinect_starts = [int(os.path.split(kinect_rgb_files[idx][0])[1].replace('.jpg', '')) for idx in range(len(captures))]
kinect_ends = [int(os.path.split(kinect_rgb_files[idx][-1])[1].replace('.jpg', '')) for idx in range(len(captures))]

# Get the first timestamp which has images for all cameras.
timestamp = omni_start

# Difference between omni and kinects (ok_diff)
#ok_diff = 800
ok_diff=0
# Timestamps for each device
ts_o = timestamp + ok_diff
ts_k = [timestamp for idx in range(len(captures))]

k_index = [syncro.find_min_ts_diff_image(kinect_rgb_files[idx], ts_k[idx]) for idx in range(len(captures))]
o_index = syncro.find_min_ts_diff_image(omni_rgb_files, ts_o)

space_pressed = False
fo = None

sel_kinect = 0

k_selected = [None for idx in range(len(captures))]
k_img = [None for idx in range(len(captures))]
k_ir_selected = ["" for idx in range(len(captures))]

while True:

    for idx in range(len(captures)):
        if k_index[idx] >= 0:
            k_selected[idx] = kinect_rgb_files[idx][k_index[idx]]
            k_img[idx] = cv2.imread(k_selected[idx])
            cv2.imshow(f'kinect_{idx}', cv2.resize(k_img[idx],None,fx=show_ratio,fy=show_ratio))

    o_selected = omni_rgb_files[o_index]
    o_img = cv2.imread(o_selected)
    cv2.imshow('omnicam', cv2.resize(o_img,None,fx=show_ratio,fy=show_ratio))

    ch = cv2.waitKey(0)
    if ch & 0xFF == 27:
        break
    elif ch >= ord('0') and ch <= ord('9'):
        num = ch - ord('0')
        if num < len(captures):
            print(f'Selected Kinect {num}')
            sel_kinect = num
        else:
            print('No such Kinect capture.')
    elif ch == ord('q'):
        ts_k[sel_kinect] -= ms_frame
    elif ch == ord('e'):
        ts_k[sel_kinect] += ms_frame
    elif ch == ord('z'):
        ts_o -= ms_frame
    elif ch == ord('c'):
        ts_o += ms_frame
    elif ch == ord(' '):
        if not space_pressed:
            fo = open(work_dir + '/shots.txt', 'w')
            fo.write(f'{ok_diff};')
            for idx in range(len(captures)):
                fo.write(f'{ts_k[idx]/1000.};')
            fo.write(f'{ts_o/1000.:.3f}\n')
            space_pressed = True
        if fo:
            for idx in range(len(captures)):
                if k_index[idx] >= 0:
                    k_ir_selected[idx] = k_selected[idx].replace('/rgb/', '/ir/')
                    fo.write(f'{k_ir_selected[idx]};')
            fo.write(f'{o_selected}\n')
            print(k_ir_selected)
            print(o_selected)
    elif ch == ord('j'):
        print('back 100ms')
        ts_k = [val-100 for val in ts_k]
        ts_o -= 100
    elif ch == ord('l'):
        print('fwd. 100ms')
        ts_k = [val+100 for val in ts_k]
        ts_o += 100
    elif ch == ord('m'):
        print('back 1 frame')
        ts_o -= ms_frame
        ts_k = [val-ms_frame for val in ts_k]
    elif ch == ord('.'):
        print('fwd. 1 frame')
        ts_o += ms_frame
        ts_k = [val+ms_frame for val in ts_k]

    k_index = [syncro.find_min_ts_diff_image(kinect_rgb_files[idx], ts_k[idx]) for idx in range(len(captures))]
    o_index = syncro.find_min_ts_diff_image(omni_rgb_files, ts_o)

if fo:
    fo.close()
