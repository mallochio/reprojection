"""
    Script to syncronize several video streams from several kinects and an omnidir. fisheye camera.
"""
# First of all, let this module find others.
import sys
sys.path.append('..')

import subprocess
from glob import glob
import cv2
from sync.utils import syncro
from config import load_config as conf
import os
from rich import print

# Reduce image size by set percent (to see all capture in one screen)
show_ratio=0.5

# Calculate milliseconds per frame.
ms_frame = (1000./30.)  # 2*Hz of recording.

# Load the configuration file, where the working directory is set.
config = conf.load_config()
work_dir = config['base_dir']

# Find Kinect captures (capture0, capture1, etc.)
folders = next(os.walk(work_dir))[1]
print(folders)
captures = sorted([os.path.join(work_dir, f) for f in folders if 'capture' in f])
print(f'Found {len(captures)} Kinect captures.')

# Kinect RGB images
kinect_rgb_files = [sorted(glob(f'{capture}/rgb/*.jpg')) for capture in captures]
for idx, k in enumerate(kinect_rgb_files):
    print(f'  Kinect {idx} captured {len(k)} RGB images')

kinect_starts = [int(os.path.split(kinect_rgb_files[idx][0])[1].replace('.jpg', '')) for idx in range(len(captures))]
kinect_ends = [int(os.path.split(kinect_rgb_files[idx][-1])[1].replace('.jpg', '')) for idx in range(len(captures))]

# Get the first timestamp which has images for all cameras.
timestamp = max(kinect_starts)

# Difference between omni and kinects (ok_diff)
ok_diff=0

# Timestamps for each device
ts_k = [timestamp for idx in range(len(captures))]

k_index = [syncro.find_min_ts_diff_image(kinect_rgb_files[idx], ts_k[idx]) for idx in range(len(captures))]

space_pressed = False
hyphen_pressed = False
fo = None

sel_kinect = 0

k_selected = [None for idx in range(len(captures))]
k_img = [None for idx in range(len(captures))]
k_ir_selected = ["" for idx in range(len(captures))]

# Frames from the Ego cam
aux_frames = -1
ego_id = 9
actual_frame = 0
ego_selected = False
ego_fps = 0
url = glob(work_dir + '/ego/*.MP4')

if (len(url) > 0):
    creation_mp4_time = int(os.path.getctime(url[0]))
    mp4_cap = cv2.VideoCapture(url[0])
    total_frames = int(mp4_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ego_fps = int(mp4_cap.get(cv2.CAP_PROP_FPS))
    

while True:
    for idx in range(len(captures)):
        if k_index[idx] >= 0:
            k_selected[idx] = kinect_rgb_files[idx][k_index[idx]]
            k_img[idx] = cv2.imread(k_selected[idx])
            if (k_img[idx] is not None):
                cv2.imshow(f'kinect_{idx}', cv2.resize(k_img[idx],None,fx=show_ratio,fy=show_ratio))

    if (len(url) > 0 and actual_frame >= 0 and actual_frame < total_frames and actual_frame != aux_frames):
        mp4_cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
        isTrue, frame = mp4_cap.read()
        if (frame is not None):
            cv2.imshow('Ego', cv2.resize(frame,None,fx=0.15,fy=0.15))
            
    aux_frames = actual_frame
    ch = cv2.waitKey(0)
    if ch & 0xFF == 27:
        break

    elif ch >= ord('0') and ch <= ord('9'):
        num = ch - ord('0')
        if num == ego_id and len(url) > 0:
            ego_selected = True
            print('Selected Ego.')
        else:
            if num < len(captures):
                print(f'Selected Kinect {num}')
                sel_kinect = num
                ego_selected = False
            else:
                print('No such Kinect capture.')

    elif ch == ord('q'):
        ts_k[sel_kinect] -= ms_frame

    elif ch == ord('e'):
        ts_k[sel_kinect] += ms_frame

    elif ch == ord('z'):
        print('back 100ms')
        ts_k = [val-100 for val in ts_k]
        actual_frame -= 3

    elif ch == ord('c'):
        print('fwd. 100ms')
        ts_k = [val+100 for val in ts_k]
        actual_frame += 3

    elif ch == ord('-'):
        if not hyphen_pressed:    
            hyphen_pressed = True     
            if ego_selected:
                fo = open(work_dir + '/shots_wearable.txt', 'w')
                fo.write(f'{ego_id};{actual_frame};{url[0]}')
                print("Ego saved", ego_id, actual_frame, url[0])
            else:
                fo = open(work_dir + '/shots_wearable.txt', 'w')
                fo.write(f'{sel_kinect};{ts_k[sel_kinect]/1000.};{k_selected[sel_kinect]}')
                print("Kinect saved", sel_kinect, ts_k[sel_kinect]/1000., {k_selected[sel_kinect]})
                subprocess.run(['date', '--utc', f'--date=@{ts_k[sel_kinect]/1000.}'])

    elif ch == ord(' '):
        if not space_pressed:
            if len(url) > 0:
                fo = open(work_dir + '/shots_ego.txt', 'w')
                fo.write(f'{ts_k[0]/1000.};{actual_frame}\n')
                fo.write(f'{k_selected[0]};{url[0]}')
                space_pressed = True
                print(f'The shots_ego.txt have been saved with Capture_0 {ts_k[0]/1000.} and Ego {actual_frame}')
            else:
                print(f"There is no Ego folder")

    elif ch == ord('j'):
        print('back 1000ms')
        ts_k = [val-1000 for val in ts_k]
        actual_frame -= ego_fps

    elif ch == ord('l'):
        print('fwd. 1000ms')
        ts_k = [val+1000 for val in ts_k]
        actual_frame += ego_fps

    elif ch == ord('m'):
        print('back 100 frame on the Ego')
        actual_frame -= 100

    elif ch == ord('.'):
        print('fwd. 100 frame on the Ego')
        actual_frame += 100

    elif ch == ord('i'):
        actual_frame -= 1

    elif ch == ord('p'):
        actual_frame += 1

    print("Ego selected", ego_selected)
    k_index = [syncro.find_min_ts_diff_image(kinect_rgb_files[idx], ts_k[idx]) for idx in range(len(captures))]
    
if fo:
    fo.close()
