from glob import glob
import cv2
import re
from utils import syncro
from utils import load_config as conf
import os

show_ratio=0.4

ms_frame = (1000./30.)  # 2*Hz of recording.

# work_dir = '/media/pau/Data/Pictures/data_24may/'
config = conf.load_config()
work_dir = config['base_dir']

# Kinect 0 RGB images
kinect0_rgb_files = sorted(glob(work_dir + '/capture0/rgb/*.jpg'))
print('Kinect 0 captured %d RGB images' % len(kinect0_rgb_files))

# Kinect 1 RGB images
kinect1_rgb_files = sorted(glob(work_dir + '/capture1/rgb/*.jpg'))
print('Kinect 1 captured %d RGB images' % len(kinect1_rgb_files))

# Omnicam RGB images
omni_rgb_files = sorted(glob(work_dir + '/omni/*.jpg'))
print('Omnicam captured %d RGB images' % len(omni_rgb_files))

# Find initial and final timestamps for omnicam:
omni_start = int(os.path.split(omni_rgb_files[0])[1].replace('.jpg', ''))
omni_end = int(os.path.split(omni_rgb_files[-1])[1].replace('.jpg', ''))
kinect0_start = int(os.path.split(kinect0_rgb_files[0])[1].replace('.jpg', ''))
kinect0_end = int(os.path.split(kinect0_rgb_files[-1])[1].replace('.jpg', ''))
kinect1_start = int(os.path.split(kinect1_rgb_files[0])[1].replace('.jpg', ''))
kinect1_end = int(os.path.split(kinect1_rgb_files[-1])[1].replace('.jpg', ''))

# Get the first timestamp which has images for all cameras.
timestamp = omni_start

# Difference between omni and kinects (ok_diff)
#ok_diff = 800
ok_diff=0
# Timestamps for each device
ts_o = timestamp + ok_diff
ts_k0 = timestamp
ts_k1 = timestamp

k0_index = syncro.find_min_ts_diff_image(kinect0_rgb_files, ts_k0)
k1_index = syncro.find_min_ts_diff_image(kinect1_rgb_files, ts_k1)
o_index = syncro.find_min_ts_diff_image(omni_rgb_files, ts_o)

space_pressed = False
fo = None

while True:

    k0_selected = kinect0_rgb_files[k0_index]
    k0_img = cv2.imread(k0_selected)
    cv2.imshow('kinect0', cv2.resize(k0_img,None,fx=show_ratio,fy=show_ratio))

    if k1_index >= 0:
        k1_selected = kinect1_rgb_files[k1_index]
        k1_img = cv2.imread(k1_selected)
        cv2.imshow('kinect1', cv2.resize(k1_img,None,fx=show_ratio,fy=show_ratio))

    o_selected = omni_rgb_files[o_index]
    o_img = cv2.imread(o_selected)
    cv2.imshow('omnicam', cv2.resize(o_img,None,fx=show_ratio,fy=show_ratio))

    ch = cv2.waitKey(0)
    if ch & 0xFF == 27:
        break
    if ch == ord('a'):
        ts_k1 -= ms_frame
    elif ch == ord('d'):
        ts_k1 += ms_frame
    elif ch == ord('q'):
        ts_k0 -= ms_frame
    elif ch == ord('e'):
        ts_k0 += ms_frame
    elif ch == ord('z'):
        ts_o -= ms_frame
    elif ch == ord('c'):
        ts_o += ms_frame
    elif ch == ord(' '):
        if not space_pressed:
            fo = open(work_dir + '/shots.txt', 'w')
            fo.write('%d;%.3f;%.3f;%.3f\n' % (ok_diff, ts_k0/1000., ts_k1/1000., ts_o/1000.))
            space_pressed = True
        if fo:
            k0_ir_selected = k0_selected.replace('/rgb/', '/ir/')
            if k1_index >= 0:
                k1_ir_selected = k1_selected.replace('/rgb/', '/ir/')
            else:
                k1_ir_selected = ""
            fo.write('%s;%s;%s;\n' % (k0_ir_selected, k1_ir_selected, o_selected))
            print(k0_ir_selected, k1_ir_selected, o_selected)
    elif ch == ord('j'):
        print('back 100ms')
        ts_k0 -= 100
        ts_k1 -= 100
        ts_o -= 100
    elif ch == ord('l'):
        print('fwd. 100ms')
        ts_k0 += 100
        ts_k1 += 100
        ts_o += 100
    elif ch == ord('m'):
        print('back 1 frame')
        ts_o -= ms_frame
        ts_k0 -= ms_frame
        ts_k1 -= ms_frame
    elif ch == ord('.'):
        print('fwd. 1 frame')
        ts_o += ms_frame
        ts_k0 += ms_frame
        ts_k1 += ms_frame

    k0_index = syncro.find_min_ts_diff_image(kinect0_rgb_files, ts_k0)
    k1_index = syncro.find_min_ts_diff_image(kinect1_rgb_files, ts_k1)
    o_index = syncro.find_min_ts_diff_image(omni_rgb_files, ts_o)

if fo:
    fo.close()
