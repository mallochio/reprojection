# coding: utf-8

# An example using startStreams

import numpy as np
import cv2
import sys
import time
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
#from smbh.py.video.capture import VideoCaptureAsync

base_path = '/home/pau/Pictures/Kinect/'

# Remove comment if images are to be resized.
# r, c = 900, 1280
# desired_size = (r, c, 3)

cap = cv2.VideoCapture("rtsp://192.168.0.176:554/live1.sdp")


try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

enable_rgb = True
enable_depth = True

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

if enable_rgb and enable_depth:
    device.start()
else:
    device.startStreams(rgb=enable_rgb, depth=enable_depth)

# NOTE: must be called after device.start()
if enable_depth:
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)


# cap.start()
key_pressed = False
frame_no = 0
while True:
    frames = listener.waitForNewFrame()

    timestamp = str(int(round(time.time() * 1000)))

    if enable_rgb:
        color = frames["color"]
    if enable_depth:
        ir = frames["ir"]
        depth = frames["depth"]

    if enable_rgb and enable_depth:
        registration.apply(color, depth, undistorted, registered)
    elif enable_depth:
        registration.undistortDepth(depth, undistorted)


    # Async. read omnicam frame
    ret, bgr_omni = cap.read()
    cv2.imshow('omni', bgr_omni)
    if key_pressed:
        br, bc = bgr_omni.shape[:2]
        # bigger = np.zeros(desired_size)
        # bigger[int(r / 2 - br / 2):int(r / 2 + br / 2), int(c / 2 - bc / 2):int(c / 2 + bc / 2)] = bgr_omni
        # cv2.imwrite("%s/omni/%s.jpg" % (base_path, timestamp), bigger)
        cv2.imwrite("%s/omni/%s.jpg" % (base_path, timestamp), bgr_omni)

    if enable_depth:
        ir_norm = ir.asarray() / 65535.
        ir_255 = ir_norm * 255.
        cv2.imshow("ir", ir_norm)
        # cv2.imshow("depth", depth_norm)
        # cv2.imshow("undistorted", undistorted.asarray(np.float32) / 4500.)
        if key_pressed:
            br, bc = ir_norm.shape[:2]
            # bigger = np.zeros(desired_size)
            # bigger[int(r / 2 - br / 2):int(r / 2 + br / 2), int(c / 2 - bc / 2):int(c / 2 + bc / 2)] = \
            #    cv2.flip(cv2.cvtColor(ir_255, cv2.COLOR_GRAY2BGR), 1)
            # cv2.imwrite("%s/ir/%s.jpg" % (base_path, timestamp), bigger)
            cv2.imwrite("%s/ir/%s.jpg" % (base_path, timestamp), ir_255)

            depth_array = np.uint16(depth.asarray())
            cv2.imwrite("%s/depth/%s.png" % (base_path, timestamp), cv2.flip(depth_array, 1))
    if enable_rgb:
        cv2.imshow("rgb", cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3))))
        if key_pressed:
            cv2.imwrite("%s/rgb/%s.jpg" % (base_path, timestamp), cv2.flip(color.asarray(), 1))
    # if enable_rgb and enable_depth:
        # cv2.imshow("registered", registered.asarray(np.uint8))

    listener.release(frames)

    if key_pressed:
        key_pressed = False

    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    if key == ord(' '):
        key_pressed = True

    frame_no += 1

cap.stop()
device.stop()
device.close()

sys.exit(0)