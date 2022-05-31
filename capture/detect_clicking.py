#!/usr/bin/env python3

import sounddevice as sd
import sys
import time as stime
from readchar import readchar

_downsample = 10

def find_kinect():
    device_list = sd.query_devices()
    for index, device in enumerate(device_list):
        name = device['name']
        if 'Xbox' in name or 'NUI' in name:
            sys.stderr.write('%s ...\n' % name)
            return index, device['default_samplerate'] #, device['channels']
    print('Error: No kinect found')
    exit(-1)

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # print((indata[::_downsample, 0] >= 0.02).sum())
    if (indata[::_downsample, 0] >= 0.03).sum() > 0:
        posix = stime.time()
        print('%.9f;sound' % posix)  # TODO: Save this to a file


def main():
    device_index, sample_rate = find_kinect()

    stream = sd.InputStream(
        device=device_index, channels=4,
        samplerate=sample_rate, callback=audio_callback)

    with stream:
        while True:
            stime.sleep(.2)
            c = readchar()
            if c == 'q':
                exit(0)
            elif c == ' ':
                posix = stime.time()
                print('%.9f;spacebar' % posix)



if __name__ == '__main__':
    main()
