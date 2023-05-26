"""
    Gets the TimeStamps.
"""
import re
from os import path
from rich import print

"""
Gets the TimeStamp of the URL filename.
"""
def get_timestamp_from_url(url):
    filename = path.split(url)[1]
    return int(filename[:-4])

"""
Gets the TimeStamps of the different cameras from the shots.txt file.
"""
def get_diff_from_shots(url: str):
    ts_cameras = {}
    try:
        f = open(path.join(url, 'shots.txt'))
        lines = f.readlines()
        f.close()
        line = lines[1]
        urls = re.split(';', line)
        
        for idx, e in enumerate(urls):
            filename = path.split(e)[1]
            print(filename)
            ts = re.split('.jpg',filename)
            if '/omni/' in e:
                print("OMNI")
                camera_name = e.split("/")[-2]
            elif '/capture' in e:
                camera_name = e.split("/")[-3]
            ts_cameras[camera_name] = {}
            ts_cameras[camera_name]['timestamp'] = int(ts[0])

    except FileNotFoundError:
        print("Error: shots.txt file does not exist in '%s'." % url)
        exit(-1)

    return ts_cameras

"""
Gets the first TimeStamp of the different synchronized cameras.
"""
def sync_from_start(k_files, o_files):
    __tmp_ts = {}
    max_ts = None
    name = None
    for file in k_files:
        camera_name = file[0].split("/")[-3]
        file = path.split(file[0])[1]
        f_ts = int(file[:-4])
        __tmp_ts[camera_name] = f_ts

    camera_name = o_files[0].split("/")[-2]
    file = path.split(o_files[0])[1]
    f_ts = int(file[:-4])
    __tmp_ts[camera_name] = f_ts

    for idx,ts in enumerate(__tmp_ts):
        if idx == 0 or __tmp_ts[ts] > max_ts:
            max_ts = __tmp_ts[ts]
            name = ts

    return (max_ts, name)
