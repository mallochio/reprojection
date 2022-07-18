"""
    Functions to help with loading synchronisation data from a "shots.txt" file.
"""
from os import path
import re
import numpy as np
import json

__timestamps_cache = {}


def find_min_ts_diff_image(fileset, timestamp):
    min_diff = 1e9
    min_diff_index = -1
    if len(fileset) == 0:
        return -1;
    if not fileset[0] in __timestamps_cache:
        __timestamps_cache[fileset[0]] = np.zeros((len(fileset)))

        for file_idx, filepath in enumerate(fileset):
            filename = path.split(filepath)[1]
            filename_ts = int(filename[:-4])
            __timestamps_cache[fileset[0]][file_idx] = filename_ts

    differences = np.abs(__timestamps_cache[fileset[0]] - timestamp)
    return np.argmin(differences)


def get_timestamp_differences(dir_path):
    syncro_file = path.join(dir_path, 'syncro_data.json')
    if not path.exists(syncro_file):
        try:
            f = open(path.join(dir_path, 'shots.txt'))
            lines = f.readlines()
            f.close()

            line = lines[0]
            chunks = re.split(';', line)

            timestamps_long = np.array([np.int64(ts.replace('.', '')) for ts in chunks[1:]], dtype=np.int64)
            timestamps = np.array([float(ts) / 1000. for ts in timestamps_long])
            differences = timestamps - timestamps[len(timestamps)-1] # Last timestamp is always Omni.

            line = lines[1][:-2]
            chunks = re.split(';', line)

            dico = {}
            for c, chunk in enumerate(chunks):
                file_path = path.normpath(chunk)
                if not '/omni/' in file_path:
                    capture_name = re.split('/', chunk)[-3]
                else:
                    capture_name = 'omni'
                dico[capture_name] = {}
                dico[capture_name]['timestamp'] = timestamps[c]
                dico[capture_name]['diff_to_lead'] = differences[c]

            fh = open(syncro_file, 'w')
            json.dump(dico, fh)
            fh.close()

        except FileNotFoundError:
            print("Error: shots.txt file does not exist in '%s'." % dir_path)
            exit(-1)
    else:
        try:
            fh = open(syncro_file, 'r')
            dico = json.load(fh)
            fh.close()
        except json.decoder.JSONDecodeError:
            print("Error: Malformed sync data file (invalid JSON).")
            exit(-1)

    return dico


# Self-test main function
if __name__ == '__main__':
    t = get_timestamp_differences('/home/pau/Pictures/data_6mar_pm')
    print(t)
