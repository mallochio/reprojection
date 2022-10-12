import datetime
from re import I
from select import select
import yaml
import os
from subprocess import Popen


def select_item(collection, name):
    for i, item in enumerate(collection)    :
        print(f'{i}: {item}')

    index = input(f'Please choose {name} [0-{len(collection)-1}]:')
    index = int(index)
    label = collection[index]
    return index, label

def main():
    
    fh = open('kinects.yaml', 'r')
    config = yaml.load(fh, yaml.FullLoader)

    print('Welcome to the capture launch wizard ...')

    h_index, house = select_item(config['houses'], 'location')
    print(f" Location '{house}'")

    r_index, room = select_item(config['rooms'], 'room')
    print(f" Room '{room}'")

    a_index, actor = select_item(config['actors'], 'actor')
    print(f" Actor '{actor}'")

    k_index, serial = select_item(config['kinects'], 'kinect')
    kinect = f"capture{k_index}"
    print(f"  Kinect {k_index}: s/n {config['kinects'][k_index]}")

    date = datetime.date.today().isoformat()
    route = f"{date}/{house}/{room}/{actor}"
    
    if not os.path.exists(route):
        os.makedirs(route)
    else:
        rep = 2
        original = route
        while True:
            route = f"{original}{rep}"
            if not os.path.exists(route):
                os.makedirs(route)
                break
            rep += 1

    omni_path = f"{route}/omni"
    if k_index == 0:
        print('This computer will also record the omni')
        if not os.path.exists(omni_path):
            os.makedirs(omni_path)

    images = ['ir', 'rgb', 'depth']
    kinect_path = f"{route}/{kinect}"
    for img in images:
        sf = f"{kinect_path}/{img}"
        if not os.path.exists(sf):
            os.makedirs(sf)

    fps = config['default_fps']
    
    kinect_cmd = ['./Kinect-capture', f'{k_index}', kinect_path, f'{fps}']
    commands = [kinect_cmd]
    
    if k_index == 0:
        omni_cmd = ['./Omnidir-capture', omni_path, f'{fps}']
        commands.append(omni_cmd)

    procs = [ Popen(i) for i in commands ]
    for p in procs:
        p.wait()


if __name__ == '__main__':
    main()