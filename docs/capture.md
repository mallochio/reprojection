# Capture

**Capture binaries:** These are meant for much quicker capture than Python scripts, and are therefore implemented in C++.

**Capture process:** Is described in this document, after binary compilation.

## Compiling ‘capture’ binaries

From a clean machine using Ubuntu 18.04 (or later, however, not tested), it is necessary to install the following packages:

```
apt-get install -y build-essential python3 python3-pip libportaudio2 	cmake pkg-config libusb-dev libusb-1.0-0 libusb-1.0-0-dev libturbojpeg0-dev libopencv-dev libglfw3-dev libopenni2-dev
```

Then, use Python’s PIP to install the following:

```
pip3 install sounddevice readchar tqdm
```

Once the prerequisites are installed, proceed to install libfreenect2 (download from official repo and build using cmake):

```
cd libfreenect2
mkdir build && cd build
cmake ..
make && make install
```

Finally, the 'capture' subfolder binaries, are also built via cmake:

```
cd capture
mkdir build && cd build
cmake ..
make
```

## Usage of the binaries

To run “Kinect-capture”:

```
./Kinect-capture 0 15
```

(i.e. Kinect #0, 15 fps)

To run “Omidir-capture”:

```
./Omnidir-capture 15
```

(It will search for **“omnidai.dtic.ua.es” in your network by default**, and uses 15 fps)

“Kinect-capture” **expects a directory** tree to be created in the folder where it is run from. It should be called “captureX” (where X is the number of the Kinect, i.e. zero). Inside it, there should be **three more directories: “ir”, “rgb”, and “depth”** (lowercase in all instances).

Furthermore, the PC in which the omnidirectional camera is also captured, there should be an **additional directory inside capture0, called “omni”**.

## Setting up the cameras and PCs

Two PCs (at least) are required. Each Kinect will be connected to one PC. The omnidirectional camera and at least one PC should be on the same network. By default the TP Link router that we bought will give the omni camera the following IP: 192.168.0.176 . **It is necessary to create an /etc/hosts entry so that omnidai.dtic.ua.es is redirected to the IP assigned to the omni camera, so that the camera may be found by the software just described.**

**IMPORTANT:** Check the omni camera is well focused, open a browser, go to the address above and use the credentials: admin and [password] to access the omni camera viewer on the browser.

## Capture process (Recording)

A “Kinect-capture” process is required for each machine, and an “Omnidir-capture” on either of them. Once recorded, stop the processes, and copy the “capture0” directory to a common directory in the machine where synchronisation is to be carried out. One of the captures will have to be renamed to “capture1” (**NOTE: consistency is key here!**). Copy “omni” outside, i.e. to the common  folder, so that three subfolders should remain: capture0, capture1 and omni. Each “capture0” and “1” contain “ir”, “rgb” and “depth” folders inside, as already described.

**NOTE:** Before starting the recording on several machines, and to facilitate synchornisation afterwards, check both/all machines are in the same time zone, and that their hardware clock is not deviating much. This will avoid having to rename files, and problems in fine-tuing the syncing. Also, please check the deviation between the local machine time and the Internet time (NTP) is less than 1 second.

### Run kinect-capture and omnidir-capture

As with calibration, a process is required per each PC+Kinect, and another “Omnidir-capture” in either of the machines. Once recorded, stop the processes, and copy the populated “capture0” from each machine into a **common folder** in the machine where syncing is carried out. Rename one to “capture1”. Copy “omni” outside, to the common folder, so that it contains: capture0, capture1, and omni. Each of “capture0” and “1” have “ir”, “rgb” and “depth” inside, as described.

### Note: In case of multiple recording sessions in the same day -
In case there are multiple recording sessions that take place in the same day, name the folder containing the files from each session with a suffix indicating the order of recording, starting with '01' for the first session. 

## Troubleshooting

Kinects connected to a new computer after just installing libfreenect2 **won't work** unless the proper USB *permissions* file is copied as per the documented instructions, that is:

Set up udev rules for device access:

```
sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
```

then **replug** the Kinect.

## Checklist for setting up the room for capture
- [ ] Study the room layouts before installation
- [ ] Figure out configuration of each room so that laptops and cameras can be placed without appearing on frames
- [ ] Figure out where persons are going to be standing before recording
- [ ] Check whether each item is in recording condition (and relaying info) after setting up -
    - [ ] Kinects
    - [ ] Omni camera
    - [ ] Egocentric camera
    - [ ] Empatica device


## Procedure for synchronization before beginning recording each ADL session
To enable the synchronization of kinects with each other, lights in the recording area are switched on and off. This quick action can then be verified in during the synchronization stage.

Actors doing the ADLs are to be equipped with an egocentric camera and an empatica wearable device. Each actor stands facing one of the kinects in the area and then switches on the empatica device so the change in lights on the empatica indicating that it is operational is registered on both the mounted egocentric camera and on the kinect. This procedure allows that all the recording devices can be synchronized with each other during the synchronization stage.

In the case of long recording sequences, this should also be done at regular intervals in order to check for visual drift between recording devices.

## Recording items checklist
- [ ] Kinect 1
    - [ ] Power cable
    - [ ] Connector cord
- [ ] Kinect 2
    - [ ] Power cable
    - [ ] Connector cord
- [ ] Kinect 3
    - [ ] Power cable
    - [ ] Connector cord
- [ ] Omnicam
    - [ ] Power cable
    - [ ] Magnet
    - [ ] Cable ties
    - [ ] 3 Bars for omni mounting
    - [ ] Power Bank
- [ ] Egocentric camera
    - [ ] Chest mount setup
    - [ ] Safety pins
    - [ ] Power Bank
- [ ] Empatica device and connector
- [ ] Router 
    - [ ] Power cable
    - [ ] Ethernet cable
- [ ] Double sided tape
- [ ] Kinect stand wooden horse 
- [ ] Checkerboard
- [ ] Power Strip / Extension cord
- [ ] Ethernet to USB converter
- [ ] Laptops 
- [ ] Folding stair
- [ ] Scissors
- [ ] USB drive
- [ ] External hard drive
- [ ] Human(s)
