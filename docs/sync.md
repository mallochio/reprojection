# Synchronisation

Carry out the synchronisation via the “sync_videos.py” script in the project, and once all sequences show the exact same timestamp, press the “space bar” once, to create a file registering the “offsets” between the “common” and each computers’ timestamps.

## Usage of the sync-videos.py script

The “ceilcam-demo” project contains a script named “sync-videos.py” that must be run to perform the syncing. It is necessary to have an “events.txt” file in each “captureX” folder, that describes the “noise” or “keypress” events captured during recording (this is based on an old idea, that we could sync the cameras by clapping our hands loudly using the cameras’ integrated microphones). However, the script expects these files to exist. To avoid errors, simply create an empty events.txt file in the “captureX” folder for the kinect that was started the earliest. And another with a timestamp and (e.g. the name of the first image, which will be a timestamp in milliseconds) followed by “;spacebar” for the Kinect that was started latest.

When running the script, the user will be presented with RGB images corresponding to each camera (Kinect 0, 1, etc. and omnidir camera). This way, using the following keys, it is possible to navigate through the video sequences until the user finds the frame that shows the exact same moment in all cameras (we usually throw an object from a shoulders distance to the floor, and capture the moment it touches the floor). The keys are:

- [0] .. [9] Select working Kinect.
- [Q] <> [E] Move one frame earlier/later on the selected Kinect.
- [Z] <> [C] Move one frame earlier/later on the Omni camera.
- [J] <> [L] Move backwards/forwards 100 ms on all sequences (this helps check the adjustments so far).
- [M] <> [ . ] Move one frame back/forwards on all sequences (also used to check if syncing is done correctly).
- [Spacebar] saves the “out/shots0.txt” file with the deviations of each camera in milliseconds. Furthermore, for each time it is pressed, it adds a line in which the name of the “equivalent” images is stored. This information is used by “prepare_for_matlab.py” to copy these images into a separate folder, with a naming convention to make it easy to load on MATLAB, as well as adapting their size to the common 1280x900 (used for calibration).

## Usage of the sync-extras.py script

The “ceilcam-demo” project contains a script named “sync-videos.py” that must be run to perform the syncing. It is necessary to have an “events.txt” file in each “captureX” folder, that describes the “noise” or “keypress” events captured during recording (this is based on an old idea, that we could sync the cameras by clapping our hands loudly using the cameras’ integrated microphones). However, the script expects these files to exist. To avoid errors, simply create an empty events.txt file in the “captureX” folder for the kinect that was started the earliest. And another with a timestamp and (e.g. the name of the first image, which will be a timestamp in milliseconds) followed by “;spacebar” for the Kinect that was started latest.

When running the script, the user will be presented with RGB images corresponding to each camera (Kinect 0, 1, etc. and omnidir camera). This way, using the following keys, it is possible to navigate through the video sequences until the user finds the frame that shows the exact same moment in all cameras (we usually throw an object from a shoulders distance to the floor, and capture the moment it touches the floor). The keys are:

- [0] .. [8] Select working Kinect.
- [9]        Select the Ego camera.
- [Q] <> [E] Move one frame earlier/later on the selected Kinect.
- [Z] <> [C] Move backwards/forwards 100 ms on all sequences (this helps check the adjustments so far).
- [J] <> [L] Move backwards/forwards 1000 ms on all sequences (this helps check the adjustments so far).
- [M] <> [ . ] Move backwards/forwards 100 ms on the Ego camera.
- [Spacebar] Save the file “out/shots_ego.txt” with the deviations in milliseconds of the kinect 0 and the corresponding frame of the Ego camera. In addition, the path to the corresponding image and the path of the Ego camera video file is stored.
- [ - ] Save the file “out/shots_wearable.txt”. Inside the file is the number of the selected kinect, for example, if kinect 0 has been saved, only the number 0 will appear. In addition, inside the file is the deviation in milliseconds of the selected camera and the path to the corresponding image. This time corresponds to the exact moment of the change from blue to red of the bracelet. Otherwise, the Ego camera may be selected instead of a kinect (by pressing the 9 key). In this case, the frame and path of the video will be saved.

## TO DO

* [x] Make the script work with more than 2 Kinects, e.g. by having keys [0], [1], [2] work as 'select Kinect', then [Q] and [E] work on the selected Kinect.