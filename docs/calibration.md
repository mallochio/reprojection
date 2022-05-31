# Calibration

During calibration, it is important to check what it is each Kinect sees before taking a “shot”, given that it is important to register the board on the floor as part of calibration. Both Kinects should be able to see the full board on the floor, or at least, the “internal” part of the board (i.e. the “crossings” of each checker board). This is easy to accomplish if using the “read_all_streams.py” Python script, since it shows what each camera sees “live”. In contrast, the “kinect-capture” executable does not show anything, it only records the images to disk “as fast as possible” (highest FPS possible).

## 1. Acquiring image sets

### Using read_all_streams.py

With this method, if the script is installed on all machines, it is possible to capture calibration images for the Kinect connected to that machine, and for the omnidirectional camera. It is not necessary to synchronise the streams later, since the board is left leaning on a surface (leaning on a wall, on a chair, or other object, ideally mounted on a tripod). That is, because when the user presses the “spacebar” key the object (board) has not been moving at all in the previous frame, even if the images are not perfectly synced (their timestamps may differ substantially) the board appears in the same position in all cameras.

**NOTE:** This method should give the best results, given that the board is not moved between consecutive frames or “shots” (i.e. the board is in the “exact” same position in all views). This is how the images capture in the lab were taken (least amount of error).

### Using kinect-capture

**NOTE:** Not recommended for calibration!

## 2. Preparing image sets for Matlab

Once the sequences have been **[synced](sync.md)**, it is possible to prepare image sets so that it is easier to load the from the Calibration app within Matlab. The script named “prepare_for_matlab.py” does exactly this. If the “shots0.txt” file generated in the step above (via “sync_videos.py”) contains one or more  “shots” (i.e. moments or “instants” in which the “space bar” was pressed), it will use this information to create pairs of images that are “equivalent” (i.e. showing the exact same rotation/position of the board), that are named in such way that it is easy to load in the Calibration app within Matlab.

**NOTE:** This “renaming” convention made it easy when we were using the “stereo” calibration in Matlab, but **this is not the case any more**. Therefore, we now load each set of images for the “mono” calibration of each camera (non-stereo).

## 3. Methods for calibation

### Method 1: Using only new images

The images taken with any of the programs described above can be used directly in the calibration app of Matlab.

### Method 2: Using a mixture of old and new images

Before, we were using Matlab’s stereo calibration app, but this is not the case anymore. We now calibrate each camera intrinsically and obtain the rotation and translation vectors for each camera with respect to the object (the board) on the floor. Therefore, we can use “pre-recorded” images for each camera along with **a single new image**. This is in contrast to having to record a new set of images every single time for each camera.

There are several “pre-recorded” image sets. These are to be found under “Calibration/imágenes pregrabadas Matlab” (i.e. “Calibration/pre-recorded images *for* Matlab”). The existing sets are either “casa_paco_mezcla” (mixture of images taken in the lab and Paco’s house), or “casa_paco_nuevas_solo” (all images are new, i.e. taken at Paco’s).

Therefore, with a “single” new image, and using the pre-recorded images, the Matlab app should be able to calibrate. The new image should be “adapted” to the size of the rest of pre-recorded images, given that the rest of the scripts work with images of 1280x900. There is a script named “kinect_ir_adapt.py” under “Calibration/Código/calibration_suite” that takes images in a directory and adapts them to the size required for successful calibration.

**NOTE:** It is important that the **first** image in the set given to Matlab be the one showing the board on the floor seen from all cameras. This is so because it will be the first position of the rotation and translation vectors that will be passed to the re-projection script.

## 4. Running MATLAB calibration

Once adapted, the calibration app within MATLAB should be able to give an output for calibrating a single camera. The parameters to give are: for each Kinect in “options” chose “3 coefficients” and “tangential”. Fot the omnidirectional camera chose the “fisheye” model instead of “standard”. In all three calibrations, chose to store the result in a variable in the working space (last button in the app menu bar). Give each variable a different name. From the Matlab console run:

```matlab
fh = fopen(‘your_filename.json’, ‘w’);
fprintf(fh, jsonencode(variable_name));
fclose(fh);
```

This is to be done for each camera being calibrated. Files should follow the naming convention: “k0Params.json”, “k1Params.json”, and “omniParams.json”. Additionally, they should be placed in the “common” recording (captures) folder.

**IMPORTANT:** To be able to check the calibration from MATLAB (see below, “checking the calibration”), it is necessary to store these variables also in the native .mat format of MATLAB.