To use synchronisation tool on Linux. 
	1/ Install Wine ver 7.19 developer version with Ubuntu Software
	2/ Download the whole folder "Executable" that you can find in the repository
	3/ Right click on Synchronisation_toolV2.exe and click on "Open with Wine Windows Program Loader"

How to use the software: 
	1/ When launching the software select "browse file". 
	2/ the path to choose should be: Date of the dataset/Room/Name
	3/ The Folders inside this path should be : 
			-omni
			-kinect0
			-kinect1
			-kinect2
			-egovision
The folders should be write with the exact same spelling otherwise the software will not detect the folders, one or several folder can be missing, the program can deal with this configuration. 
	3/Click on previous and next button for each video flux to spot the moment where the light is blinking 
	4/ when the synchronisation is right click on "Save Synchronisation "
	5/ Now you can start using the annotation tool, you can start an action by clicking on "Start action"
	6/ After the synchronisation you can use the button below annotation tool to skip frames all together
	7/ when the action is finished you can click on "End action" and select the action with the combo box where it is written "select an action name"
	8/ After that click on "save current action" to write a new line in the "ActionTimeStamps.txt" file.
	The file is created in the path you've selected before, as long as a file called "SynchronisationTimeStamps", in this file you have all the Time stamps of the synchronisation. 
	9/One line of the "SynchronisationTimeStamps" look like this:
	Omnidirectionnal camera,1664459315978,29,9,2022,15:48:35.9780000
	It is: name of the camera,Time Stamp of the synchonisation in microsecond, Day, Month,year, hour in local time of the machine. 
	If one camera file is misssing the line will look like this: Chest camera,0,1,1,1970,01:00:00. The date is the basic unix epoch

	10/ One line of the "ActionTimeStamp.txt" looks like this:
	kitchen,sid,Reading,1664459335862,1664459324693,1664459327431,1664459324910,0,1664459402151,1664459366904,1664459368833,1664459369637,0
	It is: room,name of the people,action, Timestamp Start for Omni,Timestamp Start for Kinect 1,Timestamp Start for kinect 2,Timestamp Start for Kinect 3,Timestamp Start for ego camera, Timestap End for Omni,Timestap End for kinect 1,Timestap End for kinect 2,Timestap End for Omni kinect 3,Timestap End for ego camera.
	
	The zeros in the line mean that there is no images for the ego camera so no timestamp to show. 


