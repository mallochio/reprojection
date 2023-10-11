import sys
sys.path.append("..")
import customtkinter
from os import path, walk
from glob import glob 
from rich import print
from PIL import Image

# Own classes and Modules
import activity_tagger.utils.get_timestams as gts
from config import load_config as conf
from sync.utils import syncro
from Activities import Activities as A

# Read the config JSON.
config = conf.load_config()
work_dir = config['base_dir']

# Gets the first TimeStamp from the different cameras to read.
ts_cameras = gts.get_diff_from_shots(work_dir)
print(ts_cameras)

# Find Kinect captures (capture0, capture1, etc.).
folders = next(walk(work_dir))[1]
print(folders)
captures = sorted([path.join(work_dir, f) for f in folders if 'capture' in f])
print(f'Found {len(captures)} Kinect captures.')

# Kinect RGB images.
captures_len = []
kinect_rgb_files = [sorted(glob(f'{capture}/rgb/*.jpg')) for capture in captures]
for idx, k in enumerate(kinect_rgb_files):
    print(f'  Kinect {idx} captured {len(k)} RGB images')
    captures_len.append(len(k))

# Omni RGB images.
omni_rgb_files = sorted(glob(work_dir + '/omni/*.jpg'))

# Gets the first TimeStamp from the different cameras to read.
max_ts, name_camera = gts.sync_from_start(kinect_rgb_files, omni_rgb_files)
print("MAX_TS", max_ts, "name_camera", name_camera)
diff = None
cameras_struct = {}

for idx, camera in enumerate(ts_cameras):
    print(f"---- {str.upper(camera)} -----------")
    if camera != "omni":
        print(ts_cameras[camera]["timestamp"])
        k_idx = syncro.find_min_ts_diff_image(kinect_rgb_files[idx], ts_cameras[camera]["timestamp"])
        cameras_struct[camera] = kinect_rgb_files[idx][k_idx]
    else:
        print(ts_cameras[camera]["timestamp"])
        o_idx = syncro.find_min_ts_diff_image(omni_rgb_files, ts_cameras[camera]["timestamp"])
        cameras_struct[camera] = omni_rgb_files[o_idx]

for file in cameras_struct:
    filename = path.split(cameras_struct[file])[1]
    ts_camera = int(filename[:-4])
    cameras_struct[file] = ts_camera

diff = cameras_struct[name_camera] - max_ts
print("Diff ->", diff)
print(cameras_struct)

for camera in cameras_struct:
    cameras_struct[camera] = cameras_struct[camera] - diff

files = {}
cont = 0
activities_started = {}

def dict_to_string(dict: dict):
    text = str(dict)
    return text.replace('{', '{\n',1).replace(': {\'', ':\n\t\'{').replace(',', ',\n').replace(str('}}'), '}\n}')

def select_activity(choice): 
    global activity
    activity = choice

def configure_textbox():
    textbox.configure(state="normal", text_color="#07DF3C")
    textbox.delete("0.0", "end")  # delete all text
    textbox.insert("0.0", f"{dict_to_string(activities_started)}\n")  # insert at line 0 character 0
    textbox.configure(state="disabled")

def button_save_start():
    global activity, files, textbox
    if activity in activities_started:
        print("It is already in progress")
    else: 
        print("Start TimeStamp Just added")
        print(type(activity))
        activities_started[activity] = {}
        activities_started[activity]['Start'] = gts.get_timestamp_from_url(files["capture0"])
        configure_textbox()

def delete_activity():
    global activity
    if activity in activities_started:
        print("Activity removed")
        activities_started.pop(activity)
        configure_textbox()

def button_save_end():
    global activity, files
    if activity not in activities_started:
        print("The activity hasn't started yet")
    else: 
        print("End TimeStamp Just added")
        activities_started[activity]['End'] = gts.get_timestamp_from_url(files["capture0"])
        print(activities_started)
        # "/home/carlo/Escritorio/"
        with open(work_dir + '/tag.txt','a') as f:
            f.write(f"{activities_started[activity]['Start']};{activities_started[activity]['End']};{activity}\n")
        activities_started.pop(activity)
        configure_textbox()

def upload_images(event, color):
    try:
        if "capture0" in files:
            image_k0 = customtkinter.CTkImage(light_image=Image.open(files["capture0"]),
                                            size=(640, 360))
            label_k0.configure(image=image_k0)

        if "capture1" in files:
            image_k1 = customtkinter.CTkImage(light_image=Image.open(files["capture1"]),
                                            size=(640, 360))
            label_k1.configure(image=image_k1)

        if "capture2" in files:
            image_k2 = customtkinter.CTkImage(light_image=Image.open(files["capture2"]),
                                            size=(640, 360))
            label_k2.configure(image=image_k2)

        if "omni" in files:
            image_omni = customtkinter.CTkImage(light_image=Image.open(files["omni"]),
                                            size=(640, 360))
            label_omni.configure(image=image_omni, border_color=color)
    except OSError:
        print("Image file is truncated")



def clicker(event):
    global cont
    print(event)
    if event.char == 'j':
        print('back 1000ms')
        cont -= 1000

    elif event.char == 'l':
        print('fwd. 1000ms')
        cont += 1000
    
    elif event.char == 'm':
        print('back 5000ms')
        cont -= 5000

    elif event.char == '.':
        print('fwd. 5000ms')
        cont += 5000

    elif event.char == 'i':
        print('back 100ms')
        cont -= 100

    elif event.char == 'p':
        print('fwd. 100ms')
        cont += 100

    elif event.char == 'q':
        app.destroy()
        exit(0)

    for idx, camera in enumerate(cameras_struct):
        if camera == 'omni':
            o_idx = syncro.find_min_ts_diff_image(omni_rgb_files, cameras_struct[camera] + cont)
            files[camera] = omni_rgb_files[o_idx]
        else:
            k_idx = syncro.find_min_ts_diff_image(kinect_rgb_files[idx], cameras_struct[camera] + cont)
            files[camera] = kinect_rgb_files[idx][k_idx]

    # Shows the image frame in red if the differences between TimeStamp exceeds a threshold.
    # if o_idx[1] > 50:
    #     color = "RED"
    # else: 
    #     color = ""

    color = ""
    upload_images(event, color)
    # threading_upload_images = threading.Thread(target=upload_images, args=(event,color))
    # threading_upload_images.start()


activity = None
activities_started = {}
activities_array = [e.name for idx, e in enumerate(A)]
activities_array.sort()

customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('dark-blue')

app = customtkinter.CTk()
app.geometry("1620x780") # 1280 x 720
app.bind("<Key>", clicker)
app.title("Tagger")

for idx, camera in enumerate(cameras_struct):
    if camera == 'omni':
        o_idx = syncro.find_min_ts_diff_image(omni_rgb_files, cameras_struct[camera] + cont)
        files[camera] = omni_rgb_files[o_idx]
    else:
        k_idx = syncro.find_min_ts_diff_image(kinect_rgb_files[idx], cameras_struct[camera] + cont)
        files[camera] = kinect_rgb_files[idx][k_idx]

image_k0 = None
image_k1 = None
image_k2 = None
image_omni = None

if files.get("capture0") is not None:
    image_k0 = customtkinter.CTkImage(light_image=Image.open(files["capture0"]), size=(640, 360))

if files.get("capture1") is not None:
    image_k1 = customtkinter.CTkImage(light_image=Image.open(files["capture1"]), size=(640, 360))

if files.get("capture2") is not None:
    image_k2 = customtkinter.CTkImage(light_image=Image.open(files["capture2"]), size=(640, 360))

if files.get("omni") is not None:
    image_omni = customtkinter.CTkImage(light_image=Image.open(files["omni"]), size=(640, 360))

frame_right = customtkinter.CTkFrame(master=app, width=1280)
frame_right.grid(row=0, column=0, padx=1, pady=1, sticky="w")

# label_k0 = customtkinter.CTkLabel(master=frame_right, text="", image=image_k0)
label_k0 = customtkinter.CTkButton(master=frame_right, text="", image=image_k0, corner_radius=0, border_spacing=0, border_width=2, border_color="", fg_color="transparent")
label_k1 = customtkinter.CTkButton(master=frame_right, text="", image=image_k1, corner_radius=0, border_spacing=0, border_width=2, border_color="", fg_color="transparent")
label_k2 = customtkinter.CTkButton(master=frame_right, text="", image=image_k2, corner_radius=0, border_spacing=0, border_width=2, border_color="", fg_color="transparent")
label_omni = customtkinter.CTkButton(master=frame_right, text="", image=image_omni, corner_radius=0, border_spacing=0, border_width=2, border_color="", fg_color="transparent")

label_k0.grid(row=0, column=0, padx=1, pady=1, sticky="w")
label_k1.grid(row=0, column=1, padx=1, pady=1, sticky="w")
label_k2.grid(row=1, column=0, padx=1, pady=1, sticky="w")
label_omni.grid(row=1, column=1, padx=1, pady=1, sticky="w")

frame_left = customtkinter.CTkFrame(master=app, width=316)
frame_left.grid(row=0, column=1, sticky="nsw")

# Tab view
tabview = customtkinter.CTkTabview(master=frame_left, width=296, height=100)
tabview.grid(column=0, padx=10, pady=10, sticky="nsw")
tabview.add("Start")
tabview.add("End") 

# Start - Option menu
optionmenu_var = customtkinter.StringVar(value="None")
optionmenu = customtkinter.CTkOptionMenu(master=tabview.tab("Start"),values=activities_array,
                                        command=select_activity,
                                        variable=optionmenu_var,
                                        width=280)
optionmenu.grid(column=0, padx=0, pady=10, sticky="nsw")

# Start - Button remove
button = customtkinter.CTkButton(master=tabview.tab("Start"), text="Remove", command=delete_activity, width=280, fg_color="#EA5454", text_color="#000")
button.grid(column=0, padx=0, pady=10, sticky="nsw")

# Start - Button add
button = customtkinter.CTkButton(master=tabview.tab("Start"), text="Add", command=button_save_start, width=280, fg_color="#24B7D5", text_color="#000")
button.grid(column=0, padx=0, pady=10, sticky="nsw")

# End - Option menu
optionmenu_var = customtkinter.StringVar(value="None")
optionmenu = customtkinter.CTkOptionMenu(master=tabview.tab("End"),values=activities_array,
                                        command=select_activity,
                                        variable=optionmenu_var,
                                        width=280)
optionmenu.grid(column=0, padx=0, pady=10, sticky="nsw")

# End - Button save
button = customtkinter.CTkButton(master=tabview.tab("End"), text="Save", command=button_save_end, width=280, fg_color="GREEN")
button.grid(column=0, padx=0, pady=10, sticky="nsw")

# Text box
textbox = customtkinter.CTkTextbox(master=frame_left, width=296, height=400)
textbox.grid(column=0, padx=10, pady=10, sticky="nsw")
        
app.mainloop()