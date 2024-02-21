import os, sys, PIL
from PIL import ImageOps, Image


def remove_broken_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if any([d.startswith("out_capture") for d in dirs]):
            # Don't search this root folder at all
            dirs.clear()
            continue

        if os.path.basename(root) == "rgb" and "calib" not in root:
            print(root)
            for file in files:
                if file.endswith("jpg"):
                    try:
                        img = Image.open(os.path.join(root, file)).convert("RGB")
                        # print(os.path.join(root, file))
                        img.verify()
                        img = ImageOps.mirror(img)
                    except Exception as e:
                        # print the exception and remove the file
                        print(f"Error: {e}")
                        print(f"Removing {os.path.join(root, file)}")
                        os.remove(os.path.join(root, file))
                        # # Also remove correspondingly named file from ir and depth folders
                        # check_folder_1 = os.path.join(os.path.dirname(root), "ir")
                        # check_folder_2 = os.path.join(os.path.dirname(root), "depth")
                        # # replace jpg with png for ir and depth folders
                        # file = file.replace("jpg", "png")
                        # if os.path.exists(os.path.join(check_folder_1, file)):
                        #     print(f"Removing {os.path.join(check_folder_1, file)}")
                        #     # os.remove(os.path.join(check_folder_1, file))

                        # if os.path.exists(os.path.join(check_folder_2, file)):
                        #     print(f"Removing {os.path.join(check_folder_1, file)}")
                        #     # os.remove(os.path.join(check_folder_2, file))


def main():
    remove_broken_files(sys.argv[1])


if __name__ == "__main__":
    main()

