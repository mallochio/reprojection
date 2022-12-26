import PySimpleGUI as sg
import cv2
import threading
import queue
import functools
import argparse


@functools.lru_cache(maxsize=None)
def load_image(path):
    """Loads and returns an image object given its file path."""
    return cv2.imread(path)


def update_image(window, image):
    """Updates the image displayed in the GUI."""
    window["image"].update(data=cv2.imencode(".png", image)[1].tobytes())


# Create a thread to load the images in the background
def image_loader_thread(image_paths, image_queue):
    for path in image_paths:
        image = load_image(path)
        image_queue.put(image)


def main(image_list):
    # Read in list of image file paths from text file
    with open(image_list, "r") as f:
        image_paths = f.read().splitlines()

    # Create a PySimpleGUI window
    layout = [
        [sg.Image(filename="", key="image")],
        [sg.Button("Previous"), sg.Button("Next")],
    ]
    window = sg.Window(
        "Image Viewer",
        layout,
        # key="window"
    )

    # Initialize index to the first image in the list
    index = 0

    # Create a queue to store the images that have been loaded
    image_queue = queue.Queue()
    thread = threading.Thread(target=image_loader_thread, args=(image_paths, image_queue))
    thread.start()

    while True:
        # Read the window's event values and update the image displayed
        event, values = window.read(timeout=100)
        if event in (None, "Exit"):
            break
        elif event == "Next" or event == "window:down:40":
            index = (index + 1) % len(image_paths)
            try:
                # Try to get the next image from the queue
                image = image_queue.get(block=False)
            except queue.Empty:
                # If the queue is empty, use a placeholder image
                image = cv2.imread("placeholder.png")
            update_image(window, image)
        elif event == "Previous" or event == "window:down:38":
            index = (index - 1) % len(image_paths)
            try:
                # Try to get the previous image from the queue
                image = image_queue.get(block=False)
            except queue.Empty:
                # If the queue is empty, use a placeholder image
                image = cv2.imread("placeholder.png")
            update_image(window, image)

    window.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_list",
        type=str,
        required=True,
        help=f"Path to the text file containing the list of images to scroll through",
    )
    args = parser.parse_args()
    main(args.image_list)
