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


def image_loader_thread(image_paths, image_queue):
    """ Loads images in the background and puts them in a queue."""
    for path in image_paths:
        image = load_image(path)
        image_queue.put(image)


def main(image_list):
    """
    Creates a GUI window that displays images from a list of image file paths, and also allows the user to scroll through the images using the arrow keys.
    """
    # Read in list of image file paths from text file
    with open(image_list, "r") as f:
        image_paths = f.read().splitlines()

    # # Create a PySimpleGUI window
    layout = [
        [sg.Image(filename="", key="image")],
        [sg.Button("Previous"), sg.Button("Next")],
    ]

    # # Create a PySimpleGUI window
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
        elif event in ["Next", "Down:40"]:
            index = (index + 1) % len(image_paths)
            try:
                # Try to get the next image from the queue
                image = image_queue.get(block=False)
            except queue.Empty:
                # If the queue is empty, use a placeholder image
                image = cv2.imread("placeholder.png")
            update_image(window, image)
        elif event in ["Previous", "Down:38"]:
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
        "--image_list", "-i",
        type=str,
        required=True,
        help="Path to the text file containing the list of images to scroll through",
    )
    args = parser.parse_args()
    main(args.image_list)
