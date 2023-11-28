import cv2
import os
from tkinter import *
from PIL import Image, ImageTk

class_names = ["pronation", "supination"]

# Create directories if not exist
for class_name in class_names:
    if not os.path.exists(os.path.join("images", class_name)):
        os.makedirs(os.path.join("images", class_name))

imgtk = None

# prepare camera settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width of the frames in the video stream.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height of the frames in the video stream.
cap.set(cv2.CAP_PROP_FPS, 60)  # Frame rate.

# Function to capture and save image in the specified directory
def capture_and_save(class_dir):
    ret, frame = cap.read()
    if ret:
        img_name = os.path.join(class_dir, "{}.jpg".format(len(os.listdir(class_dir))))
        cv2.imwrite(img_name, frame)
        return frame

# Function to update display window
def update_image():
    global imgtk
    ret, frame = cap.read()
    if ret:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        image_canvas.create_image(0, 0, anchor=NW, image=imgtk)
    window.after(10, update_image)  # Call this function again after 10ms

# Create a window
window = Tk()

# Create a canvas for image display
image_canvas = Canvas(window, width=640, height=480)
image_canvas.pack()

# Create buttons
for i in range(len(class_names)):
    class_name = class_names[i]
    button = Button(window, text=class_name, command=lambda class_name=class_name: capture_and_save(os.path.join("images", class_name)))
    button.pack()

# Start the update function
update_image()

# Start the main loop
window.mainloop()

# Release the camera
cap.release()