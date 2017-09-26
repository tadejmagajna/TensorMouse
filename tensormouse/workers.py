import cv2
import argparse
import multiprocessing
import numpy as np
from multiprocessing import Value, Process
from tensormouse import object_detection

#Mouse state constants
MOUSE_NULL, MOUSE_CLICK, MOUSE_DRAG, MOUSE_DRAGGING, MOUSE_RELEASE, QUIT = range(6)


def key_listener():
    """ Ket listener thread """
    from pynput.keyboard import Listener
    return Listener(on_press=keypress_listener, on_release=keyrelease_listener)

def mouse_move_worker(objectX, objectY, mouse_state):
    """ Separate process worker to move mouse and perform clicks"""

    import tkinter
    from pynput.mouse import Button, Controller
    import time

    mouse = Controller()
    x,y = tkinter.Tk().winfo_screenwidth(), tkinter.Tk().winfo_screenheight()

    while True:
        if mouse_state.value == MOUSE_CLICK:
            time.sleep(0.2)
            mouse.press(Button.left)
            mouse_state.value = MOUSE_RELEASE
        if mouse_state.value == MOUSE_DRAG:
            time.sleep(0.2)
            mouse.press(Button.left)
            mouse_state.value = MOUSE_DRAGGING
        if (mouse_state.value == MOUSE_RELEASE):
            mouse.release(Button.left)
            mouse_state.value = MOUSE_NULL

        if (objectX.value > 0 and objectY.value > 0):
            mouse.position = (int((1-objectX.value)*x), int(objectY.value*y))


def keypress_listener(key):
    """ Key press listener """

    from pynput.keyboard import Key
    global mouse_state

    if key == Key.ctrl:
        mouse_state.value = MOUSE_CLICK
    elif key == Key.alt:
        mouse_state.value = MOUSE_DRAG
    elif key == Key.caps_lock:
        mouse_state.value = QUIT

def keyrelease_listener(key):
    """ Key release listener """
    global mouse_state
    if mouse_state.value == MOUSE_DRAGGING:
        mouse_state.value = MOUSE_RELEASE

def main_worker(object_id, camera_id, object_label):
    """ Main worker that connects webcam with tensorflow and mouse_move_worker """

    global mouse_state

    # init process safe variables for workers
    objectX, objectY = Value('d', 0.0), Value('d', 0.0)
    mouse_state = Value('i', MOUSE_NULL)
    mouse_state = mouse_state

    # init mouse worker
    mouse_process = Process(target=mouse_move_worker, args=(objectX, objectY, mouse_state))
    mouse_process.start()

    # init keyboard listener
    keyboard_thread = key_listener()
    keyboard_thread.start()

    # init webcam
    webcam = cv2.VideoCapture(camera_id)
    webcam.set(3,480)
    webcam.set(4,360)

    # get tensofrlow object detection graph
    detection_graph, sess = object_detection.graph_and_sess()

    # nofify used of successful startup
    print(u'\x1b[6;30;42m \u2713 TensorMouse started successfully! \033[0m')
    print('Tracking object: ' + object_label)
    print("Use CTRL to perform clicks, ALT to cursor drag and press CAPS_LOCK to exit")

    while(True):
        _, frame = webcam.read()

        frame_expanded = np.expand_dims(frame, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Actual detection.
        (boxes, scores, classes) = sess.run([boxes, scores, classes], feed_dict={image_tensor: frame_expanded})

        # indices of all instances of object of interest
        objects = np.where(classes[0] == object_id)[0]

        # calculate center of box if detection exceeds threshold
        if len(objects) > 0 and scores[0][objects][0] > 0.15:
            b = boxes[0][objects[0]]
            objectX.value, objectY.value = (b[1]+b[3])/2, (b[0]+b[2])/2
        else:
            objectX.value, objectY.value = 0.0, 0.0

        if mouse_state.value == QUIT:
            break

    mouse_process.terminate()
    keyboard_thread.stop()

    webcam.release()
    cv2.destroyAllWindows()
