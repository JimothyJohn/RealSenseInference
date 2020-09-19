# Import Numpy for easy array manipulation
import numpy as np
# Import KDTree for 3D coordinate comparison
from scipy.spatial import KDTree
import cv2
import time
import threading
from flask import Response, Flask
# Import Jetson tools
import jetson.inference as ji
import jetson.utils as ju

# First import the library
import pyrealsense2 as prs2
# Import standard tools
import sys
import argparse
import ctypes
import math

# Import custom class for 3D AI detection
import rsinfer as rsi

# Input: findViolators takes a list of 3D locations [[x1,y1,z1],[x2,y2,z2],...]
# Function: It then checks to see if any are too close (<2000)
# Output: It returns a list of each index where they're too close
def findViolators(xyz_locations, social_distance):
    
    if len(xyz_locations) < 2: return None # skip if only one detection present
    violator_list = [] # initialize list
    dist_tree = KDTree(xyz_locations) # create KD Tree
    for location in xyz_locations:
        # Extract distances under 2m and their corresponding indexes
        distances, idxs = dist_tree.query(location,k=len(xyz_locations),distance_upper_bound=social_distance)

        # For all distances except for its own
        for distance, idx in zip(distances[1:], idxs[1:]):
            # If close to any other locations
            if idx < len(xyz_locations) and idx not in violator_list:
                violator_list.append(idx) # add to list of violators
                break # move to next detection location):
    
    return violator_list

width, height, fps = 640, 480, 30 # "low-res"
#width, height, fps = 1280, 720, 6 # "hi-res"
channels = 4

## Configured JetsonDetect input
# Load the networks
detect_network = 'pednet'

rs = rsi.RealSense(width, height, fps)
detector = rsi.RealDetector(detect_network)
if rs.Start(): print('Camera started!')

target_classes = ['person', 'shirt', 'face']
    
# Image frame sent to the Flask object
global video_frame
video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

# Create the Flask object for the application
app = Flask(__name__)

def captureFrames():
    global video_frame, thread_lock

    while True:
        # Get new frame
        color_np = rs.Capture()
        detections = detector.Detect(color_np)
        bb_locations = [] # bounding box coordinates
        xyz_locations = [] # 3D coordinates
        # Find requested class detections
        if detections is not None: # make sure there's actually a detection
            for detection in detections: # for every found object
                if detector.network.GetClassDesc(detection.ClassID) == target_classes[0]: # if it is the target
                    if detector.network.GetClassDesc(detection.ClassID) == target_classes[0]: # if it is the target
                        # Add BB corners to list
                        bb_locations.append([int(detection.Left),
                                            int(detection.Right),
                                            int(detection.Top),
                                            int(detection.Bottom)])
                        ## Extract XYZ positions
                        x, y, z = rsi.Locate3D(rs, int(detection.Center[0]),int(detection.Center[1]))
                        xyz_locations.append([x, y, z]) # add XYZ locations to list

                # Get list of violators
                violator_list = findViolators(xyz_locations, 2000)

                # for all detection locations
                for idx in range(len(xyz_locations)):
                    color = (0,255,0) # greed

                    # if violating color appropriately
                    if violator_list is not None and idx in violator_list:
                        color = (255,0,0) # red

                    # draw bounding box around person
                    cv2.rectangle(color_np, (bb_locations[idx][0], bb_locations[idx][2]),
                                    (bb_locations[idx][1], bb_locations[idx][3]), color, 3) 
                    
        
        bgr_image = cv2.cvtColor(color_np, cv2.COLOR_RGBA2BGR)
        
        # Create a copy of the frame and store it in the global variable,
        # with thread safe access
        with thread_lock:
            video_frame = bgr_image.copy()
        
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
        
def encodeFrame():
    global thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock:
            global video_frame
            if video_frame is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame)
            if not return_key:
                continue

        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')

@app.route("/")
def streamFrames():
    return Response(encodeFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':

    # Create a thread and attach the method that captures the image frames, to it
    process_thread = threading.Thread(target=captureFrames)
    process_thread.daemon = True

    # Start the thread
    process_thread.start()

    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network 
    app.run("0.0.0.0", port="9000")