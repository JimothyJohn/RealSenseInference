# First import the RealSense library
from pyrealsense2 import pyrealsense2 as prs2

# Import Numpy for easy array manipulation
import numpy as np
import math


class RealSense:

    ## Initializes Jetson networks and classes
    def __init__(self, width, height, fps):

        # User inputs
        self.width = width
        self.height = height
        self.fps = fps

        # reset status flags
        self.frames = None
        self.running = False
        self.aligned = False
        self.color_np = None

        # Create a pipeline
        self.pipeline = prs2.pipeline()

        # Create a config and configure the pipeline to stream
        # different resolutions of color and depth streams
        self.config = prs2.config()
        self.config.enable_stream(
            prs2.stream.depth, self.width, self.height, prs2.format.z16, self.fps
        )
        self.config.enable_stream(
            prs2.stream.color, self.width, self.height, prs2.format.rgba8, self.fps
        )

        # Create an align object
        # self.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self.align = prs2.align(prs2.stream.color)

    ## Begins RealSense Pipeline and gets depth scale, returns True if successful
    def Start(self):
        try:
            self.profile = self.pipeline.start(self.config)  # Start streaming
            self.running = True  # Set run flag to true

            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
        except:
            self.running = False
            print("Unable to start camera!")

        return self.running

    ## Takes an image and outputs color array
    def Capture(self):
        if not self.running:
            self.Start()

        try:
            ## Extract 2D image array
            self.frames = (
                self.pipeline.wait_for_frames()
            )  # get frameset of color and depth
            color_frame = (
                self.frames.get_color_frame()
            )  # pull color frame from pipeline
            self.color_np = np.asanyarray(
                color_frame.get_data()
            )  # convert frame to array

            ## Reset inference since we have a new image
            self.aligned = False
        except:
            print("Unable to acquire image!")
            self.color_np = None

        return self.color_np

    ## Aligns depth to color array and outputs calibrated depth
    def Align(self):
        if self.frames is None:
            self.Capture()  # grab an image if we don't have information already

        # Align the depth frame to color frame
        aligned_frames = self.align.process(self.frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not aligned_depth_frame:
            print("Error in frame")
            return

        # Extract aligned arrays
        self.depth_np = np.asanyarray(aligned_depth_frame.get_data())
        self.color_np = np.asanyarray(aligned_color_frame.get_data())
        self.aligned = True

        return self.depth_np

    ## Stops pipeline dead in its tracks and outputs success
    def Stop(self):
        try:
            self.pipeline.stop()
            self.running = False
            return True
        except:
            print("Could not stop camera!")
            return False

    ## Finds real-world location of pixel value
    def Locate(self, x_pixel, y_pixel):

        if not self.aligned:
            self.Align()  # align camera if not done already

        ## Bail if no coordinates given
        if x_pixel + y_pixel == 0:
            print("No coordinates provided!")
            return 0, 0, 0  # skip drawing if nothing detected

        PI = 3.14157  # If you don't know what this is I don't care
        WIDTH_FOV_ANGLE = 87 / 2 * (PI / 180)  # Intel hardware spec +/-3
        HEIGHT_FOV_ANGLE = 58 / 2 * (PI / 180)  # Intel hardware spec +/-2

        h = self.depth_np[y_pixel][x_pixel]  # stay with mm

        ## Obtain angles of center pixels from sensor datum
        if x_pixel < (self.width / 2):
            rx = -(self.width / 2 - x_pixel) / (self.width / 2) * WIDTH_FOV_ANGLE
        else:
            rx = ((x_pixel - self.width / 2) / (self.width / 2)) * WIDTH_FOV_ANGLE

        if y_pixel < (self.height / 2):
            ry = (self.height / 2 - y_pixel) / (self.height / 2) * HEIGHT_FOV_ANGLE
        else:
            ry = -((y_pixel - self.height / 2) / (self.height / 2)) * HEIGHT_FOV_ANGLE

        # Calculate offset using distance and angle
        x = int(h * math.sin(rx))
        y = int(h * math.sin(ry))
        z = int(h * math.cos(ry))

        return x, y, z


"""
# Import Jetson tools
import jetson.inference as ji
import jetson.utils as ju


class RealDetector:

    ## Initializes Jetson networks and classes
    def __init__(self, network, threshold=0.4):

        ## Jetson config
        self.network = ji.detectNet(
            network, threshold
        )  # load the object detection network

        ## Create detection class dictionaries
        self.detect_dict = {
            clss: self.network.GetClassDesc(clss)
            for clss in range(self.network.GetNumClasses())
        }
        self.reverse_dict = {v: k for k, v in self.detect_dict.items()}

    ## Runs detection and outputs detection object if found
    ## Takes in an RGBA np array
    def Detect(self, color_np):
        (
            self.height,
            self.width,
            self.channels,
        ) = color_np.shape  # extract dimensions from input array
        cuda_image = ju.cudaFromNumpy(
            color_np
        )  # create cuda memory capsule from np array
        detectnet_output = self.network.Detect(
            cuda_image, self.width, self.height
        )  # detect objects in the image

        if len(detectnet_output) == 0:
            return None  # if no detections are found output nothing

        return detectnet_output



import ctypes
# Import scipy to calculate center of mass
from scipy.ndimage.measurements import center_of_mass
import os
import re

class RealSegmenter:

    ## Initializes Jetson networks and classes
    def __init__(self, network):

        ## Jetson config
        self.network = ji.segNet(network)  # load the object detection network
        self.ignore_class = "void"  # find all classes
        self.filter_mode = (
            "point"  # ensures segmentation masks are exact, not fuzzy (linear)
        )

        ### Create segmentation class dictionaries
        # Declare paths to find network information
        class_path = "/usr/local/bin/networks"  # network directory
        class_file = "classes.txt"  # class filename
        color_file = "coloself.txt"  # color filename
        ## create strings out of files
        classes = [
            clss.strip()
            for clss in open(os.path.join(class_path, network, class_file))
        ]  # get list of class strings
        colors = []  # initiate color list
        ## open and format colors then create list
        for color in open(os.path.join(class_path, network, color_file)):
            px = (
                color.strip().split()
            )  # strip outside whitespace and split by space delimiters
            coloself.append(tuple(map(int, px)))  # add pixel values to list

        # Write dictionaries
        self.segment_dict = {color: clss for clss, color in zip(classes, colors)}
        self.reverse_dict = {v: k for k, v in self.segment_dict.items()}

    ## Outputs segment RGBA image
    ## Takes in RGBA np array
    def Segment(self, color_np):
        (
            self.height,
            self.width,
            self.channels,
        ) = color_np.shape  # get size of image from input
        # Allocate CUDA memory
        self.cuda_out = ju.cudaAllocMapped(
            self.width * self.height * 4 * ctypes.sizeof(ctypes.c_float)
        )
        cuda_image = ju.cudaFromNumpy(color_np)  # convert to CUDA memory capsule

        # Process the segmentation network
        self.network.Process(cuda_image, self.width, self.height, self.ignore_class)
        # output segmentation mask
        self.network.Mask(self.cuda_out, self.width, self.height, self.filter_mode)
        # create np array from CUDA memory capsule
        seg_np_out = ju.cudaToNumpy(self.cuda_out, self.width, self.height, 4)

        return seg_np_out

    ## Extracts average depth over segmentation area
    ## Takes in RealSense class and np segmentation output
    def Depth(self, rs, seg_np_out):
        if not self.aligned:
            self.Align()  # align camera if not done already

        ## Find mean/median of target areas
        net_mask = np.all(
            seg_np_out.astype("uint8") != self.segment_dict[self.seg_class], axis=-1
        )  # retrieve class location
        depth_array = np.ma.array(
            self.depth_np, mask=net_mask
        )  # create masked depth array
        self.depth_value = float(
            depth_array.mean() * self.depth_scale
        )  # get values inside mask and then scale

        return self.depth_value

    ## Runs detection and outputs a list of detected classes
    ## Takes in np segmentation output
    def Classify(self, seg_np_out):
        found_colors = np.unique(
            seg_np_out.reshape(-1, seg_np_out.shape[2]), axis=0
        ).astype(
            int
        )  # find unique pixel values in image

        # create tuple of RGB values found in segmentation and create list of corresponding classes
        detections = [self.segment_dict[tuple(color[:3])] for color in found_colors]

        return detections

    ## Finds XY location of first detected object's center
    ## Requires np segmentation map and a target class to find
    def Locate(self, seg_np, target_class=None):
        ## Bail if no class given
        if target_class == None:
            print("No class provided!")
            return 0, 0

        ## Retrieve class location
        # where RGB channel of segmentation map matches a color label of a class (converted to single channel)
        class_px = np.where(
            seg_np.astype("uint8")[:, :, :3]
            == np.asarray(self.reverse_dict[target_class]),
            1,
            0,
        )[:, :, 1]
        com = center_of_mass(class_px)  # get center of segment shape
        ## IMPORTANT NOTE - Crescent/edge shapes will output a location not over a pixel value

        ## Bail if class not found in image
        if math.isnan(com[0]) or math.isnan(com[1]):
            y_coord, x_coord = 0, 0  # output nothing
            print(target_class, "not found!")
            return 0, 0

        y_coord, x_coord = int(com[0]), int(
            com[1]
        )  # convert to integer coordinates for legibilty

        return x_coord, y_coord

"""
# Import OpenCV for easy image rendering
import cv2

## Draw crosshairs on specified image at Xp, Yp, and provide real-world location
def drawCenter(
    color_np, x_pixel, y_pixel, x, y, z, color=(0, 255, 0), target_class=None
):
    ## Bail if no coordinates given
    if x_pixel + y_pixel + x + y + z == 0 or target_class == None:
        print("Skipping", target_class)
        return color_np  # return original drawing if nothing detected

    ## Annotate image for all classes requested
    cv2.circle(color_np, (x_pixel, y_pixel), 4, color, -1)

    return


def drawText(color_np, x_pixel, y_pixel, x, y, z, color=(0, 255, 0), target_class=None):
    text_np = color_np.copy()
    ## Bail if no coordinates given
    if x_pixel + y_pixel == 0 or target_class == None:
        print("Skipping", target_class)
        return text_np  # return original drawing if nothing detected

    ## Annotatation settings
    font = cv2.FONT_HERSHEY_SIMPLEX  # choose font

    # Add all text to image
    cv2.putText(
        text_np,
        "{} found".format(target_class),
        (x_pixel + 5, y_pixel + 20),
        font,
        0.75,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        text_np,
        "X: {}mm".format(x),
        (x_pixel + 5, y_pixel + 45),
        font,
        0.75,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        text_np,
        "Y: {}mm".format(y),
        (x_pixel + 5, y_pixel + 70),
        font,
        0.75,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        text_np,
        "Z: {}mm".format(z),
        (x_pixel + 5, y_pixel + 95),
        font,
        0.75,
        color,
        1,
        cv2.LINE_AA,
    )

    return text_np


## Draw crosshairs on specified image at Xp, Yp, and provide real-world location
def drawBox(color_np, left, right, top, bottom, color=(0, 255, 0), target_class=None):
    ## Bail if no coordinates given
    if left + right + top + bottom == 0 or target_class == None:
        print("Skipping", target_class)
        return color_np  # return original drawing if nothing detected        out_np = color_np.copy() # create copy to annotate

    ## Annotatation settings
    thickness = 3  # thiccness setting

    # Annotate image
    cv2.rectangle(
        color_np, (left, top), (right, bottom), color, thickness
    )  # annotate image

    return
