# import the necessary packages
import argparse
import imutils
import time
import cv2
import os
import numpy as np

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
    
    # Convert focal length, f, from [mm] to [pixel]
    # TODO: WITH NEW CAMERA CALIBRATION, DIRECTLY INPUT PIXEL VALUES
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # Calculate the disparity
    disparity = x_left-x_right #Displacement between left and right frames [pixels]

    # Calculate depth, z
    zDepth = (baseline*f_pixel)/disparity #Depth in [cm]

    return zDepth

#load cone classifier
print("[INFO] loading haar cascades...")
cone_classifier = cv2.CascadeClassifier("cascade9.xml")#CHANGE THIS

# initialize the video stream
print("[INFO] starting image...")

########## Initialize HSV Parameters ###########

lower_yellow = np.array([10, 100, 100])
upper_yellow = np.array([30, 255, 255])

lower_blue = np.array([60, 80, 40])
upper_blue = np.array([150, 255, 255])

# need two separate orange filters due to the hue passing over 180
lower_orange = np.array([170, 120, 40])
upper_orange_mid = np.array([180, 255, 255])
lower_orange_mid = np.array([0, 120, 40])
upper_orange = np.array([15, 255, 255])

########## Function to process the frame and convert it to HSV ###########

# Don't think we need this, test speed without this
def claheHSV(image):
    cone_image_bgr = image
    cone_image_rgb = cv2.cvtColor(cone_image_bgr, cv2.COLOR_BGR2RGB)

    lab_image = cv2.cvtColor(cone_image_rgb, cv2.COLOR_RGB2LAB)

    lab_planes = list(cv2.split(lab_image))
    lab_clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 8))

    lab_planes[0] = lab_clahe.apply(lab_planes[0])
    lab_image = cv2.merge(lab_planes)
    re_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
	
    lab_result = cv2.cvtColor(re_rgb, cv2.COLOR_RGB2HSV)
    return lab_result

########## Function for colour labelling ###########

# TODO: make a faster colour labelling algorithm
# https://data-flair.training/blogs/project-in-python-colour-detection/
def labelColour(image, HSV_image, x, y, w, h):
    blue_count = 0
    yellow_count = 0
    orange_count = 0
    for i in range (x, x + w + 1):
        for j in range (y - h, y + 1):
            test_pixel = HSV_image[j, i]

            hue, sat, val = test_pixel[0], test_pixel[1], test_pixel[2]

            #update blue_count, yellow_count, orange_count
            #improvement: do we really need to go through the whole bounding box? we could just go through a portion of the box near the centre which would be indicative of cone colour

            if hue >= lower_blue[0] and hue <= upper_blue[0] and sat >= lower_blue[1] and sat <= upper_blue[1] and val >= lower_blue[2] and val <= upper_blue[2]:
                blue_count = blue_count + 1
            elif hue >= lower_yellow[0] and hue <= upper_yellow[0] and sat >= lower_yellow[1] and sat <= upper_yellow[1] and val >= lower_yellow[2] and val <= upper_yellow[2]:
                yellow_count = yellow_count + 1
            elif ((hue >= lower_orange[0] and hue <= upper_orange_mid[0]) or (hue >= lower_orange_mid[0] and hue <= upper_orange[0])) and sat >= lower_orange[1] and sat <= upper_orange[1] and val >= lower_orange[2] and val <= upper_orange[2]:
                orange_count = orange_count + 1

            #display result
            if blue_count > yellow_count and blue_count > orange_count:
                cv2.putText(image, text="blue", org=(x, y - h - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            elif yellow_count > blue_count and yellow_count > orange_count:
                cv2.putText(image, text="yellow", org=(x, y - h - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            elif orange_count > blue_count and orange_count > yellow_count:
                cv2.putText(image, text="orange", org=(x, y - h - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                cv2.putText(image, text="none", org=(x, y - h - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

########## Initialize Cameras ###########

# NOTE: This section will have to be updated depending on your hardware!
cap_right = cv2.VideoCapture(0)                    
cap_left =  cv2.VideoCapture(2)

# Stereo vision setup parameters
# Need to update!
frame_rate = 30    #Camera frame rate [fps]
B = 25.4           #Distance between the cameras [cm]
f = 8              #Camera lense's focal length [mm]
alpha = 55         #Camera field of view in the horisontal plane [degrees]

# Main program loop with detector and depth estimation using stereo vision
while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    # If cannot catch any frame, break
    if not succes_right or not succes_left:   
        print("Cannot catch any frame")                 
        break

    else:
        start = time.time()
        
        # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Process the image and make detections using classifier
        results_right = cone_classifier.detectMultiScale(frame_right, 1.1, 4) #face_detection.process(frame_right)
        results_left = cone_classifier.detectMultiScale(frame_left, 1.1, 4)  #face_detection.process(frame_left)

        # Convert the RGB image to BGR
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
	
	# Generate HSV images for both frames
    HSV_right = claheHSV(frame_right)
    HSV_left = claheHSV(frame_left)

    ################## Calculating Depth ##################

    center_right = 0
    center_left = 0

    center_point_left = None
    center_point_right = None

    # find cones and label colour in right camera
    for (x, y, w, h) in results_right:
        cv2.rectangle(frame_right, (x, y), (x+w, y+h), (255, 0, 0), 2)
        center_point_right = (x, y)
        labelColour(frame_right, HSV_right, x, y, w, h)

    # find cones and label colour in left camera
    for (x, y, w, h) in results_left:
        cv2.rectangle(frame_left, (x, y), (x+w, y+h), (255, 0, 0), 2)
        center_point_left = (x, y)
        labelColour(frame_left, HSV_left, x, y, w, h)

    # If no detections can be caught in one camera we display "TRACKING LOST"
    if center_point_left == None or center_point_right == None:
        cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    else:
        # Function to calculate depth of object. Outputs vector of all depths in case of several detections.
        # TODO: Make a presentation documenting formulas used. This will later be linked in the README of this repo.
        depth = find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

        cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        
        # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
        print("Depth: ", str(round(depth,1)))

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   

    # Show the frames
    cv2.imshow("frame right", frame_right) 
    cv2.imshow("frame left", frame_left)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()