import cv2
import numpy as np


def ORB_detector(new_image, image_template):
    # Function that compares input image to template
    # It then returns the number of ORB matches between them

    # image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image1 = new_image

    print(image1.shape)
    print(image_template.shape)

    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    # orb = cv2.ORB_create(1000, 1.2)
    orb = cv2.ORB_create()

    # Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(image1, None)

    # Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(image_template, None)

    # Create matcher
    # Note we're no longer using Flannbased matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Do matching
    matches = bf.match(des1,des2)

    # Sort the matches based on distance.  Least distance
    # is better
    matches = sorted(matches, key=lambda val: val.distance)

    return int(len(matches) *100 / len(kp2))

# cap = cv2.VideoCapture(0)


# Load our image template, this is our reference image
image_template = cv2.imread('base0.jpg', cv2.IMREAD_GRAYSCALE)
# image_template = cv2.imread('images/kitkat.jpg', 0)



# Get webcam images
# ret, frame = cap.read()
# frame = cv2.imread('0image.png')
frame = cv2.imread('images/test/20c0_re_20210223_132744(019)_No.18.jpg')

# Get height and width of webcam frame
height, width = frame.shape[:2]

# Define ROI Box Dimensions (Note some of these things should be outside the loop)
top_left_x = 350
top_left_y = 100
bottom_right_x = 460
bottom_right_y = 380

# Draw rectangular window for our region of interest
cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 1)

# Crop window of observation we defined above
cropped = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
# cv2.imshow('Object Detector using ORB', cropped)
# cv2.imwrite("base0.jpg", cropped)
# asd
# Flip frame orientation horizontally
# frame = cv2.flip(frame,1)




# Get number of ORB matches
matches = ORB_detector(cropped, image_template)
print(matches)

# Display status string showing the current no. of matches
output_string = "Matches = " + str(matches) + "%"
cv2.putText(frame, output_string, (50,450), cv2.FONT_HERSHEY_COMPLEX, 2, (250,0,150), 2)

# Our threshold to indicate object deteciton
# For new images or lightening conditions you may need to experiment a bit
# Note: The ORB detector to get the top 1000 matches, 350 is essentially a min 35% match
threshold = 100

# If matches exceed our threshold then object has been detected
if matches > threshold:
    cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
    cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)

cv2.imshow('Object Detector using ORB', frame)


cv2.waitKey(0)
cv2.destroyAllWindows()
