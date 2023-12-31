'''Align Images tester
Usage is python align_images.py --template bluecourtreference.png --image CourtPhotos/court.jpg
'''
from image_registration import ImageRegistration
import numpy as np

import imutils
import cv2

# align the images
print("[INFO] aligning images...")

IR=ImageRegistration()
aligned, template=IR.align_images('images/court.jpg', 'bluecourtreference.png', debug=True) 

# resize both the aligned and template images so we can easily
# visualize them on our screen
if aligned is not None:
    aligned = imutils.resize(aligned, width=700)
    template = imutils.resize(template, width=700)
    # our first output visualization of the image alignment will be a side-by-side comparison of the output aligned image and the template
    stacked = np.hstack([aligned, template])
    # our second image alignment visualization will be *overlaying* the aligned image on the template, that way we can obtain an idea of how good our image alignment is
    overlay = template.copy()
    output = aligned.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    # show the two output image alignment visualizations
    cv2.imshow("aligned", aligned)
    cv2.imshow("Image Alignment Stacked", stacked)
    cv2.imshow("Image Alignment Overlay", output)
    cv2.waitKey(0)
else:
    print("Error: Aligned image is None")
    

