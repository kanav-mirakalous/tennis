'''File to align images
Written by Kanav Kahol @2023
'''
import numpy as np 
import cv2
import imutils
import matplotlib.pyplot as plt

class ImageRegistration:
    '''Class to align images'''
    def __init__(self):
        pass

    def align_images(self, img:str, templ:str, maxFeatures=500, keepPercent=0.2, debug=False):
        '''function to align images
        image: Our input photo/scan of the tennis court. This is the image that we want to align with the template.
        template: The template tennis image. The features should be present in the image. This is a limitation where we must have all features or many of them visible and from some angles we may not have all of them visible.
        maxFeatures: Places an upper bound on the number of candidate keypoint regions to consider.
        keepPercent: Designates the percentage of keypoint matches to keep, effectively allowing us to eliminate noisy keypoint matching results
        debug: A flag indicating whether to display the matched keypoints. By default, keypoints are not displayed; however I recommend setting this value to True for debugging purposes.
        '''
        if not isinstance(img, str):
            print('img is not a string')
        elif not isinstance(templ, str):
            print('templ is not a string')
        else:
            image = cv2.imread(img)
            template = cv2.imread(templ)

        if image is None:
            print('Could not open or find the image')
        elif template is None:
            print('Could not open or find the template')
        else:
            cv2.imshow("Image", image)
            cv2.imshow("Template", template)
            cv2.waitKey(0)
        if image is not None or template is not None:
            imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(maxFeatures) #opensource version of SIFT
            (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
            (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
            # match the features
            method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
            matcher = cv2.DescriptorMatcher_create(method)
            matches = matcher.match(descsA, descsB, None)
            # sort the matches by their distance (the smaller the distance,
            # the "more similar" the features are)
            matches = sorted(matches, key=lambda x:x.distance)
            # keep only the top matches
            keep = int(len(matches) * keepPercent)
            matches = matches[:keep]
            # check to see if we should visualize the matched keypoints
            if debug:
                matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                    matches, None)
                matchedVis = imutils.resize(matchedVis, width=1500)
                cv2.imshow("Matched Keypoints", matchedVis)
                cv2.waitKey(0)
            # allocate memory for the keypoints (x, y)-coordinates from the
            # top matches -- we'll use these coordinates to compute our
            # homography matrix
            ptsA = np.zeros((len(matches), 2), dtype="float")
            ptsB = np.zeros((len(matches), 2), dtype="float")
            # loop over the top matches
            for (i, m) in enumerate(matches):
                # indicate that the two keypoints in the respective images
                # map to each other
                ptsA[i] = kpsA[m.queryIdx].pt
                ptsB[i] = kpsB[m.trainIdx].pt
            # compute the homography matrix between the two sets of matched
            # points
            (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
            # use the homography matrix to align the images
            (h, w) = template.shape[:2]
            print(f" the h is {H} and the mask is {mask}")
            aligned = cv2.warpPerspective(image, H, (w, h),  cv2.INTER_LINEAR)
            # return the aligned image
            return aligned, template
        else:
            print("Error: Image or template is None")
            return None
        





