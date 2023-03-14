import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('funktionelle-villa.jpg', 0)
    # img = cv.imread('funkisbungalowen.jpg', 0)

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

    cv.imshow('fast_true', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()