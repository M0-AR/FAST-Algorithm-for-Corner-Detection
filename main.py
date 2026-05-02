import cv2 as cv
import os

def detect_and_save(image_path, output_path):
    """
    Detects corners using the FAST algorithm and saves the resulting image.
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # Load image in grayscale
    img = cv.imread(image_path, 0)
    if img is None:
        print(f"Error: Could not read {image_path}.")
        return

    # Initiate FAST object with default values
    # FAST: Features from Accelerated Segment Test
    fast = cv.FastFeatureDetector_create()

    # Find and draw the keypoints
    kp = fast.detect(img, None)
    img_with_keypoints = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    # Print detection parameters and results
    print(f"\n--- Detection Results: {image_path} ---")
    print(f"Threshold: {fast.getThreshold()}")
    print(f"Non-maximum Suppression: {fast.getNonmaxSuppression()}")
    print(f"Neighborhood Type: {fast.getType()}")
    print(f"Total Keypoints Detected: {len(kp)}")

    # Save the result
    cv.imwrite(output_path, img_with_keypoints)
    print(f"Result saved to: {output_path}")

if __name__ == '__main__':
    # Define images to process
    samples = [
        ('funktionelle-villa.jpg', 'result_villa.jpg'),
        ('funkisbungalowen.jpg', 'result_bungalow.jpg')
    ]

    for input_img, output_img in samples:
        detect_and_save(input_img, output_img)
