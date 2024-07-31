import os
import cv2
import math
import numpy as np
from tqdm import tqdm


def show_img(name, img):
    temp = cv2.resize(img, (960, 540))
    cv2.imshow(name, temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_img(fpath, img_name, img):
    os.makedirs(fpath, exist_ok=True)
    img_path = os.path.join(fpath, img_name)
    cv2.imwrite(str(img_path), img)


def vignetting_correction(img, data):
    height, width, depth = img.shape

    for x in tqdm(range(height)):

        for y in range(width):
            r = math.sqrt((x - data['centerX']) ** 2 + (y - data['centerY']) ** 2)
            for k in range(3):
                value = img[x, y][k] * (
                        (data['vignetting'][5] * (r ** 6)) + (data['vignetting'][4] * (r ** 5)) + (
                        data['vignetting'][3] * (r ** 4)) + (data['vignetting'][2] * (r ** 3)) + (
                                data['vignetting'][1] * (r ** 2)) + (
                                data['vignetting'][0] * (r ** 1)) + 1)
                if value > 255:
                    value = 255

                img[x, y][k] = value

    return img


def distortion_correction(img, data):
    height, width, depth = img.shape
    camera_matrix = [(data['dewarp']['fx'], 0, (data['centerX'] + data['dewarp']['cx'])),
                     (0, data['dewarp']['fy'], (data['centerY'] + data['dewarp']['cy'])),
                     (0, 0, 1)]
    camera_matrix = np.asmatrix(np.array([np.array(xi) for xi in camera_matrix]))
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, data['dewarp']['dist'], (width, height), 0.9,
                                                      (width, height))

    dst = cv2.undistort(img, camera_matrix, data['dewarp']['dist'])
    #dst = cv2.undistort(img, camera_matrix, data['dewarp']['dist'], None, newcameramtx)
    return dst, roi


def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def alignmentECC(im1, im2):
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape
    # Smoothing (Exposure time alignment optimization)
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed_im1 = cv2.filter2D(im1_gray, -1, kernel)

    smoothed_im2 = cv2.filter2D(im2_gray, -1, kernel)

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    # warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                50, 1e-4)
    # get_gradient(im1_gray), get_gradient(im1_gray)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(smoothed_im1), get_gradient(smoothed_im2), warp_matrix,
                                             warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR
                                                                             + cv2.WARP_INVERSE_MAP
                                     )

    return im2_aligned, warp_matrix


def crop_img_yolo_format(img, centerx, centery, w, h,offset=70):
    shape = img.shape

    centerX = centerx * shape[1]
    centerY = centery * shape[0]
    W = w * shape[1]
    H = h * shape[0]

    # lenght of pixel to start
    x = centerX - W / 2
    y = centerY - H / 2

    crop_img = img[int(y-offset):int(y + H+offset), int(x-offset):int(x + W+offset)].copy()

    return crop_img


def overlap_images(img1, img2):

        # Define a blending factor (adjust according to your preference)
        alpha = 0.5

        # Blend the False Color NDVI image with the RGB image
        blended_image = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)

        return blended_image




# Function to overlay text on image
def overlay_text(img, label_id):
    # Define health status labels
    health_status_labels = {0: 'Asymptomatic', 1: 'Mild symptoms', 2: 'Evident symptomatic/compromised'}
    label = health_status_labels[label_id]

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    img = cv2.rectangle(img, (10, 10), (10 + w, 10+h), (0, 255, 0), -1)
    img = cv2.putText(img, label, (10, 10 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img