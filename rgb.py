from loadData_mavic3m import get_rgb_data
import imageProcessingUtils as ip
import os
import numpy as np
import cv2
def undistort_bounding_boxes(labels, fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height):
    """
    Applies undistortion to bounding box coordinates using camera calibration parameters.

    Parameters:
    labels (list of lists): Bounding boxes in the format [class_id, x_center, y_center, width, height].
    fx (float): Focal length along the x-axis.
    fy (float): Focal length along the y-axis.
    cx (float): Principal point along the x-axis.
    cy (float): Principal point along the y-axis.
    k1, k2, p1, p2, k3 (float): Distortion coefficients.
    width (int): Width of the image.
    height (int): Height of the image.

    Returns:
    list: Undistorted bounding boxes in the format [class_id, x_center, y_center, width, height].
    """
    undistort_boxes = []

    # Camera matrix
    cameraMatrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float32)

    # Distortion coefficients
    distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    for box in labels:
        class_id = box[0]
        x_center = box[1] * width
        y_center = box[2] * height
        box_width = box[3] * width
        box_height = box[4] * height

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2

        # Distorted points
        distorted_points = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]], dtype=np.float32)
        distorted_points = np.expand_dims(distorted_points, axis=1)

        # Apply cv2.undistortPoints
        undistorted_points = cv2.undistortPoints(distorted_points, cameraMatrix, distCoeffs, P=cameraMatrix)
        undistorted_points = np.squeeze(undistorted_points)

        # Calculate the undistorted bounding box coordinates
        x_min = max(0, min(undistorted_points[:, 0]))
        y_min = max(0, min(undistorted_points[:, 1]))
        x_max = min(width, max(undistorted_points[:, 0]))
        y_max = min(height, max(undistorted_points[:, 1]))

        # Convert back to YOLO format
        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        undistort_boxes.append([class_id, x_center, y_center, box_width, box_height])

    return undistort_boxes

def get_undistort_rgb(path, rgb_name, labels, show=False, save=False, savePath=None):

    if save:
        if savePath is None:
            savePath = os.path.join(path, 'output', 'rgb')

    rgb_data = get_rgb_data(os.path.join(path, rgb_name))

    rgb_img = rgb_data['img']

    if show:
        ip.show_img('Original RGB', rgb_img)

    print()
    print('Applying distortion correction RGB')
    rgb_img, rgb_roi = ip.distortion_correction(rgb_img, rgb_data)
    if save:
        ip.save_img(os.path.join(savePath, 'distortionCorrection'),
            rgb_name[:-3] + 'jpg', rgb_img)
    if show:
        ip.show_img('Distortion correction RGB', rgb_img)
    
    undistort_labels=undistort_bounding_boxes(labels,rgb_data['dewarp']['fx'],rgb_data['dewarp']['fy'],
                            (rgb_data['centerX'] + rgb_data['dewarp']['cx']),(rgb_data['centerY'] + rgb_data['dewarp']['cy']),
                            rgb_data['dewarp']['dist'][0],rgb_data['dewarp']['dist'][1],rgb_data['dewarp']['dist'][2],
                            rgb_data['dewarp']['dist'][3],rgb_data['dewarp']['dist'][4],rgb_img.shape[1],rgb_img.shape[0]
                            )
    return rgb_img, undistort_labels

if __name__ == "__main__":
    path = os.path.join('.', 'data', 'dati_grezzi_drone', 'Test', 'OneStack')
    rgb_name = 'DJI_20240525115637_0026_D.JPG'
    savepath = os.path.join('.', 'data', 'output', 'Test')
    rgb = get_undistort_rgb(path, rgb_name, False, True, savepath)