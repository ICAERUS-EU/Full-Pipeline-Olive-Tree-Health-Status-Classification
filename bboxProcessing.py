

def yolo2cornersbbox(yolo_box, img_size):
    # Warning: yolo bbox[0] is the object type
    xcenter = yolo_box[1] * img_size[0]
    ycenter = yolo_box[2] * img_size[1]
    box_width = yolo_box[3] * img_size[0]
    bbox_height = yolo_box[4] * img_size[1]
    x1 = int(xcenter - box_width/2)
    y1 = int(ycenter - bbox_height / 2)
    x2 = int(xcenter + box_width/2)
    y2 = int(ycenter + bbox_height/2)

    return [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]


def corners2yolobbox(corners_box, img_size):
    norm_corners = []
    for point in corners_box:
        norm_corners.append((point[0]/img_size[0], point[1]/img_size[1]))

    box_width = norm_corners[1][0] - norm_corners[0][0]
    box_height = norm_corners[2][1] - norm_corners[0][1]
    centerx = norm_corners[0][0] + box_width/2
    centery = norm_corners[0][1] + box_height/2

    return [centerx, centery, box_width, box_height]


def iscornersbboxinndvi(corners_bbox, ndvi_box):
    print(ndvi_box)
    print(corners_bbox)
    from shapely.geometry import Polygon
    polya = Polygon(ndvi_box)
    polyb = Polygon(corners_bbox)
    return polya.contains(polyb)


def getbboxinsidendvi(yolobbox_list, ndvi_box, img_size):
    valid_bbox_list = []
    for yolobbox in yolobbox_list:
        yolobbox = [float(i) for i in yolobbox]
        c_box = yolo2cornersbbox(yolobbox, img_size)
        if iscornersbboxinndvi(c_box, ndvi_box):
            valid_bbox_list.append(yolobbox)

    return valid_bbox_list


if __name__ == "__main__":
    import os
    import json
    import cv2
    from imageProcessingUtils import show_img
    import numpy as np

    pathyoloboxes = os.path.join('.', 'DJI_20240525115637_0026_BBOXES.txt')
    pathndvibox = os.path.join('.', 'data', 'Test', 'ndvi', 'DJI_20240525115637_0026_NDVI_BOX.txt')
    ndvi = cv2.imread(os.path.join('.', 'data', 'Test', 'ndvi', 'DJI_20240525115637_0026_WARPED_NDVI.JPG'))
    rgb = cv2.imread(os.path.join('.', 'data/Test/distortionCorrection/DJI_20240525115637_0026__RGB.JPG'))

    with open(pathyoloboxes) as file:
        bbox_list = [line.rstrip().split(' ') for line in file]

    with open(pathndvibox) as file:
        ndvi_bbox = json.load(file)

    # Show all bboxes on rgb (undistorted)
    print(len(bbox_list))
    img_size = [5280, 3956]
    polylines = []
    for bbox in bbox_list:
        f_bbox = [float(i) for i in bbox]
        c_box = yolo2cornersbbox(f_bbox, img_size)
        print(c_box)
        polylines.append(c_box)
    polylines = np.array(polylines, np.int32)
    rgb = cv2.polylines(rgb, polylines, True, (0, 255, 0), 10)
    show_img('rgb', rgb)


    valid_bboxes = getbboxinsidendvi(bbox_list, ndvi_bbox, img_size)

    # Show valid bboxes on NDVI
    print(len(valid_bboxes))
    polylines = []
    for bbox in valid_bboxes:
        f_bbox = [float(i) for i in bbox]
        c_box = yolo2cornersbbox(f_bbox, img_size)
        print(c_box)
        polylines.append(c_box)
    polylines = np.array(polylines, np.int32)
    rgb = cv2.polylines(ndvi, polylines, True, (0, 255, 0), 10)
    show_img('ndvi', ndvi)










