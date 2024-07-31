
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import json
import os
from classifier import DualYOLO

from bboxProcessing import getbboxinsidendvi
from loadData_mavic3m import load_images_from_folder,get_rgb_data,list_subfolders
from ndvi import get_NDVI
from rgb import get_undistort_rgb
from imageProcessingUtils import crop_img_yolo_format,overlay_text,save_img
 
'''
Set order:
    0: RGB
    1: GREEN
    2: NIR
    3: RED
    4: RED EDGE
'''


def main(imagesPath):

    img_size = [5280, 3956]
    ms_size = [2592, 1994]

    detect_model_path = "./models/object_detection/weights/best.pt"
    classification_model_path = "./models/classification_health_status/weights/epoch=28-val_loss=0.63.ckpt"

    subfolders = list_subfolders(imagesPath)
    subfolders.append(imagesPath)

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model for detection
    detect_model = YOLO(detect_model_path)

    # Load model for classification
    classification_model = DualYOLO(classification_model_path)
    classification_model = classification_model.to(device)
    
    classification_model.eval()
    output_predictions = './predictions'
    for folder in subfolders:
        image_sets = load_images_from_folder(folder)

        for img_set in image_sets:
            print('Analizing image set: ', img_set)
            
            rgb_name = img_set[0]
            nir_name = img_set[2]
            red_name = img_set[3]
            
            rgb_data = get_rgb_data(os.path.join(folder, rgb_name))
            
            rgb_img = rgb_data['img']
            
            rgb_img_resize=cv2.resize(rgb_img, (960, 640), interpolation=cv2.INTER_AREA)
            
            # NDVI calculation
            # if os.path.exists('./ndvi/'+rgb_name[:-5]+f'NDVI.JPG'):
            #     ndvi=cv2.imread('./ndvi/'+rgb_name[:-5]+f'NDVI.JPG')
            #     ndvi_area=''
            #     with open('./ndvi/'+rgb_name[:-5]+f'NDVI.txt','r') as file:
            #         ndvi_area=json.load(file)
            # else:
            #     with open('./ndvi/'+rgb_name[:-5]+f'NDVI.txt','w') as file:
            #         json.dump(ndvi_area,file)
            #         save_img('./ndvi', rgb_name[:-5]+f'NDVI.JPG', ndvi)
            
            ndvi, ndvi_area = get_NDVI(folder, nir_name, red_name, False, True)
            # save_img('./ndvi', rgb_name[:-5]+f'NDVI.JPG', ndvi)

            # detection olive tree
            boxes = []
            detection_res = detect_model.predict(rgb_img_resize,conf=0.25,iou=0.3,imgsz=960,device=device)

            for res in detection_res:
                for box in res.boxes:
                    xywhn = box.xywhn.tolist()[0]
                    temp = [0]
                    temp.extend(xywhn)
                    
                    boxes.append(temp)

            # Undistort RGB to the same plane of NDVI for allignment and undistort boxes
            rgb_img, boxes = get_undistort_rgb(folder, rgb_name, boxes, False, False)

            # Excluded predicted bboxes outside ndvi zone
            valid_bboxes = getbboxinsidendvi(boxes, ndvi_area, img_size)

            for i, bbox in enumerate(valid_bboxes):
                
                ndvi_tree_name=rgb_name[:-5]+f'NDVI_{i}.JPG'
                rgb_tree_name=rgb_name[:-5]+f'RGB_{i}.JPG'

                # crop tree box
                ndvi_tree = crop_img_yolo_format(ndvi, bbox[1], bbox[2], bbox[3], bbox[4])
                rgb_tree = crop_img_yolo_format(rgb_img, bbox[1], bbox[2], bbox[3], bbox[4])

                # classification model needs images in RGB format
                ndvi_tree_pil = Image.fromarray(cv2.cvtColor(ndvi_tree, cv2.COLOR_BGR2RGB))
                rgb_tree_pil = Image.fromarray(cv2.cvtColor(rgb_tree, cv2.COLOR_BGR2RGB))                   
                t_ndvi_tree = transform(ndvi_tree_pil).unsqueeze(0).to(device) 
                t_rgb_tree = transform(rgb_tree_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    # classification health status
                    outputs = classification_model(t_ndvi_tree,t_rgb_tree)
                    label_id = torch.argmax(outputs, dim=1).item() 
                    ndvi_tree = overlay_text(ndvi_tree,label_id)
                    rgb_tree = overlay_text(rgb_tree,label_id)
                    save_img(output_predictions+'/ndvi', ndvi_tree_name, ndvi_tree)
                    save_img(output_predictions+'/rgb', rgb_tree_name, rgb_tree)

            # END img_set

        # END folder

    # END subfolders


if __name__ == "__main__":
    import os
    path = os.path.join('.', 'raw_data')
    main(path)
