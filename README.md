<div align="center">
  <p>
    <a href="https://icaerus.eu" target="_blank">
      <img width="50%" src="https://icaerus.eu/wp-content/uploads/2022/09/ICAERUS-logo-white.svg"></a>
    <h3 align="center">Full Pipeline Olive Tree Health Status Classification🦚</h3>
    
   <p align="center">
    Full Pipeline Olive Tree Health Status Classification
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/repo-title/wiki"><strong>Explore the wiki »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/repo-title/issues">Report Bug</a>
    -
    <a href="https://github.com/icaerus-eu/repo-title/issues">Request Feature</a>
  </p>
</p>
</div>

![Downloads](https://img.shields.io/github/downloads/icaerus-eu/repo-title/total) ![Contributors](https://img.shields.io/github/contributors/icaerus-eu/repo-title?color=dark-green) ![Forks](https://img.shields.io/github/forks/icaerus-eu/repo-titlee?style=social) ![Stargazers](https://img.shields.io/github/stars/icaerus-eu/repo-title?style=social) ![Issues](https://img.shields.io/github/issues/icaerus-eu/repo-title) ![License](https://img.shields.io/github/license/icaerus-eu/repo-title) 


## Table Of Contents
- [Table Of Contents](#table-of-contents)
- [Summary](#summary)
- [Installation](#installation)
- [Documentation](#documentation)
  - [Integration of Multiple Modules](#integration-of-multiple-modules)
  - [Device Configuration and Model Loading](#device-configuration-and-model-loading)
  - [Image and Data Handling](#image-and-data-handling)
  - [Image Preprocessing](#image-preprocessing)
  - [NDVI Calculation and Analysis](#ndvi-calculation-and-analysis)
  - [Object Detection and Classification Workflow](#object-detection-and-classification-workflow)
  - [Result Management](#result-management)
- [Acknowledgements](#acknowledgements)

## Summary
The script files orchestrates a image processing workflow designed to classify olive trees by combining data from NDVI (Normalized Difference Vegetation Index) and RGB images. The script leverages deep learning models to perform object detection and classification, specifically targeting the health and characteristics of olive trees.

The core idea behind this approach is to utilize the complementary strengths of NDVI and RGB imagery. NDVI, derived from multispectral images, provides crucial insights into the vegetation health by highlighting areas with different levels of photosynthetic activity. RGB images, on the other hand, offer detailed visual information about the structure and appearance of the trees.

## Installation
Install python 3.10 and requred libraries listed in ```requirements.txt```.

## Documentation

### Integration of Multiple Modules
Several critical modules and libraries have been used in the scripts:

```PyTorch``` and ```TorchVision``` for deep learning model handling and image transformations.
```OpenCV``` for image processing operations.
```PIL``` (Python Imaging Library) for handling image files.
Custom modules like ```bboxProcessing```, ```loadData_mavic3m```, ```ndvi```, and ```imageProcessingUtils```, which provide specialized functions for bounding box manipulation, image data loading, NDVI calculation, and image utilities.
This integration allows ```main.py``` to serve as a bridge between data acquisition, processing, and analysis, ensuring a seamless workflow.

### Device Configuration and Model Loading
One of the early steps in the script is to check for CUDA availability, which determines whether the computations will be offloaded to a GPU for faster processing. This is crucial in scenarios where large datasets or computationally intensive models like YOLO are involved.

The script loads two key models:

- A YOLO detection model for identifying objects within the images. This model is loaded using a pre-trained set of weights (```best.pt```).
- A classification model (DualYOLO) that is designed to classify the health status of detected objects. This model is loaded from the best checkpoint file, according to our analysis (```epoch=28-val_loss=0.63.ckpt```), and is set to evaluation mode, ensuring that it performs inference without modifying its weights.

### Image and Data Handling
The script efficiently manages image data through functions that:

List subfolders within a specified directory to organize and access multiple sets of images.
Load image sets from each folder, associating multispectral images (e.g., NIR, RED) with their corresponding RGB images. This association is crucial for the subsequent analysis steps that combine data from different spectral bands.
These steps ensure that the script can handle datasets with complex structures, where images are stored across different directories or have specific naming conventions.

### Image Preprocessing
The script includes a preprocessing pipeline where each image is transformed to a standard size (```640x640``` pixels) and converted to a tensor format. This preprocessing is critical for ensuring that the images are compatible with the input requirements of the deep learning models.

Additionally, the script resizes the RGB images to a smaller resolution (```960x640```) for faster processing during certain steps, such as displaying results or performing initial analyses.

### NDVI Calculation and Analysis
The Normalized Difference Vegetation Index (NDVI) is a crucial spectral reflectance index used to assess the health and vigor of vegetation. This index is calculated using the near-infrared (NIR) and red light (RED) bands of the electromagnetic spectrum. The basic principle behind NDVI is that healthy vegetation absorbs most of the visible light (particularly in the red spectrum) and reflects a significant portion of the NIR light. Conversely, unhealthy or stressed vegetation reflects more visible light and less NIR light.

While the general formula for NDVI is:

```NDVI = (NIR−RED)/(NIR+RED)​```

the specific calculation can vary depending on the equipment and metadata provided by the capturing device. For instance, when using drone data, the calculation may follow specific guidelines provided by the drone manufacturer to ensure accuracy.

The NDVI values range between -1 and +1, where values greater than 0.2 typically indicate the presence of vegetation. Higher NDVI values suggest healthier, more vigorous vegetation, while lower values indicate stressed, sparse, or non-vegetated areas.

Historically, NDVI has been used on a large scale to monitor vegetation health across regions, states, and even globally, primarily using satellite imagery. Over time, with the advancement of remote sensing technology and the introduction of UAVs (Unmanned Aerial Vehicles), the application of NDVI has become more precise and scalable. Nowadays, NDVI is employed not only on large-scale agricultural fields but also at the level of individual plants or even within controlled environments like greenhouses.

In this projects NDVI has been selected for its ability to provide valuable insights into the health of olive trees. The index has shown effectiveness in distinguishing between healthy trees, those with mild symptoms, and trees with significant disease symptoms. By utilizing high-resolution data, even down to sub-centimeter levels, NDVI can assess the health status of different parts of a tree's canopy with remarkable precision.

The scripts integrates NDVI analysis as follows:

Image Preprocessing: The script preprocesses the images, aligning and resizing them to ensure they are in the correct format for analysis. This step may also involve correcting for distortions and calibrating the images to enhance accuracy, following specific guidelines from the [DJI Mavic 3M Image Processing Guide](https://dl.djicdn.com/downloads/DJI_Mavic_3_Enterprise/20230829/Mavic_3M_Image_Processing_Guide_EN.pdf).

NDVI Calculation: The core NDVI calculation is performed using the specific formula and procedures recommended by the DJI guide. The script computes this index for each pixel in the image, resulting in an NDVI map that represents the vegetation health across the area of interest.

False Color Mapping: After calculating the NDVI, the script might apply a false-color gradient to the NDVI values, mapping them to a color scale that visually distinguishes between different vegetation health levels. This mapping helps in easily interpreting the results, where colors can represent different NDVI ranges, indicating varying levels of plant health or stress.

Analysis and Classification: The NDVI data is then used in conjunction with other analysis techniques to classify the health status of the vegetation. In the case of olive trees, for example, NDVI data helps in categorizing trees into different health status categories based on the severity of symptoms observed in the NDVI map.

### Object Detection and Classification Workflow
The core of the script's functionality lies in its ability to:

Detect objects in the images using the YOLO model. The bounding boxes generated from this detection are then processed to filter out only those within areas of interest (e.g., based on NDVI results).
Classify the detected objects to assess their health status using the pre-loaded classification model. This step is crucial for applications in agriculture or environmental monitoring, where identifying the health of plants or other objects is of primary interest.
This dual-step process of detection followed by classification ensures that the script not only identifies where objects are located but also provides insights into their condition.

### Result Management
The script is designed to manage the output efficiently by:

Saving predictions and processed images to a specified output directory (```./predictions```). This ensures that all results are systematically stored for further review or analysis.
Overlaying text and annotations on the images to visually communicate the detection and classification results.
These features are essential for creating a user-friendly output that can be easily interpreted by stakeholders or used for generating reports.

## Acknowledgements
This project is funded by the European Union, grant ID 101060643.

<img src="https://rea.ec.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2021-04/EN-Funded%20by%20the%20EU-POS.jpg" alt="https://cordis.europa.eu/project/id/101060643" width="200"/>
