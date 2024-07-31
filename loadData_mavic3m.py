import os
import cv2
import numpy as np
import pyexiv2

def list_subfolders(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    return subfolders


def load_images_from_folder(folder):
    image_paths = []
    for filename in os.listdir(folder):
        set = []
        if filename.endswith('_D.JPG'): # Per ogni foto RGB carico le 4 multispettrali
            set.append(filename)
            set_number = filename.split('_')[2]
            for filename_ms in os.listdir(folder):
                if filename_ms.endswith((set_number + '_MS_G.TIF', set_number + '_MS_NIR.TIF', set_number + '_MS_R.TIF', set_number + '_MS_RE.TIF')):
                    set.append(filename_ms)
            image_paths.append(set)
    return image_paths


def read_image_exif_data(image_path):
    
    img = pyexiv2.Image(image_path)
    data = img.read_exif()
    img.close()
    return data


def read_image_xmp_data(image_path):
    fd = open(image_path, 'rb')
    d = fd.read()
    xmp_start = d.find(b'<x:xmpmeta')
    xmp_end = d.find(b'</x:xmpmeta')
    xmp_str = d[xmp_start:xmp_end + 12]
    return xmp_str

def get_xmp_value(xmp, key):
    lines = xmp.split(b'\n')
    for line in lines:
        if line.find(key.encode('utf-8')) > 0:
            value = line.split(b'"')[1].decode('utf-8')
            return value

def get_dewarp_dict(xmp):
    data = get_xmp_value(xmp, 'DewarpData')
    data = data.split(';')
    day = data[0]
    data = data[1].split(',')
    dict = {
        'day': day,
        'fx': float(data[0]),
        'fy': float(data[1]),
        'cx': float(data[2]),
        'cy': float(data[3]),
        'dist': np.array([float(data[4]), float(data[5]), float(data[6]), float(data[7]), float(data[8])])
    }
    return dict


def get_rgb_data(rgb_path):
    # Carica l'immagine
    rgb_img = cv2.imread(rgb_path)
    # Carica metadati XMP
    rgb_xmp = read_image_xmp_data(rgb_path)
    # Carica metedati EXIF
    rgb_exif = read_image_exif_data(rgb_path)
    # Crea il dizionario con i dati utili
    rgb_data = {
        'img': rgb_img,
        'centerX': float(get_xmp_value(rgb_xmp, 'CalibratedOpticalCenterX')),
        'centerY': float(get_xmp_value(rgb_xmp, 'CalibratedOpticalCenterY')),
        # 'vignetting': get_xmp_value(rgb_xmp, 'VignettingData'),
        'dewarp': get_dewarp_dict(rgb_xmp),
        # 'hmatrix': get_xmp_value(rgb_xmp, 'CalibratedHMatrix'),
        # 'bps': rgb_exif['Exif.Image.BitsPerSample'],
        'focalLength': rgb_exif['Exif.Photo.FocalLength'],
        # 'blackLevel': get_xmp_value(rgb_xmp, 'BlackLevel'),
        # 'sensorGain': get_xmp_value(rgb_xmp, 'SensorGain'),
        # 'exposureTime': get_xmp_value(rgb_xmp, 'ExposureTime'),
        # 'sensorGainAdjustment': get_xmp_value(rgb_xmp, 'SensorGainAdjustment'),
        # 'irradiance': get_xmp_value(rgb_xmp, 'Irradiance')
    }
    return rgb_data


def get_ms_data(ms_path):
    # Carica l'immagine
    ms_img = cv2.imread(ms_path)
    # Carica metadati XMP
    ms_xmp = read_image_xmp_data(ms_path)
    # Carica metedati EXIF
    ms_exif = read_image_exif_data(ms_path)
    # Crea il dizionario con i dati utili
    ms_data = {
        'img': ms_img,
        'centerX': float(get_xmp_value(ms_xmp, 'CalibratedOpticalCenterX')),
        'centerY': float(get_xmp_value(ms_xmp, 'CalibratedOpticalCenterY')),
        'vignetting': [float(x) for x in get_xmp_value(ms_xmp, 'VignettingData').split(', ')],
        'dewarp': get_dewarp_dict(ms_xmp),
        'dewarpHMatrix': np.asmatrix(np.array([float(x) for x in (get_xmp_value(ms_xmp, 'DewarpHMatrix').split(','))])).reshape(3,3),
        'hmatrix': np.asmatrix(np.array([float(x) for x in (get_xmp_value(ms_xmp, 'CalibratedHMatrix').split(','))])).reshape(3,3),
        'bps': int(ms_exif['Exif.Image.BitsPerSample']),
        'focalLength' : ms_exif['Exif.Photo.FocalLength'],
        'blackLevel': int(get_xmp_value(ms_xmp, 'BlackLevel')),
        'sensorGain': float(get_xmp_value(ms_xmp, 'SensorGain')),
        'exposureTime': int(get_xmp_value(ms_xmp, 'ExposureTime')),
        'sensorGainAdjustment': float(get_xmp_value(ms_xmp, 'SensorGainAdjustment')),
        'irradiance': float(get_xmp_value(ms_xmp, 'Irradiance'))
    }
    return ms_data
