from loadData_mavic3m import get_ms_data
import imageProcessingUtils as ip
import os
import cv2
import numpy as np
from tqdm import tqdm


def get_ndvi_color_gradient():
    from colour import Color
    white = Color("#FFFFFF")
    black = Color("#000000")
    indaco = Color("#7F00FF")
    ciano = Color("#00FFFF")
    verde = Color("#00FF00")
    giallo = Color("#FFFF00")
    magenta = Color("#FF0000")
    viola = Color("#FF00FF")
    gradient = list(white.range_to(black, 37))
    gradient.extend(list(black.range_to(white, 37)))
    gradient.extend(list(white.range_to(black, 37)))
    gradient.extend(list(indaco.range_to(verde, 10)))
    #gradient.extend(list(ciano.range_to(verde, 8))[1:])
    gradient.extend(list(verde.range_to(giallo, 31))[1:])
    gradient.extend(list(giallo.range_to(magenta, 41))[1:])
    gradient.extend(list(magenta.range_to(viola, 11))[1:])

    # Converte la lista in rgb
    gradient = [(round(255*x.rgb[0]), round(255*x.rgb[1]), round(255*x.rgb[2])) for x in gradient]
    return gradient

def ranges(a, b, n):
    step = (b - a) / n
    list = [a]
    temp = a
    for i in range(n):
        temp = round(temp + step, 2)
        list.append(temp)

    return list


def ndvi_modern_false_color(gray_ndvi):
    # Definisce gli intervalli NDVI per la scala di colori
    ndvi_intervals = ranges(-1,1,200)
    # Definisce i colori corrispondenti agli intervalli NDVI (scala moderna)
    colors = get_ndvi_color_gradient()

    #Stampa colore - soglia associata
    for i in range(len(colors)):
        print(colors[i], ndvi_intervals[i])

    # Mappa i valori NDVI agli intervalli e assegna i colori corrispondenti
    false_color_ndvi = np.zeros((gray_ndvi.shape[0], gray_ndvi.shape[1], 3), dtype=np.uint8)
    for i in tqdm(range(len(ndvi_intervals) - 1)):
        mask = np.logical_and(gray_ndvi >= ndvi_intervals[i], gray_ndvi < ndvi_intervals[i + 1])
        false_color_ndvi[mask] = colors[i]

    bgr_img = cv2.cvtColor(false_color_ndvi, cv2.COLOR_RGB2BGR)

    return bgr_img


def get_NDVI(path, nir_name, red_name, show=False, save=False, savePath=None):
    
    if save:
        if savePath is None:
            savePath = os.path.join(path, 'output', 'ndvi')

    nir_data = get_ms_data(os.path.join(path, nir_name))
    red_data = get_ms_data(os.path.join(path, red_name))

    nir_img = nir_data['img']
    red_img = red_data['img']

    camera_corners = np.array([(0, 0), (0, 2592), (1944, 2592), (1944, 0)])
    camera_corners = np.flip(camera_corners)
    camera_corners = np.float32(camera_corners[:, np.newaxis, :])

    if show:
        ip.show_img('Original RED', red_img)
        ip.show_img('Original NIR', nir_img)


    # Vignettign corretion
    print()
    print('Applying RED vignetting correction')
    red_img = ip.vignetting_correction(red_img, red_data)
    if save:
        ip.save_img(os.path.join(savePath, 'vignettignCorrection'),
            red_name[:-3] + 'jpg', red_img)
    if show:
        ip.show_img('Vignetting correction RED', red_img)

    print()
    print('Applying NIR vignetting correction')
    nir_img = ip.vignetting_correction(nir_img, nir_data)
    if save:
        ip.save_img(os.path.join(savePath, 'vignettignCorrection'),
             nir_name[:-3] + 'jpg', nir_img)
    if show:
        ip.show_img('Vignetting correction NIR', nir_img)


    # Distorsion correction
    print()
    print('Applying distortion correction RED')
    red_img, red_roi = ip.distortion_correction(red_img, red_data)
    if save:
        ip.save_img(os.path.join(savePath, 'distortionCorrection'),
             red_name[:-3] + 'jpg', red_img)
    if show:
        ip.show_img('Distortion correction RED', red_img)
    red_camera_corners = cv2.perspectiveTransform(camera_corners, red_data['hmatrix'])

    print()
    print('Applying distortion correction NIR')
    nir_img, nir_roi = ip.distortion_correction(nir_img, nir_data)
    if save:
        ip.save_img(os.path.join(savePath, 'distortionCorrection'),
             nir_name[:-3] + 'jpg', nir_img)
    if show:
        ip.show_img('Distortion correction NIR', nir_img)
    nir_camera_corners = cv2.perspectiveTransform(camera_corners, nir_data['hmatrix'])


    # Optical Allignment
    print()
    print('Applying optical allignment RED')
    height, width, depth = red_img.shape
    red_img = cv2.warpPerspective(red_img, red_data['hmatrix'], (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
    if save:
        ip.save_img(os.path.join(savePath, 'phaseAlignment'),
             red_name[:-3] + 'jpg', red_img)
    if show:
        ip.show_img('Optical Allignment 1 RED', red_img)

    print()
    print('Applying optical allignment NIR')
    height, width, depth = nir_img.shape
    nir_img = cv2.warpPerspective(nir_img, nir_data['hmatrix'], (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
    if save:
        ip.save_img(os.path.join(savePath, 'phaseAlignment'),
             nir_name[:-3] + 'jpg', nir_img)
    if show:
        ip.show_img('Optical Allignment 1 NIR', nir_img)


    # Exposure time allignment
    print()
    print('Applying allignment RED on NIR')
    try:
        red_img, warp_matrix = ip.alignmentECC(nir_img, red_img)
    except cv2.error as e:
        # If not converge skip...
        pass

    if save:
        ip.save_img(os.path.join(savePath, 'TimeAlignment'), red_name[:-3] + 'jpg', red_img)
    if show:
        ip.show_img('Alligned RED on NIR', red_img)


    #NDVI
    print()
    print('Calculating NDVI')

    height, width, depth = nir_img.shape
    nir_pcam = nir_data['sensorGainAdjustment']
    nir_irradiance = nir_data['irradiance']
    nir_camera = np.empty((height,width))

    red_camera = np.empty((height,width))
    red_pcam = red_data['sensorGainAdjustment']
    red_irradiance = red_data['irradiance']

    ndvi = np.empty((height,width))

    for x in tqdm(range(height)):
        for y in range(width):
            nir_camera[x, y] = (((nir_img[x, y][0] / (2 ** nir_data['bps'])) - (nir_img[x, y][0] /
                                                                                     (nir_data['blackLevel']))) / (
                                              nir_data['sensorGain'] * (nir_data['exposureTime'] / 1e6)))

            red_camera[x, y] = (((red_img[x, y][0] / (2 ** red_data['bps'])) - (red_img[x, y][0] /
                                                                                     (red_data['blackLevel']))) / (
                                              red_data['sensorGain'] * (red_data['exposureTime'] / 1e6)))
            nir_idx = ((nir_camera[x, y] * nir_pcam) / nir_irradiance)
            red_idx = ((red_camera[x, y] * red_pcam) / red_irradiance)
            if nir_idx != 0 or red_idx != 0:
                ndvi[x, y] = ((((nir_camera[x, y] * nir_pcam) / nir_irradiance) - ((red_camera[x, y] * red_pcam) / red_irradiance)) /
                                 (((nir_camera[x, y] * nir_pcam) / nir_irradiance) + ((red_camera[x, y] * red_pcam) / red_irradiance)))

    grey_ndvi = np.array([x * 255 for x in ndvi]).astype(np.uint8)
    if save:
        ip.save_img(os.path.join(savePath, 'ndvi'), red_name[:-5] + 'GREY_NDVI.jpg', grey_ndvi)
    if show:
        ip.show_img('NDVI', grey_ndvi)
    

    # false color NDVI
    ndvi = ndvi_modern_false_color(ndvi)
    if save:
        ip.save_img(os.path.join(savePath, 'ndvi'), red_name[:-5] + 'NDVI.jpg', ndvi)
    if show:
        ip.show_img('NDVI', ndvi)

    # Get actual NDVI area in MS resolution
    ndvi_ll_w = max(nir_camera_corners[0][0][0], red_camera_corners[0][0][0])
    ndvi_ll_h = max(nir_camera_corners[0][0][1], red_camera_corners[0][0][1])
    ndvi_lr_w = min(nir_camera_corners[1][0][0], red_camera_corners[1][0][0])
    ndvi_lr_h = max(nir_camera_corners[1][0][1], red_camera_corners[1][0][1])
    ndvi_tr_w = min(nir_camera_corners[2][0][0], red_camera_corners[2][0][0])
    ndvi_tr_h = min(nir_camera_corners[2][0][1], red_camera_corners[2][0][1])
    ndvi_tl_w = min(nir_camera_corners[3][0][0], red_camera_corners[3][0][0])
    ndvi_tl_h = max(nir_camera_corners[3][0][1], red_camera_corners[3][0][1])

    ndvi_corners = np.array(
        [[[ndvi_ll_w, ndvi_ll_h]], [[ndvi_lr_w, ndvi_lr_h]], [[ndvi_tr_w, ndvi_tr_h]], [[ndvi_tl_w, ndvi_tl_h]]])
    for point in ndvi_corners:
        if point[0][0] < 0:
            point[0][0] = 0
        if point[0][0] > 2592:
            point[0][0] = 2592
        if point[0][1] < 0:
            point[0][1] = 0
        if point[0][1] > 1944:
            point[0][1] = 1944


    # NDVI to RGB Warping
    print()
    print('Warping NDVI to RGB')
    height = 3956
    width = 5280
    ndvi = cv2.warpPerspective(ndvi, red_data['dewarpHMatrix'], (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
    if show:
        ip.show_img('Optical Allignment NDVI', ndvi)
    if save:
        ip.save_img(os.path.join(savePath, 'ndvi'), red_name[:-5] + 'WARPED_NDVI.jpg', ndvi)

    ndvi_area = np.int32(cv2.perspectiveTransform(ndvi_corners, nir_data['dewarpHMatrix']))
    ndvi_box = []
    for point in ndvi_area:
        ndvi_box.append([point[0][0], point[0][1]])
    return ndvi, ndvi_box


if __name__ == "__main__":
    path = os.path.join('.', 'data', 'dati_grezzi_drone', 'Test', 'OneStack')
    red_name = 'DJI_20240525115637_0026_MS_R.TIF'
    nir_name = 'DJI_20240525115637_0026_MS_NIR.TIF'
    savepath = os.path.join('.', 'data', 'output', 'Test')
    ndvi = get_NDVI(path, nir_name, red_name, False, True, savepath)
