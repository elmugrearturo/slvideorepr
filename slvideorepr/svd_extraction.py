import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import TruncatedSVD
from measures import *

from auxiliar import mostrar_imagen, normalizar_255

def svd_segmentation(image_group):
    height, width = image_group[0].shape
    svd = TruncatedSVD(len(image_group))
    img_array = np.array([i.flatten() for i in image_group])
    svd.fit(img_array)
    for i in range(len(image_group)):
        if svd.explained_variance_ratio_[0:i].sum() >= .8:
            # We have enough variance
            break
    # Calculate a single image to display
    mean_img = svd.components_[0:i].sum(axis=0)
    # Set to zero every negative value
    mean_img[mean_img < 0] = 0
    norm_mean_img = normalizar_255(mean_img).reshape((height, width))
    mostrar_imagen(norm_mean_img, "Eigenimage")
    norm_mean_img = (norm_mean_img > np.percentile(norm_mean_img, 99)).astype("uint8")
    norm_mean_img *= 255
    norm_mean_img.reshape((height, width))
    mostrar_imagen(norm_mean_img, "Percentile")

def svd_extraction(input_video, output_dir, frame_window, measure="ssim"):
    '''Extract frames  with minimum change in a k-frame NON-ROLLING window'''
    
    # Create output directory structure
    try:
        os.makedirs(output_dir)
    except:
        pass

    capture = cv2.VideoCapture(input_video)

    if capture.isOpened() == False:
        raise ValueError("Something is wrong with video %s" % input_video)
    else:
        # Open, get width and height
        width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    image_group = []
    key_frames = []
    while capture.isOpened():
        ret, frame_img = capture.read()
        if ret == True:
            # Save a copy of frame
            current_img = frame_img.copy()
            # Calculate gray image
            gray_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY).astype("float64")
            # Normalize
            gray_img /= 255
            if len(image_group) < frame_window:
                image_group.append(gray_img)
                if len(image_group) == frame_window:
                    # PROCESS GROUP
                    key_frames.append(svd_segmentation(image_group))
            else:
                # NON-ROLLING WINDOW
                image_group = [gray_img]
            ## Calculate similarity
            #if measure == "ssim" :
            #    current_m_score, diff_img = ssim(last_img_gray, 
            #            gray_img, 
            #            gaussian_weights=True, full=True)
            #    current_m_score = 1 - current_m_score # DSSIM_modif
            #if measure == "gradient" :
            #    current_m_score = single_image_gradient_score(gray_img, window_size)
            #    current_m_score = 1 - current_m_score # Inverse acutance
            #if measure == "patches" :
            #    current_m_score = nonlinear_comparison(last_img_gray,
            #                         gray_img, 100)
            #else:
            #    current_m_score, diff_img = mse(last_img_gray, gray_img)
            #scores.append(current_m_score)
            #last_img_gray = gray_img
        else:
            if len(image_group) > 1:
                # PROCESS GROUP
                key_frames.append(svd_segmentation(image_group))
            break
   
    capture.release()

def process_corpora(input_dir, output_dir, measure="ssim"):
    # Ensure input_dir format
    if input_dir.endswith(os.sep):
        input_dir = input_dir[:-1]

    # Recursively discover videos
    for root, subdirs, subfiles in os.walk(input_dir):
        for subfile in subfiles:
            # TODO: discover by metadata and not by extension
            if subfile.endswith(".mp4") or subfile.endswith(".avi") or subfile.endswith(".mkv"):
                # Get video path
                input_video = root + "/" + subfile
                
                # Get output folder for the extracted images
                new_dir = root + "/" + subfile.split(".")[0]
                new_dir = new_dir.replace(input_dir, str(output_dir))
                
                # Extract images
                svd_extraction(input_video, new_dir, 10, measure)
