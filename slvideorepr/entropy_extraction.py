import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import TruncatedSVD
from measures import *

from auxiliar import mostrar_imagen, normalizar_255, get_resize_scale, plot_1d_series

from collections import OrderedDict

max_face_width = 50
number_of_cells = 64

def entropy_extraction(input_video, output_dir, measure):
    '''Extract frames  with minimum change depending on a 1/4 second
    SVD calculation. Rolling window or not.'''
    
    # Create output directory structure
    try:
        os.makedirs(output_dir + "/cross_gray/")
        os.makedirs(output_dir + "/cross_position/")
        os.makedirs(output_dir + "/kl_gray/")
        os.makedirs(output_dir + "/kl_position/")
    except:
        pass

    capture = cv2.VideoCapture(input_video)

    if capture.isOpened() == False:
        raise ValueError("Something is wrong with video %s" % input_video)
    else:
        # Open, get width and height
        video_name = input_video.split("/")[-1].split(".")[-2]
        width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))

    checked_resize_values = False
    last_img = None
    last_patches = None
    last_coords = None
    has_last_img = False
    all_gray_entropy = []
    all_position_entropy = []
    all_gray_kl = []
    all_position_kl = []
    while capture.isOpened():
        ret, frame_img = capture.read()
        if ret == True:
            # Save a copy of frame
            current_img = frame_img.copy()
            # Calculate gray image
            gray_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            
            # Resize if need be
            if not checked_resize_values:
                scale = get_resize_scale(gray_img, max_face_width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                checked_resize_values = True
            if scale != 1:
                gray_img = cv2.resize(gray_img, (new_width, new_height), interpolation = cv2.INTER_AREA)
           
            if has_last_img:
                # Compare current (actual) with last (assumed)
                # Cut in patches
                current_patches, current_coords = cut_in_patches(gray_img, number_of_cells)
                gray_entropy_results = per_patch_comparison(
                                                        current_patches,
                                                        last_patches,
                                                        gray_level_crossentropy)
                all_gray_entropy.append(gray_entropy_results)

                position_entropy_results = per_patch_comparison(
                                                        current_patches,
                                                        last_patches,
                                                        position_crossentropy)
                all_position_entropy.append(position_entropy_results)

                gray_kl_results = per_patch_comparison(current_patches,
                                                       last_patches,
                                                       gray_level_kldiv)
                all_gray_kl.append(gray_kl_results)

                position_kl_results = per_patch_comparison(
                                                        current_patches,
                                                        last_patches,
                                                        position_kldiv)
                all_position_kl.append(position_kl_results)
            else:
                # Can't compare, just refresh
                last_img = gray_img
                last_patches, last_coords = cut_in_patches(gray_img, number_of_cells)
                has_last_img = True
        else:
            break
    capture.release()
    
    # Arranging details for graph generation
    gray_entropy_np = np.vstack(all_gray_entropy)
    position_entropy_np = np.vstack(all_position_entropy)
    gray_kl_np = np.vstack(all_gray_kl)
    position_kl_np = np.vstack(all_position_kl)

    # Generate graphs
    for i in range(number_of_cells):
        plot_1d_series(gray_entropy_np[:, i],
                output_dir + "/cross_gray/%d.png" % i)
        plot_1d_series(position_entropy_np[:, i],
                output_dir + "/cross_position/%d.png" % i)
        plot_1d_series(gray_kl_np[:, i],
                output_dir + "/kl_gray/%d.png" % i)
        plot_1d_series(position_kl_np[:, i],
                output_dir + "/kl_position/%d.png" % i)


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
                print("Entropy extraction")
                entropy_extraction(input_video, new_dir, measure)
