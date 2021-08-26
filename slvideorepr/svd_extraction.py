import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import TruncatedSVD
from measures import *

from auxiliar import mostrar_imagen, normalizar_255, get_resize_scale, plot_1d_series

from collections import OrderedDict

max_face_width = 50

def calculate_img_histogram(gray_img):
    hist = []
    for i in range(256):
        hist.append(gray_img[gray_img == i].size)
    return np.array(hist)

def calculate_gray_entropy(gray_img):
    hist = calculate_img_histogram(gray_img)
    norm_hist = hist/hist.sum()
    entropy = 0.
    for k in range(256):
        if norm_hist[k] != 0:
            entropy += norm_hist[k] * np.log2(norm_hist[k])
    entropy *= -1
    return entropy

def calculate_space_entropy(svd_img):
    norm_svd_img = svd_img / svd_img.sum() 
    entropy = 0.
    for i in range(norm_svd_img.shape[0]):
        for j in range(norm_svd_img.shape[1]):
            if norm_svd_img[i, j] != 0:
                entropy += norm_svd_img[i, j] * np.log2(norm_svd_img[i, j])
    entropy *= -1
    return entropy

def generate_bin_index(coords):
    '''generates an index from coord patches'''
    _, max_h, _, max_w = coords[-1]

    index = OrderedDict.fromkeys(range(max_h + 1), 
                                 OrderedDict({})) 
    
    cells = []
    for h1, h2, w1, w2 in coords:
        member = []
        for i in range(h1, h2 + 1):
            for j in range(w1, w2 + 1):
                index[i][j] = member
        cells.append(member)
    return index, cells

def calculate_bins_entropy(svd_img, no_bins=64):
    divisions_per_side = np.sqrt(no_bins)
    region_height = int(svd_img.shape[0] / divisions_per_side)
    region_width = int(svd_img.shape[1] / divisions_per_side)
    patches, coords = image_crop_alt(svd_img, region_height, region_width)
    index, cells = generate_bin_index(coords)

    for i in range(svd_img.shape[0]):
        for j in range(svd_img.shape[1]):
            index[i][j].append(svd_img[i, j])
    summed_cells = np.array([np.sum(c) for c in cells])
    norm_summed_cells = summed_cells / summed_cells.sum()

    entropy = 0.
    for k in range(len(cells)):
        if norm_summed_cells[k] != 0:
            entropy += norm_summed_cells[k] * np.log2(norm_summed_cells[k])
    entropy *= -1
    return entropy

def calculate_svd_image(image_group):
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
    return mean_img.reshape((height, width))

def svd_255_normalization(mean_img, show=False):
    # Set to zero every negative value
    mean_img[mean_img < 0] = 0
    norm_mean_img = normalizar_255(mean_img)
    if show:
        mostrar_imagen(norm_mean_img, "Eigenimage")
    return norm_mean_img

def thresholding(norm_mean_img, percentile=99, show=False):
    norm_mean_img = (norm_mean_img > np.percentile(norm_mean_img, percentile)).astype("uint8")
    norm_mean_img *= 255
    if show:
        mostrar_imagen(norm_mean_img, "Percentile %d" % percentile)
    return norm_mean_img

def svd_extraction(input_video, output_dir, measure, rolling="False"):
    '''Extract frames  with minimum change depending on a 1/4 second
    SVD calculation. Rolling window or not.'''
    
    # Create output directory structure
    output_dir += "/svd/"
    try:
        os.makedirs(output_dir)
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

    frame_window = int(fps/4) # 1/4 second
    checked_resize_values = False
    image_group = []
    key_frames = []
    gray_entropy_results = []
    space_entropy_results = []
    bins_entropy_results = []
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
            
            # Normalize
            gray_img = gray_img.astype("float64")
            gray_img /= 255

            if len(image_group) < frame_window:
                image_group.append(gray_img)
                if len(image_group) == frame_window:
                    # PROCESS GROUP
                    svd_image = calculate_svd_image(image_group)
                    normalized_svd_image = svd_255_normalization(svd_image)
                    svd_binary = thresholding(normalized_svd_image)
                    space_entropy = calculate_space_entropy(svd_image)
                    gray_entropy = calculate_gray_entropy(normalized_svd_image)
                    bins_entropy = calculate_bins_entropy(svd_image)
                    print(gray_entropy, space_entropy, bins_entropy)
                    gray_entropy_results.append(gray_entropy)
                    space_entropy_results.append(space_entropy)
                    bins_entropy_results.append(bins_entropy)
                    key_frames.append(svd_binary)

                    # Finished, preparing group for next calculation
                    if not rolling:
                        image_group = [gray_img]
                    else:
                        image_group.pop(0)

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
            if not rolling:
                if len(image_group) > 1:
                    # PROCESS GROUP
                    svd_image = calculate_svd_image(image_group)
                    normalized_svd_image = svd_255_normalization(svd_image)
                    svd_binary = thresholding(normalized_svd_image)
                    space_entropy = calculate_space_entropy(svd_image)
                    gray_entropy = calculate_gray_entropy(normalized_svd_image)
                    bins_entropy = calculate_bins_entropy(svd_image)
                    print(gray_entropy, space_entropy, bins_entropy)
                    gray_entropy_results.append(gray_entropy)
                    space_entropy_results.append(space_entropy)
                    bins_entropy_results.append(bins_entropy)
                    key_frames.append(svd_binary)
            break
    
    capture.release()
    if not rolling:
        plot_1d_series(gray_entropy_results, output_dir + "/%s_not_rolling_gray.png" % video_name)
        plot_1d_series(space_entropy_results, output_dir + "/%s_not_rolling_space.png" % video_name)
        plot_1d_series(bins_entropy_results, output_dir + "/%s_not_rolling_bins.png" % video_name)
    else:
        plot_1d_series(gray_entropy_results, output_dir + "/%s_rolling_gray.png" % video_name)
        plot_1d_series(space_entropy_results, output_dir + "/%s_rolling_space.png" % video_name)
        plot_1d_series(bins_entropy_results, output_dir + "/%s_rolling_bins.png" % video_name)

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
                print("ROLLING WINDOW")
                svd_extraction(input_video, new_dir, measure, rolling=True)
                print("NO ROLLING WINDOW")
                svd_extraction(input_video, new_dir, measure, rolling=False)
