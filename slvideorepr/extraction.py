import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from measures import *


def process_individual_video_window(input_video, 
                                    output_dir,
                                    window_size,
                                    measure):
    '''Extract frames  with minimum change in a k-frame window'''
    
    if window_size % 2 == 0:
        raise ValueError("window_size has to be an odd number, not %d" % window_size)

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

    offset = np.floor(window_size/2)
    last_img_gray = None
    scores = []
    while capture.isOpened():
        ret, frame_img = capture.read()
        if ret == True:
            # Keep original
            current_img = frame_img.copy()
            
            # Calculate gray image
            gray_img = cv2.cvtColor(cv2.medianBlur(current_img, 5), 
                    cv2.COLOR_BGR2GRAY)
            
            # See if it's the first image
            try:
                last_img_gray.shape
            except:
                last_img_gray = gray_img

            # Calculate similarity
            if measure == "ssim" :
                current_m_score, diff_img = ssim(last_img_gray, 
                        gray_img, 
                        gaussian_weights=True, full=True)
                current_m_score = 1 - current_m_score # DSSIM_modif
            if measure == "gradient" :
                current_m_score = single_image_gradient_score(gray_img, window_size)
                current_m_score = 1 - current_m_score # Inverse acutance
            if measure == "patches" :
                current_m_score = nonlinear_comparison(last_img_gray,
                                     gray_img, 100)
            else:
                current_m_score, diff_img = mse(last_img_gray, gray_img)
            scores.append(current_m_score)
            last_img_gray = gray_img
        else:
            break
   
    scores = scores[:-1]
    capture.release()
    #kernel = np.ones(window_size) * (1 / window_size)
    #convolution_result = np.convolve(scores, kernel, "valid")

    #results = [(i+offset, value) for i, value in enumerate(scores)]
    results = [(i+1, value) for i, value in enumerate(scores)]
    results = []
    #for i, value in enumerate(scores):
    #    if len(results) == 0:
    #        results.append((i+1, value))
    #    else:
    #        if value < 0.1:
    #            results.append((i+1, results[-1][1]))
    #        else:
    #            results.append((i+1, value))
    i = 0
    for value in scores:
        if len(results) == 0:
            results.append((i+1, value))
            i += 1
        else:
            if value >= 1:
                results.append((i+1, value))
                i += 1
    results = np.array(results)

    plt.plot(results[:, 0],
         results[:, 1])
    
    plt.savefig(output_dir + "/{0}_plot.png".format(measure), dpi=300)
    plt.close()
    with open(output_dir + "/{0}_values.txt".format(measure), "w") as fp:
        #fp.writelines([str(n) + "\n" for n in convolution_result])
        fp.writelines([str(n) + "\n" for n in scores])

    print("Done video %s" % input_video)

def process_individual_video(input_video, 
                             output_dir,
                             measure="ssim"):
    '''Extract frames  with minimum change'''
    
    # Create output directory structure
    try:
        os.makedirs(output_dir)
    except:
        pass
   
    capture = cv2.VideoCapture(input_video)

    if capture.isOpened() == False:
        raise ValueError("Something is wrong with video %s" % input_video)

    last_img_color = None
    last_img_gray = None
    is_first = True
    descending = True
    frame = 0 
    m_series = []
    minimum_series = []
    while capture.isOpened():
        ret, frame_img = capture.read()
        if ret == True:
            frame += 1
            
            # Keep original
            current_img = frame_img.copy()
            
            # Calculate gray image
            gray_img = cv2.cvtColor(cv2.medianBlur(current_img, 5), 
                    cv2.COLOR_BGR2GRAY)
            
            if is_first:
                m_series.append(0) #DSSIM
                #m_series.append(1) #SSIM
                is_first = False
            else:
                last_m_score = m_series[-1]
                if measure == "ssim":
                    current_m_score, diff_img = ssim(last_img_gray, 
                            gray_img, 
                            gaussian_weights=True, full=True)
                    current_m_score = 1 - current_m_score #DSSIM_modif
                else:
                    current_m_score, diff_img = mse(last_img_gray, gray_img)

                if last_m_score < current_m_score:
                    if descending == True:
                        # was descending, here it goes up
                        # then, last image was keyframe
                        cv2.imwrite(output_dir + "/%04d.jpg" % (frame - 1),
                                last_img_color)
                        minimum_series.append((frame-1, last_m_score))
                    descending = False
                else:
                    descending = True
                m_series.append(current_m_score)
            last_img_color = current_img
            last_img_gray = gray_img
        else:
            break
    
    capture.release()
    results = [(i+1, value) for i, value in enumerate(m_series)]
    results = np.array(results)

    minima = np.array(minimum_series)
    plt.plot(results[:, 0],
         results[:, 1])
    plt.plot(minima[:, 0],
             minima[:, 1],
             "o")
    
    plt.savefig(output_dir + "/{0}_plot.png".format(measure), dpi=300)
    plt.close()
    with open(output_dir + "/{0}_values.txt".format(measure), "w") as fp:
        fp.writelines([str(n) + "\n" for n in m_series])

    print("Done video %s" % input_video)

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
                process_individual_video_window(input_video, new_dir, 5, measure)
