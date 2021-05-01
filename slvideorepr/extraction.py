import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import structural_similarity as ssim

def process_individual_video(input_video, output_dir):
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
    ascending = True
    frame = 0 
    ssim_series = []
    maximum_series = []
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
                ssim_series.append(1)
                is_first = False
            else:
                last_ssim_score = ssim_series[-1]
                current_ssim_score, diff_img = ssim(last_img_gray, 
                        gray_img, 
                        gaussian_weights=True, full=True)

                if last_ssim_score > current_ssim_score:
                    if ascending == True:
                        # was ascending, here it goes down
                        # last image was keyframe
                        cv2.imwrite(output_dir + "/%04d.jpg" % (frame - 1),
                                last_img_color)
                        maximum_series.append((frame-1, last_ssim_score))
                    ascending = False
                else:
                    ascending = True
                ssim_series.append(current_ssim_score)
            last_img_color = current_img
            last_img_gray = gray_img
        else:
            break
    
    capture.release()
    results = [(i+1, value) for i, value in enumerate(ssim_series)]
    results = np.array(results)

    maxima = np.array(maximum_series)
    plt.plot(results[:, 0],
         results[:, 1])
    plt.plot(maxima[:, 0],
             maxima[:, 1],
             "o")
    
    plt.savefig(output_dir + "/ssim_plot.png", dpi=300)
    plt.close()
    with open(output_dir + "/ssim_values.txt", "w") as fp:
        fp.writelines([str(n) + "\n" for n in ssim_series])

    print("Done video %s" % input_video)

def process_corpora(input_dir, output_dir):
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
                process_individual_video(input_video, new_dir)
