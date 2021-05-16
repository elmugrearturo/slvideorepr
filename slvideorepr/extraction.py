import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import structural_similarity as ssim

def mse(first_img, second_img):
    first_img_float = first_img.astype("float32")
    second_img_float = second_img.astype("float32")
    total_pixels = first_img.shape[0] * second_img.shape[1]
    
    diff_img = (first_img_float - second_img_float) ** 2
    score = diff_img.sum() / total_pixels
    diff_img = np.sqrt(diff_img).astype("uint8")
    
    return score, diff_img
    
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
                process_individual_video(input_video, new_dir, measure)
