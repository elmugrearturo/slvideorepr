import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def show_image(window_name, img):
    if not window_name:
        window_name = "%dx%d" % img.shape[:2]

    assert isinstance(window_name, str)

    cv2.imshow(window_name, img)

    while cv2.getWindowProperty(window_name,
                                cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKeyEx(1000) == 27:
            cv2.destroyWindow(window_name)
            break

def mse(first_img, second_img):
    first_img_float = first_img.astype("float32")
    second_img_float = second_img.astype("float32")
    total_pixels = first_img.shape[0] * second_img.shape[1]
    
    diff_img = (first_img_float - second_img_float) ** 2
    score = diff_img.sum() / total_pixels
    diff_img = np.sqrt(diff_img).astype("uint8")
    
    return score, diff_img

def mse_norm(first_img, second_img):
    first_img_float = first_img.astype("float32") / 255
    second_img_float = second_img.astype("float32") / 255
    total_pixels = first_img.shape[0] * second_img.shape[1]
    
    diff_img = (first_img_float - second_img_float) ** 2
    if diff_img.sum() > 0:
        import ipdb;ipdb.set_trace()
    score = diff_img.sum() / total_pixels
    diff_img = np.sqrt(diff_img) * 255
    diff_img = diff_img.astype("uint8")
    return score, diff_img

def single_image_gradient_score(gray_image, window_size):
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=window_size)
    sobelx = np.absolute(sobelx)
    
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=window_size)
    sobely = np.absolute(sobely)

    total = gray_image.shape[0] * gray_image.shape[1]

    sobel_xy = (sobelx + sobely) / 2
    #if sobel_xy.max() != 0:
    #    sobel_xy /= sobel_xy.max()
    return np.mean(sobel_xy)

def mse_window(image_array):
    first_img_float = image_array[0].astype("float32")
    total_pixels = first_img_float.shape[0] * first_img_float.shape[1]
    all_scores = []
    for second_image in image_array[1:] :

        second_img_float = second_image.astype("float32")

        diff_img = (first_img_float - second_img_float) ** 2
        score = diff_img.sum() / total_pixels
        # diff_img = np.sqrt(diff_img).astype("uint8")

        all_scores.append(score)
    final_score = np.mean(all_scores)
    return final_score

def image_crop(img, k, show=False):
    if show:
        show_image("Original", img)
   
    y_residual = img.shape[0] % k
    x_residual = img.shape[1] % k

    patches = []
    coords = []
    
    y = 0
    while (y+k) < img.shape[0]:
        x = 0
        while (x+k) < img.shape[1]:
            patches.append(img[y:y+k, x:x+k])
            coords.append((y, y+k, x, x+k))
            x = x+k
        if x_residual != 0:
            patches.append(img[y:y+k, x:x+x_residual])
            coords.append((y, y+k, x, x+x_residual))
        y = y+k
    
    if y_residual != 0:
        x = 0
        while (x+k) < img.shape[1]:
            patches.append(img[y:y+y_residual, x:x+k])
            coords.append((y, y+y_residual, x, x+k))
            x = x+k
        if x_residual != 0:
            patches.append(img[y:y+y_residual, x:x+x_residual])
            coords.append((y, y+y_residual, x, x+x_residual))

    if show:
        for i, p in enumerate(patches):
            show_image("%s" % str(coords[i]), p)

    return patches, coords

def nonlinear_comparison(f_img, s_img, k):
    if f_img.shape[0] == s_img.shape[0]:
        if f_img.shape[1] == s_img.shape[1]:
            pass
        else:
            raise ValueError("Incompatible dimensions")
    else:    
        raise ValueError("Incompatible dimensions")
    f_patches, f_coords = image_crop(f_img, k)
    s_patches, s_coords = image_crop(s_img, k)

    scores = []
    diff_imgs = []
    for i in range(len(f_patches)):
        f_patch = f_patches[i]
        s_patch = s_patches[i]
        #MSE
        score, diff_img = mse(f_patch, s_patch)
        # DSSIM
        #try:
        #    score, diff_img = ssim(f_patch,
        #        s_patch, 
        #        gaussian_weights=True, full=True)
        #except:
        #    score = 1.0
        # Gradient
        #score = single_image_gradient_score(s_patch, 3)
        
        #score = 1 - score # DSSIM_modif; invverse acutance

        scores.append(score)
        diff_imgs.append(diff_img)

        #show_image("%f: %s" % (score, str(f_coords[i])), diff_img)
    if len(scores) == 0:
        import ipdb;ipdb.set_trace()
    return np.max(scores)
