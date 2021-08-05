import numpy as np
import cv2
import argparse
import os

def read_img(image_path):
    """
    Read image from the path
    and extract the image name from the path
    Input: image path

    Output: image, image name
    """

    image_name = os.path.split(image_path)[-1]

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, image_name


def calc_mean(img):
    """
    calculate the mean as given in equation 7 of the paper
    Input: image

    Output: mean
    """

    D_r = np.float64(img[:,:,0])
    D_g = np.float64(img[:,:,1])
    D_b = np.float64(img[:,:,2])

    H, W, _ = img.shape

    summation = (0.299 * D_r) + (0.587 * D_g) + (0.114 * D_b)

    area = W * H

    mean = summation / area

    return mean

def calc_alpha(p0, mean):
    """
    calculate the value of alpha as given in equation 9 of the paper
    Input: p0 (default to 1.6)
           mean 

    Output: alpha

    """
    p1 = -0.018

    alpha = p0 + (p1 * mean)

    return alpha


def enhancement(img, alpha):
    """
    Final enhancement as shown in equation 10 of the paper
    Input: image
           alpha

    Output: enhanced image in BGR format
    """

    D_r = img[:,:,0]
    D_g = img[:,:,1]
    D_b = img[:,:,2]

    E_r = alpha * (170.7 / (np.max(D_r) + 15.49))
    E_g = alpha * (179.3 / (np.max(D_g) + 15.42))
    E_b = alpha * (160.4 / (np.max(D_b) + 15.81))

    enhanced_image = np.zeros(img.shape)
    enhanced_image[:,:,2] = cv2.multiply(np.uint8(E_r), D_r)
    enhanced_image[:,:,1] = cv2.multiply(np.uint8(E_g), D_g)
    enhanced_image[:,:,0] = cv2.multiply(np.uint8(E_b),  D_b)

    return enhanced_image


def save_image(path, image_name, img):
    """
    saves the output image in the give folder
    Input: output path
           name of the input image
           image
    
    Output: None
    """
    out_path = os.path.join(path, image_name)
    cv2.imwrite(out_path, img)


def main(image_path, p0, out_path):
    image, image_name = read_img(image_path)

    mean = calc_mean(image)
    
    alpha = calc_alpha(p0, mean)
    
    enhanced_image = enhancement(image, alpha)

    save_image(out_path, image_name, enhanced_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Input Values")

    parser.add_argument("Image_path",
                        metavar='path',
                        type = str,
                        help="path of the input image")

    parser.add_argument("output_path",
                        metavar='out_path',
                        type = str,
                        help="folder path where to save the image")

    parser.add_argument("p0",
                        type = np.float64,
                        nargs = '?',
                        default=1.6,
                        help="p0 value, default 1.6")

    args = parser.parse_args()

    main(args.Image_path, args.p0, args.output_path)