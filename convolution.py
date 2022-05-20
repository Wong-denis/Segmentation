import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolution(image, kernel, average=False, verbose=False):
    image_row, image_col , rgb = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width),rgb))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            for r in  range(rgb):
                output[row, col,r] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col,r])
                if average:
                    output[row, col,r] /= kernel.shape[0] * kernel.shape[1]
    real_output = np.zeros([image_row,image_col])


    print("Output Image size : {}".format(output.shape))
    # for row in range(image_row):
    #     for col in range(image_col):
    #         real_output[row,col] = sum(output[row,col])
    real_output = output.sum(-1)

    return real_output