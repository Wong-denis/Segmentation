import numpy as np


def convolution(image, kernel, average=False, verbose=False):
    image_row, image_col = image.shape
    kernel_shape = kernel.shape
    if len(kernel_shape) == 2:
        kernel_row, kernel_col = kernel.shape
    elif len(kernel_shape) == 1:
        print("kernel is 1d")
        kernel_col = kernel_shape[0]
        kernel_row = 1
    else:
        raise Exception("kernel too many dimension")

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):       
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))


    return output