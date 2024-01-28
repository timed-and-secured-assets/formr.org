import copy

import numpy as np
import cv2
import pywt


def embed_watermark(image, watermark):
    """
    Embed a watermark into an image.

    :param image: The image to add the watermark to.
    :param watermark: The grayscale watermark to add.
    :return: The watermarked image.
    """

    """
    Set the parameters for the watermark embedding.
    Note that these parameters must be identical when extracting the watermark.
    """
    # Parameters for the embedding strength (larger parameters are more stable but the output is more distorted)
    d1 = 36
    d2 = 20

    # Shape of the blocks used in the embedding (dimensions must be equal and even)
    block_shape = np.array([4, 4])

    # Seed to use for the RNG (used to shuffle watermark data and embedding indices)
    rng_seed = 1

    """
    Read the image to add the watermark to.
    """
    # Extract the alpha channel (transparency) from the image if it is present
    alpha_channel = None
    if image.shape[2] == 4 and image[:, :, 3].min() < 255:
        alpha_channel = image[:, :, 3]
        image = image[:, :, :3]

    # Convert the image to float and YUV color space
    image = image.astype(np.float32)
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Add a padding to the image to make the dimensions even (needed for DWT)
    image_yuv = cv2.copyMakeBorder(image_yuv, 0, image.shape[0] % 2, 0, image.shape[1] % 2,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))

    """
    Initialize data structures for the Discrete Wavelet Transform.
    """
    # Approximation coefficients array (channel-wise)
    # These coefficients represent the low-frequency components of the image and are used to embed the watermark into
    ca = [np.array([])] * 3

    # Block-wise approximation coefficients array (channel-wise)
    ca_block = [np.array([])] * 3
    ca_part = [np.array([])] * 3

    # Calculate the shapes of the ca and block-wise ca array
    ca_shape = [(i + 1) // 2 for i in image.shape[:2]]
    ca_block_shape = (ca_shape[0] // block_shape[0], ca_shape[1] // block_shape[1], block_shape[0], block_shape[1])

    # Calculate strides that are needed to transform the ca array into the block-wise ca array
    strides = 4 * np.array([ca_shape[1] * block_shape[0], block_shape[1], ca_shape[1], 1])

    # Horizontal, vertical, and diagonal coefficients array (channel-wise)
    # These coefficients represent the high-frequency components of the image and will not be altered
    hvd = [np.array([])] * 3

    """
    Perform channel-wise Discrete Wavelet Transform on the YUV image.
    """
    for channel in range(3):
        ca[channel], hvd[channel] = pywt.dwt2(image_yuv[:, :, channel], 'haar')
        # Transform the ca array into the block-wise ca array
        # TODO: Think of a safer way to do this (see notes to np.lib.stride_tricks.as_strided)
        ca_block[channel] = np.lib.stride_tricks.as_strided(ca[channel].astype(np.float32), ca_block_shape, strides)

    """
    Set up the watermark and assert it can be embedded into the image. 
    """
    # Convert the grayscale watermark image into an array of bits (booleans), shuffle it and store the size
    watermark_bits = watermark.flatten() > 128
    np.random.RandomState(rng_seed).shuffle(watermark_bits)
    watermark_size = watermark_bits.size

    # Calculate the available block size (number of blocks in the image / bits to embed in the image)
    block_size = ca_block_shape[0] * ca_block_shape[1]

    # Assert that the available block size is larger than the watermark size
    assert block_size > watermark_size, \
        f"Not enough space available in the image ({block_size} bits) to embed the watermark ({watermark_size} bits)"

    """
    Initialize data structures for the embedding process. 
    """
    # Calculate the shape of the part of the image that contains the ca blocks and the indices of the blocks
    part_shape = ca_block_shape[:2] * block_shape
    ca_block_index = [(i, j) for i in range(ca_block_shape[0]) for j in range(ca_block_shape[1])]

    # Arrays to store the embedded approximation coefficients and YUV values
    embedded_ca = copy.deepcopy(ca)
    embedded_yuv = [np.array([])] * 3

    # Randomized indices used to shuffle the positioning inside the blocks
    shuffle_indices = np.random.RandomState(rng_seed).random(size=(block_size, block_shape[0] * block_shape[1])) \
        .argsort(axis=1)

    """
    Embed the watermark bits into the image for each channel.
    """
    for channel in range(3):
        # Iterate through every block in the image
        for i in range(block_size):
            # Get the current ca block and perform a Discrete Cosine Transform on it
            block = ca_block[channel][ca_block_index[i]]
            block_dct = cv2.dct(block)

            # Get the shuffled indices for the current block and use them to shuffle the DCT coefficients block
            shuffler = shuffle_indices[i]
            block_dct_shuffled = block_dct.flatten()[shuffler].reshape(block_shape)

            # Get the current bit of the watermark bits (use modulo to repeat the bits until all blocks are filled)
            watermark_bit = watermark_bits[i % watermark_size]

            # Perform a Singular Value Decomposition on the shuffled DCT coefficients block
            # Embed the watermark bit into the first two singular values
            u, s, v = np.linalg.svd(block_dct_shuffled)
            s[0] = (s[0] // d1 + 1 / 4 + 1 / 2 * watermark_bit) * d1
            if d2:
                s[1] = (s[1] // d2 + 1 / 4 + 1 / 2 * watermark_bit) * d2

            # Reversal of the Singular Value Decomposition with the modified singular values
            block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()

            # Reverse the shuffle of the DCT coefficients block
            block_dct_flatten[shuffler] = block_dct_flatten.copy()

            # Perform an inverse Discrete Cosine Transform on the modified coefficients block
            ca_block[channel][ca_block_index[i]] = cv2.idct(block_dct_flatten.reshape(block_shape))

        # Reverse the block-wise transform for the ca
        ca_part[channel] = np.concatenate(np.concatenate(ca_block[channel], 1), 1)

        # Swap the part of the image that contains the ca blocks with the modified ca blocks
        embedded_ca[channel][:part_shape[0], :part_shape[1]] = ca_part[channel]

        # Perform an inverse Discrete Wavelet Transform to reconstruct the image with the modified coefficients
        embedded_yuv[channel] = pywt.idwt2((embedded_ca[channel], hvd[channel]), "haar")

    """
    Reconstruct and return the embedded image.
    """
    # Construct a three-channel image and restore the original shape of the image without the possibly added padding
    embedded_image_yuv = np.stack(embedded_yuv, axis=2)
    embedded_image_yuv = embedded_image_yuv[:image.shape[0], :image.shape[1]]

    # Convert the image back to the standard BGR color space and clip values that are out of range for valid colors
    embedded_image = cv2.cvtColor(embedded_image_yuv, cv2.COLOR_YUV2BGR)
    embedded_image = np.clip(embedded_image, a_min=0, a_max=255)

    # Add the extracted alpha channel (transparency) back if it was present
    if alpha_channel is not None:
        embedded_image = cv2.merge([embedded_image.astype(np.uint8), alpha_channel])

    return embedded_image


def extract_watermark(image, watermark_shape):
    """
    Extract a watermark from an image.

    :param image: The image to extract the watermark from.
    :param watermark_shape: The shape of the watermark.
    :return: The extracted watermark.
    """
    # Initialization
    password_wm = 1

    block_shape = np.array([4, 4])
    password_img = 1
    d1 = 36
    d2 = 20

    ca = [np.array([])] * 3
    hvd = [np.array([])] * 3
    ca_block = [np.array([])] * 3

    wm_shape = watermark_shape

    # Extract watermark
    wm_size = np.array(wm_shape).prod()

    if image.shape[2] == 4:
        if image[:, :, 3].min() < 255:
            image = image[:, :, :3]

    image = image.astype(np.float32)
    img_shape = image.shape[:2]

    img_yuv = cv2.copyMakeBorder(cv2.cvtColor(image, cv2.COLOR_BGR2YUV), 0, image.shape[0] % 2, 0,
                                 image.shape[1] % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    ca_shape = [(i + 1) // 2 for i in img_shape]

    ca_block_shape = (ca_shape[0] // block_shape[0], ca_shape[1] // block_shape[1], block_shape[0], block_shape[1])
    strides = 4 * np.array([ca_shape[1] * block_shape[0], block_shape[1], ca_shape[1], 1])

    for channel in range(3):
        ca[channel], hvd[channel] = pywt.dwt2(img_yuv[:, :, channel], 'haar')
        ca_block[channel] = np.lib.stride_tricks.as_strided(ca[channel].astype(np.float32), ca_block_shape, strides)

    block_num = ca_block_shape[0] * ca_block_shape[1]
    assert wm_size < block_num, IndexError(
        'Available {}kb，Watermark{}kb'.format(block_num / 1000, wm_size / 1000))
    block_index = [(i, j) for i in range(ca_block_shape[0]) for j in range(ca_block_shape[1])]

    wm_block_bit = np.zeros(shape=(3, block_num))

    idx_shuffle = np.random.RandomState(password_img).random(size=(block_num, block_shape[0] * block_shape[1])) \
        .argsort(axis=1)

    for channel in range(3):
        results = []
        for i in range(block_num):
            block = ca_block[channel][block_index[i]]
            shuffler = idx_shuffle[i]
            block_dct_shuffled = cv2.dct(block).flatten()[shuffler].reshape(block_shape)

            u, s, v = np.linalg.svd(block_dct_shuffled)
            wm = (s[0] % d1 > d1 / 2) * 1
            if d2:
                tmp = (s[1] % d2 > d2 / 2) * 1
                wm = (wm * 3 + tmp * 1) / 4
            result = wm
            results.append(result)
        wm_block_bit[channel, :] = results

    wm_avg = np.zeros(shape=wm_size)
    for i in range(wm_size):
        wm_avg[i] = wm_block_bit[:, i::wm_size].mean()

    wm_index = np.arange(wm_size)
    np.random.RandomState(password_wm).shuffle(wm_index)
    wm_avg[wm_index] = wm_avg.copy()
    wm = wm_avg

    wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])

    return wm
