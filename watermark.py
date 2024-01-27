import copy

import cv2
import numpy as np
from numpy.linalg import svd
from pywt import dwt2, idwt2
from cv2 import dct, idct


def embed_watermark(image, watermark):
    """
    Embed a watermark into an image.

    :param image: The image to add the watermark to.
    :param watermark: The watermark to add.
    :return: The watermarked image.
    """
    # Initialization
    block_shape = (4, 4)
    password_wm = 1
    wm_bit = None
    wm_size = 0

    block_shape = np.array([4, 4])
    password_img = 1
    d1, d2 = 36, 20

    img, img_YUV = None, None
    ca, hvd, = [np.array([])] * 3, [np.array([])] * 3
    ca_block = [np.array([])] * 3
    ca_part = [np.array([])] * 3

    wm_size, block_num = 0, 0

    fast_mode = False
    alpha = None

    # Read image
    img = image
    alpha = None
    if img.shape[2] == 4:
        if img[:, :, 3].min() < 255:
            alpha = img[:, :, 3]
            img = img[:, :, :3]

    img = img.astype(np.float32)
    img_shape = img.shape[:2]

    img_YUV = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_BGR2YUV),
                                 0, img.shape[0] % 2, 0, img.shape[1] % 2,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

    ca_shape = [(i + 1) // 2 for i in img_shape]

    ca_block_shape = (ca_shape[0] // block_shape[0], ca_shape[1] // block_shape[1],
                      block_shape[0], block_shape[1])
    strides = 4 * np.array([ca_shape[1] * block_shape[0], block_shape[1], ca_shape[1], 1])

    for channel in range(3):
        ca[channel], hvd[channel] = dwt2(img_YUV[:, :, channel], 'haar')
        ca_block[channel] = np.lib.stride_tricks.as_strided(ca[channel].astype(np.float32),
                                                            ca_block_shape, strides)

    # Read watermark
    wm = watermark
    wm_bit = wm.flatten() > 128

    wm_size = wm_bit.size

    np.random.RandomState(password_wm).shuffle(wm_bit)

    # Embed watermark
    block_num = ca_block_shape[0] * ca_block_shape[1]
    assert wm_size < block_num, IndexError(
        'Available {}kb，Watermark{}kb'.format(block_num / 1000, wm_size / 1000))
    part_shape = ca_block_shape[:2] * block_shape
    block_index = [(i, j) for i in range(ca_block_shape[0]) for j in range(ca_block_shape[1])]

    embed_ca = copy.deepcopy(ca)
    embed_YUV = [np.array([])] * 3

    idx_shuffle = np.random.RandomState(password_img) \
        .random(size=(block_num, block_shape[0] * block_shape[1])) \
        .argsort(axis=1)

    for channel in range(3):
        tmp = []
        for i in range(block_num):
            block, shuffler, i = ca_block[channel][block_index[i]], idx_shuffle[i], i
            wm_1 = wm_bit[i % wm_size]
            block_dct = dct(block)

            block_dct_shuffled = block_dct.flatten()[shuffler].reshape(block_shape)
            u, s, v = svd(block_dct_shuffled)
            s[0] = (s[0] // d1 + 1 / 4 + 1 / 2 * wm_1) * d1
            if d2:
                s[1] = (s[1] // d2 + 1 / 4 + 1 / 2 * wm_1) * d2

            block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
            block_dct_flatten[shuffler] = block_dct_flatten.copy()
            result = idct(block_dct_flatten.reshape(block_shape))

            tmp.append(result)

        for i in range(block_num):
            ca_block[channel][block_index[i]] = tmp[i]

        ca_part[channel] = np.concatenate(np.concatenate(ca_block[channel], 1), 1)
        embed_ca[channel][:part_shape[0], :part_shape[1]] = ca_part[channel]
        embed_YUV[channel] = idwt2((embed_ca[channel], hvd[channel]), "haar")

    embed_img_YUV = np.stack(embed_YUV, axis=2)
    embed_img_YUV = embed_img_YUV[:img_shape[0], :img_shape[1]]
    embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
    embed_img = np.clip(embed_img, a_min=0, a_max=255)

    if alpha is not None:
        embed_img = cv2.merge([embed_img.astype(np.uint8), alpha])

    return embed_img
