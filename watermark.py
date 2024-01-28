import copy

import numpy as np
import cv2
import pywt


def embed_watermark(image, watermark):
    """
    Embed a watermark into an image.

    :param image: The image to add the watermark to.
    :param watermark: The watermark to add.
    :return: The watermarked image.
    """
    # Initialization
    password_wm = 1

    block_shape = np.array([4, 4])
    password_img = 1
    d1 = 36
    d2 = 36

    ca = [np.array([])] * 3
    hvd = [np.array([])] * 3
    ca_block = [np.array([])] * 3
    ca_part = [np.array([])] * 3

    # Read image
    alpha = None
    if image.shape[2] == 4:
        if image[:, :, 3].min() < 255:
            alpha = image[:, :, 3]
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

    # Read watermark
    wm_bit = watermark.flatten() > 128

    wm_size = wm_bit.size

    np.random.RandomState(password_wm).shuffle(wm_bit)

    # Embed watermark
    block_num = ca_block_shape[0] * ca_block_shape[1]
    assert wm_size < block_num, IndexError(
        'Available {}kb，Watermark{}kb'.format(block_num / 1000, wm_size / 1000))
    part_shape = ca_block_shape[:2] * block_shape
    block_index = [(i, j) for i in range(ca_block_shape[0]) for j in range(ca_block_shape[1])]

    embed_ca = copy.deepcopy(ca)
    embed_yuv = [np.array([])] * 3

    idx_shuffle = np.random.RandomState(password_img).random(size=(block_num, block_shape[0] * block_shape[1])) \
        .argsort(axis=1)

    for channel in range(3):
        tmp = []
        for i in range(block_num):
            block = ca_block[channel][block_index[i]]
            shuffler = idx_shuffle[i]
            wm_1 = wm_bit[i % wm_size]
            block_dct = cv2.dct(block)

            block_dct_shuffled = block_dct.flatten()[shuffler].reshape(block_shape)
            u, s, v = np.linalg.svd(block_dct_shuffled)
            s[0] = (s[0] // d1 + 1 / 4 + 1 / 2 * wm_1) * d1
            if d2:
                s[1] = (s[1] // d2 + 1 / 4 + 1 / 2 * wm_1) * d2

            block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
            block_dct_flatten[shuffler] = block_dct_flatten.copy()
            result = cv2.idct(block_dct_flatten.reshape(block_shape))

            tmp.append(result)

        for i in range(block_num):
            ca_block[channel][block_index[i]] = tmp[i]

        ca_part[channel] = np.concatenate(np.concatenate(ca_block[channel], 1), 1)
        embed_ca[channel][:part_shape[0], :part_shape[1]] = ca_part[channel]
        embed_yuv[channel] = pywt.idwt2((embed_ca[channel], hvd[channel]), "haar")

    embed_img_yuv = np.stack(embed_yuv, axis=2)
    embed_img_yuv = embed_img_yuv[:img_shape[0], :img_shape[1]]
    embed_img = cv2.cvtColor(embed_img_yuv, cv2.COLOR_YUV2BGR)
    embed_img = np.clip(embed_img, a_min=0, a_max=255)

    if alpha is not None:
        embed_img = cv2.merge([embed_img.astype(np.uint8), alpha])

    return embed_img


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
    d2 = 36

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
