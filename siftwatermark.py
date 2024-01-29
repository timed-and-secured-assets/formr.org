import cv2 as cv
import numpy as np
import pywt

# based on:
# https://link.springer.com/article/10.1007/s11042-022-12738-x
# https://www.mdpi.com/2227-7390/11/7/1730#B45-mathematics-11-01730
# https://ieeexplore.ieee.org/abstract/document/8513859

watermark = 'UniMuensterformr'

# concatenate the ascii-representation of the watermark
bin_watermark = ''.join(format(ord(x), 'b') for x in watermark)
print(bin_watermark)
print(len(bin_watermark))

def embedding(image):
    # cv.namedWindow('watermarked img', cv.WINDOW_NORMAL)
    # cv.namedWindow('original img', cv.WINDOW_NORMAL)
    # cv.namedWindow('compare img', cv.WINDOW_NORMAL)

    img = cv.imread(image)

    # TODO: cover different color spaces
    # convert image to YCrCb and extract the Y-Channel to operate on
    gray = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    ychannel, crchannel, cbchannel = cv.split(gray)
    ychannel_test = cv.extractChannel(gray, 0)

    # watermark matrix and intensity factor of embedding
    watermark_matrix = np.zeros((4, 4))

    # 0 0  0  0
    # 0 0  0  0     these coefficients seem to be highly stable to noises (paper 1)
    # 0 0  0 10
    # 0 0 10  0

    watermark_matrix[2][3] = 10
    watermark_matrix[3][2] = 10
    alpha = 0.5  # optimal value for alpha ?
    watermark_matrix = watermark_matrix * alpha

    # does the image have the right size?
    height = np.size(img, 0)
    width = np.size(img, 1)

    # TODO: assertions and exception handling
    if height < 150:
        print("Das Bild muss größer als 150x150 Pixel sein.")
        return
    if width < 150:
        print("Das Bild muss größer als 150x150 Pixel sein.")
        return

    # create a mask for sift algorithm
    mask = 255 * (np.ones((height, width), dtype=np.uint8))
    distanceToEdge = 35  # 32 ?

    # 0 0 0 0
    # 0 1 1 0
    # 0 0 0 0

    mask[: distanceToEdge, :] = 0
    mask[height - distanceToEdge:, :] = 0
    mask[distanceToEdge: height - distanceToEdge, : distanceToEdge] = 0
    mask[distanceToEdge: height - distanceToEdge, width - distanceToEdge: width] = 0

    # detect keypoints with sift
    sift = cv.SIFT.create()
    kp = sift.detect(ychannel, mask)

    # sort keypoints by response value in descending order
    kp_sorted = sorted(kp, key=lambda x: x.response, reverse=True)

    # create marker_matrix to detect non-overlapping regions of the strongest keypoints
    marker_matrix = np.zeros((height, width), dtype=np.uint8)

    # region size is 48x48, thus 24 in each direction around the keypoint
    # optimization: 74x74 with empty 26x26 around the keypoint to not distract the feature value calculation
    region_size = 24  # must be divisible by 4
    sub_size = 4  # size of the sub matrices 4x4
    number_of_regions = 5
    keypoints_final = []

    # check whether the watermark fits the region size
    if len(bin_watermark) >= (region_size / sub_size) * region_size:
        # error
        return

    i = 0
    j = 0
    # optimization 1: use structural similarity index (SSIM) between original region and embedded region
    # optimization 2: use formula in paper 1 and 3, greedy algorithm: max(sum(strongest non-overlapping keypoints))
    # fill marker_matrix with 1's in a 48x48 window around the strongest keypoints (hamming window)
    while i < number_of_regions:
        x = round(kp_sorted[j].pt[0])
        y = round(kp_sorted[j].pt[1])

        if np.max(marker_matrix[y - region_size: y + region_size, x - region_size: x + region_size]) == 0:
            marker_matrix[y - region_size: y + region_size, x - region_size: x + region_size] = 1
            keypoints_final.append(kp_sorted[j])
            i += 1
        j += 1

    # embedding the watermark!!

    k = region_size // 2
    # TODO: coordinates of the keypoints need to be stored too, keypoints_final ?
    svd_matrices = np.ndarray((len(keypoints_final), len(bin_watermark), 3, sub_size, sub_size))

    for i in range(len(keypoints_final)):
        x = int(round(keypoints_final[i].pt[0]))
        y = int(round(keypoints_final[i].pt[1]))

        for j in range(len(bin_watermark)):

            # optimization with two additional for loops maybe or slicing the array more efficient
            if (k + j) % k != 0:
                row = y - region_size + (sub_size * (j // k))
                col = x - region_size + ((j % k) * sub_size)

                # perform singular value decomposition (SVD) on 4x4 matrices in the feature region
                # U, S, Vh = cv.SVDecomp(ychannel_float[row: row + sub_size, col: col + sub_size], flags=cv.SVD_FULL_UV)
                U, S, Vh = np.linalg.svd(ychannel[row: row + sub_size, col: col + sub_size], full_matrices=False)

            else:
                row = y - region_size + (sub_size * j // k)
                col = x - region_size

                # perform singular value decomposition (SVD) on 4x4 matrices in the feature region
                # U, S, Vh = cv.SVDecomp(ychannel_float[row: row + sub_size, col: col + sub_size], flags=cv.SVD_FULL_UV)
                U, S, Vh = np.linalg.svd(ychannel[row: row + sub_size, col: col + sub_size], full_matrices=False)

            # TODO: maybe use DWT-SVD or DCT-SVD instead of SVD
            # embeds the watermark bits in the singular value matrix
            if bin_watermark[j] == 0:
                t_emb = np.diag(S) - watermark_matrix
            else:
                t_emb = np.diag(S) + watermark_matrix

            # perform SVD on the watermarked singular value matrix
            # Uw, Sw, Vwh = cv.SVDecomp(t_emb, flags=cv.SVD_FULL_UV)
            Uw, Sw, Vwh = np.linalg.svd(t_emb, full_matrices=False)

            # SVD matrices needed for extraction of the watermark
            svd_matrices[i][j][0] = Uw
            svd_matrices[i][j][1] = S
            svd_matrices[i][j][2] = Vwh

            # embeds the watermark in the ychannel
            ychannel[row: row + sub_size, col: col + sub_size] = U @ np.diag(Sw) @ Vh

    # merge the watermarked channel with the others and reconstruct the image
    watermark_ycrcb = cv.merge((ychannel, crchannel, cbchannel))
    watermark_img = cv.cvtColor(watermark_ycrcb, cv.COLOR_YCrCb2BGR)

    cv.imwrite('watermarked.jpg', watermark_img)
    cv.imshow('watermarked img', watermark_img)

    compare_img = cv.drawKeypoints(img, keypoints_final, 0, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('compare img', compare_img)
    cv.imshow('original img', img)

    cv.waitKey(0)

# TODO: optimizations, exception handling, watermark extraction
# if __name__ == '__main__':
    # embedding('Img/james.jpg')
