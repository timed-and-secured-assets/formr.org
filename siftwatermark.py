import cv2 as cv
import numpy as np
import pywt

# based on:
# https://link.springer.com/article/10.1007/s11042-022-12738-x
# https://www.mdpi.com/2227-7390/11/7/1730#B45-mathematics-11-01730
# https://ieeexplore.ieee.org/abstract/document/8513859
# projective transformation: https://www.youtube.com/watch?v=2BIzmFD_pRQ

# Diameter of the embedding regions.
region_diameter = 64


def embedding(image, watermark, alpha):
    """
    Embeds a watermark in the given image.
    :param image: The image to be watermarked.
    :param watermark: A string of 1 to 20 common chars (.,/abc123 etc.), e.g. 'UniMuensterformr'.
    :return: The watermarked image and an array containing specific information needed for the extraction.

    Note that the maximum capacity of the watermark is limited to 144 bits. The string will be encoded using ASCII,
    which means that 'uni-muenster.de/formr' with a total of 21 chars is also valid.
    """
    # Get the ascii values of the watermark chars
    ascii_values = watermark.encode('ascii')
    # concatenate the ascii-representation of the watermark chars
    bin_watermark = ''.join(f'{x:07b}' for x in ascii_values)
    print(bin_watermark)

    if len(bin_watermark) > pow(region_diameter // 4, 2):
        raise ValueError('Der Wasserzeichentext darf maximal 36 Zeichen lang sein.')

    img = cv.imread(image)

    # TODO: cover different color spaces
    # convert image to YCrCb and extract the Y-Channel to operate on
    gray = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    # gray = gray.astype(np.float32)
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

    watermark_matrix = watermark_matrix * alpha

    # does the image have the right size?
    height = np.size(img, 0)
    width = np.size(img, 1)

    # TODO: assertions and exception handling
    if height < 150:
        raise ValueError('Das Bild muss größer als 150x150 Pixel sein.')
    if width < 150:
        raise ValueError('Das Bild muss größer als 150x150 Pixel sein.')

    # create a mask for sift algorithm
    mask = 255 * (np.ones((height, width), dtype=np.uint8))
    distanceToEdge = (region_diameter // 2) + 5

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

    # Create Wavelet object (Daubechies Wavelet)
    wavelet = pywt.Wavelet('haar')

    coeffs = pywt.dwt2(ychannel, wavelet)
    cA, (cH, cV, cD) = coeffs
    cv.imwrite('Img/DwtTestcA.jpg', cA)
    cv.imwrite('Img/DwtTestcD.jpg', cD)
    cv.imwrite('Img/DwtTestcH.jpg', cH)
    cv.imwrite('Img/DwtTestcV.jpg', cV)
    print('Mean of cA: ', np.mean(cA))
    print('Min of cA: ', np.min(cA))
    print('Mean of cD: ', np.mean(cD))
    print('Min of cD: ', np.min(cD))
    print('Max of cD: ', np.max(cD))


    # sort keypoints by response value in descending order
    kp_sorted = sorted(kp, key=lambda x: x.response, reverse=True)

    # create marker_matrix to detect non-overlapping regions of the strongest keypoints
    marker_matrix = np.zeros((height, width), dtype=np.uint8)

    # region size is 48x48, thus 24 in each direction around the keypoint
    # optimization: 74x74 with empty 26x26 around the keypoint to not distract the feature value calculation
    region_size = region_diameter // 2  # must be divisible by 4
    sub_size = 4  # size of the sub matrices 4x4
    number_of_regions = 5
    keypoints_final = []

    # check whether the watermark fits the region size
    if len(bin_watermark) > (region_size / sub_size) * region_size:
        raise ValueError('')

    i = 0
    j = 0
    # optimization 1: use structural similarity index (SSIM) between original region and embedded region
    # optimization 2: use formula in paper 1 and 3, greedy algorithm: max(sum(strongest non-overlapping keypoints))
    # fill marker_matrix with 1's in a 64x64 window around the strongest keypoints (hamming window)
    while i < number_of_regions and j < len(kp_sorted):
        x = int(round(kp_sorted[j].pt[0]))
        y = int(round(kp_sorted[j].pt[1]))

        if np.max(marker_matrix[y - region_size: y + region_size, x - region_size: x + region_size]) == 0:
            marker_matrix[y - region_size: y + region_size, x - region_size: x + region_size] = 1
            keypoints_final.append(kp_sorted[j])
            i += 1
        j += 1

    # embedding the watermark!!

    k = region_size // 2
    # TODO: coordinates of the keypoints need to be stored too, keypoints_final ?
    svd_matrices = np.ndarray((len(keypoints_final), len(bin_watermark) + 1, 3, sub_size, sub_size))

    for i in range(len(keypoints_final)):
        x = int(round(keypoints_final[i].pt[0]))
        y = int(round(keypoints_final[i].pt[1]))

        # testing
        print('x = ', x)
        print('y = ', y)
        print('Mittelwert ychannel: ', np.mean(ychannel[y - region_size: y + region_size, x - region_size: x + region_size]))

        coeffs2 = pywt.dwt2(ychannel[y - region_size: y + region_size, x - region_size: x + region_size], wavelet)
        cAA, (cHH, cVV, cDD) = coeffs2
        print('Summe LL: ', np.sum(np.abs(cAA)))
        # print('Mean of LL: ', np.mean(cAA))
        print('Summe HH: ', np.sum(np.abs(cDD)))
        # print('Mean of HH: ', np.mean(cDD))

        mean_ychannel = np.mean(ychannel[y - region_size: y + region_size, x - region_size: x + region_size])
        sum_LL = np.sum(np.abs(cAA))
        sum_HH = np.sum(np.abs(cDD))
        alpha = 0.01
        if sum_HH < 1200:
            alpha = 0.008
        if 1200 < sum_HH < 2400:
            alpha = 0.01
            if sum_LL > 300000 or mean_ychannel > 130:
                alpha = 0.008
            if mean_ychannel < 80:
                alpha = 0.015
        if 2400 < sum_HH < 5000:
            alpha = 0.015
            if sum_LL > 110:
                alpha = 0.01
            if mean_ychannel < 80:
                alpha = 0.02
        if 5000 < sum_HH < 10000:
            alpha = 0.02
        if sum_HH > 10000:
            alpha = 0.03

        # print('Mittelwert FFT: ', np.sum(np.power(np.fft.rfft2(ychannel[y - region_size: y + region_size, x - region_size: x + region_size]), 2)))

        for j in range(len(bin_watermark)):
            # optimization with two additional for loops maybe or slicing the array more efficient
            # 4x4 matrix is NOT in the left column of the square around the keypoint
            if (k + j) % k != 0:
                row = y - region_size + (sub_size * (j // k))
                col = x - region_size + ((j % k) * sub_size)
            # 4x4 matrix is in the left column of the square around the keypoint
            else:
                row = y - region_size + (sub_size * j // k)
                col = x - region_size

            # DWT testing
            '''coeffs = pywt.dwt2(ychannel[row: row + sub_size, col: col + sub_size], wavelet)
            cA, (cH, cV, cD) = coeffs
            print('Shape of cA: ', np.shape(cA))
            print('Shape of cH: ', np.shape(cH))
            print('Shape of cV: ', np.shape(cV))
            print('Shape of cD: ', np.shape(cD))'''


            # perform singular value decomposition (SVD) on 4x4 matrices in the feature region
            # U, S, Vh = cv.SVDecomp(ychannel_float[row: row + sub_size, col: col + sub_size], flags=cv.SVD_FULL_UV)
            U, S, Vh = np.linalg.svd(ychannel[row: row + sub_size, col: col + sub_size], full_matrices=False)
            # U, S, Vh = np.linalg.svd(cA, full_matrices=False)
            # print('Shape of U: ', np.shape(Vh))
            # TODO: maybe use DWT-SVD or DCT-SVD instead of SVD
            # embeds the watermark bits in the singular value matrix
            if bin_watermark[j] == '0':
                U[1][0] = max(U[1][0], U[2][0]) + alpha
                U[2][0] = min(U[1][0], U[2][0]) - alpha
                t_emb = np.diag(S) - watermark_matrix
            else:
                U[1][0] = min(U[1][0], U[2][0]) - alpha
                U[2][0] = max(U[1][0], U[2][0]) + alpha
                t_emb = np.diag(S) + watermark_matrix

            # perform SVD on the watermarked singular value matrix
            # Uw, Sw, Vwh = cv.SVDecomp(t_emb, flags=cv.SVD_FULL_UV)
            Uw, Sw, Vwh = np.linalg.svd(t_emb, full_matrices=False)

            # SVD matrices needed for extraction of the watermark
            # svd_matrices[i][j][0] = Uw
            # svd_matrices[i][j][1] = np.diag(S)
            # svd_matrices[i][j][2] = Vwh

            # this matrix will replace its corresponding part in the ychannel
            watermarked_square = U @ np.diag(S) @ Vh
            np.clip(watermarked_square, 0, 255, watermarked_square)

            # DWT testing
            # ychannel[row: row + sub_size, col: col + sub_size] = pywt.idwt2((watermarked_square, (cH, cV, cD)), wavelet)

            # embeds the watermark in the ychannel
            ychannel[row: row + sub_size, col: col + sub_size] = watermarked_square

    # Store the coordinates of the keypoints used for embedding
    for k in range(len(keypoints_final)):
        svd_matrices[k][len(bin_watermark)][0][0][0] = keypoints_final[k].pt[0]
        svd_matrices[k][len(bin_watermark)][0][0][1] = keypoints_final[k].pt[1]

    # Store the image's width in svd_matrices
    svd_matrices[0][len(bin_watermark)][0][1][0] = np.shape(img)[1]
    # Store the image's height in svd_matrices
    svd_matrices[0][len(bin_watermark)][0][1][1] = np.shape(img)[0]

    # testing
    kpwm = sift.detect(ychannel, mask)
    kp_sortedwm = sorted(kpwm, key=lambda x: x.response, reverse=True)

    for p in range(len(keypoints_final)):
        print('Original x: ')
        print(keypoints_final[p].pt[0])
        print('Original y:')
        print(keypoints_final[p].pt[1])

    # merge the watermarked channel with the others and reconstruct the image
    watermark_ycrcb = cv.merge((ychannel, crchannel, cbchannel))
    watermark_img = cv.cvtColor(watermark_ycrcb, cv.COLOR_YCrCb2BGR)

    cv.imwrite('Img/watermarked.jpg', watermark_img)

    compare_img = cv.drawKeypoints(img, keypoints_final, 0, color=500, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('compare_img.jpg', compare_img)

    # cv.waitKey(0)

    return watermark_img, svd_matrices


def projection_transformation(dist_img, v, vo):
    """
    Projective Transformation is needed if the watermarked image is e.g. not downloaded but captured with a camera from
    a different angle. It transforms the watermarked image to its original shape.
    :param dist_img: The distorted image showing the watermarked image from a different angle.
    :param v: A 4x2 matrix containing the vertex x- and y-coordinates of the watermarked image inside the dist_img.
    :param vo: A 4x2 matrix containing the vertex x- and y-coordinates of the original watermarked image.
    :return: The distorted watermarked image but with its original shape.

    Note that v and vo have to follow the same order of vertices:

    (x0,y0)         ...         (x1,y1)
        .                           .
        .                           .
    (x3,y3)         ...         (x2,y2)

    """
    # First the unknown coefficients for the projection transformation matrix have to be determined (a)
    pj_tr = np.array([[vo[0][0], vo[0][1], 1, 0, 0, 0, -vo[0][0] * v[0][0], -vo[0][1] * v[0][0]],
                      [0, 0, 0, vo[0][0], vo[0][1], 1, -vo[0][0] * v[0][1], -vo[0][1] * v[0][1]],
                      [vo[1][0], vo[1][1], 1, 0, 0, 0, -vo[1][0] * v[1][0], -vo[1][1] * v[1][0]],
                      [0, 0, 0, vo[1][0], vo[1][1], 1, -vo[1][0] * v[1][1], -vo[1][1] * v[1][1]],
                      [vo[2][0], vo[2][1], 1, 0, 0, 0, -vo[2][0] * v[2][0], -vo[2][1] * v[2][0]],
                      [0, 0, 0, vo[2][0], vo[2][1], 1, -vo[2][0] * v[2][1], -vo[2][1] * v[2][1]],
                      [vo[3][0], vo[3][1], 1, 0, 0, 0, -vo[3][0] * v[3][0], -vo[3][1] * v[3][0]],
                      [0, 0, 0, vo[3][0], vo[3][1], 1, -vo[3][0] * v[3][1], -vo[3][1] * v[3][1]]])
    # Ordinate values (b)
    val = np.array([v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1], v[3][0], v[3][1]])

    # Solve the following equation: ax = b
    solution = np.linalg.solve(pj_tr, val)
    # Reshape the output array to get the 3x3 transformation matrix
    solution = np.append(solution, 1).reshape((3, 3))

    img = cv.imread(dist_img)

    # Perform projection transformation on the distorted image
    transformed_img = cv.warpPerspective(img, solution, (vo[2][0], vo[2][1]), 0, flags=cv.WARP_INVERSE_MAP)
    cv.imwrite('Img/projection.jpg', transformed_img)

    return transformed_img


def extraction(image, svd_matrices, distorted=False, coords=None):
    """
    Extracts the watermark
    :param image:
    :param distorted:
    :param coords:
    :return:
    """
    # Optimization: use Wiener filtering method to denoise the image
    img = cv.imread(image)

    # Length of the watermark.
    len_wm = np.shape(svd_matrices)[1] - 1

    # Get the width and height of the original watermarked image.
    width = int(svd_matrices[0][len_wm][0][1][0])
    height = int(svd_matrices[0][len_wm][0][1][1])

    # Number of used keypoints
    kp_numb = np.shape(svd_matrices)[0]

    # Number of extracting regions is doubled to enhance error-tolerant rate
    ext_reg_numb = 2 * kp_numb

    # Get the coordinates of the original keypoints
    orig_coords = []
    for i in range(kp_numb):
        c = [svd_matrices[i][len_wm][0][0][0], svd_matrices[i][len_wm][0][0][1]]
        orig_coords.append(c)

    # If the watermarked image is distorted, perform projection_transformation first
    if distorted:
        # Reconstruct the original vertex coordinates
        original_vertices = np.array([[0, 0],
                                      [width, 0],
                                      [width, height],
                                      [0, height]])
        # Projection Transformation provides the geometric corrected image
        img = projection_transformation(image, coords, original_vertices)

        ext_reg_numb += 5

    # Make sure the image to operate on has the original sizes
    if np.shape(img)[0] != height:
        raise ValueError(
            'Etwas ist schiefgegangen: Die Bildgröße stimmt nicht mit der des ursprünglichen Bildes überein.')
    if np.shape(img)[1] != width:
        raise ValueError(
            'Etwas ist schiefgegangen: Die Bildgröße stimmt nicht mit der des ursprünglichen Bildes überein.')

    # convert image to YCrCb and extract the Y-Channel to operate on
    gray = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    # gray = gray.astype(np.float32)
    ychannel, crchannel, cbchannel = cv.split(gray)

    # create a mask for sift algorithm
    mask = 255 * (np.ones((height, width), dtype=np.uint8))
    distanceToEdge = (region_diameter // 2) + 1

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
    kp_sorted = sorted(kp, key=lambda z: z.response, reverse=True)

    if len(kp_sorted) < ext_reg_numb:
        raise ValueError('Es wurden zu wenig Keypoints gefunden. Extrahierung nicht möglich.')

    restart = True
    while restart:
        restart = False
        for i in range(ext_reg_numb):
            for j in range(i + 1, ext_reg_numb):
                a = kp_sorted[i].pt[0]
                b = kp_sorted[j].pt[0]
                if abs(a - b) <= max(1e-06 * max(abs(a), abs(b)), 0.0):
                    kp_sorted.pop(j)
                    restart = True
                    break

    keypoints_final = []
    dist_coords = []

    for i in range(kp_numb):
        orig_x = svd_matrices[i][len_wm][0][0][0]
        orig_y = svd_matrices[i][len_wm][0][0][1]

        for j in range(ext_reg_numb):
            dist_x = kp_sorted[j].pt[0]
            dist_y = kp_sorted[j].pt[1]

            if abs(orig_x - dist_x) <= 0.1 and abs(orig_y - dist_y) <= 0.1:
                keypoints_final.append(kp_sorted[j])

                dc = [dist_x, dist_y]
                dist_coords.append(dc)
                break
        else:
            dist_coords.append([orig_x, orig_y])

    possible_watermarks = []
    k = region_diameter // 4
    region_size = region_diameter // 2
    sub_size = 4
    possible_wm = ''
    print('len distcoords')
    print(len(dist_coords))
    for i in range(len(dist_coords)):
        x = round(dist_coords[i][0])
        y = round(dist_coords[i][1])

        neighbors = [[x - 1, y - 1], [x, y - 1], [x + 1, y - 1],
                     [x - 1, y], [x, y], [x + 1, y],
                     [x - 1, y + 1], [x, y + 1], [x + 1, y + 1]]

        for m in range(9):
            x = neighbors[m][0]
            y = neighbors[m][1]

            for j in range(len_wm):
                # optimization with two additional for loops maybe or slicing the array more efficient
                # 4x4 matrix is NOT in the left column of the square around the keypoint
                if (k + j) % k != 0:
                    row = y - region_size + (sub_size * (j // k))
                    col = x - region_size + ((j % k) * sub_size)
                # 4x4 matrix is in the left column of the square around the keypoint
                else:
                    row = y - region_size + (sub_size * j // k)
                    col = x - region_size

                # perform singular value decomposition (SVD) on 4x4 matrices in the feature region
                Un, Sn, Vnh = np.linalg.svd(ychannel[row: row + sub_size, col: col + sub_size], full_matrices=False)

                D = svd_matrices[i][j][0] @ np.diag(Sn) @ svd_matrices[i][j][2]

                W = D - svd_matrices[i][j][1]

                if Un[1][0] >= Un[2][0]:
                    watermark_bit = '0'
                else:
                    watermark_bit = '1'

                possible_wm += watermark_bit

            possible_watermarks.append(possible_wm)
            possible_wm = ''

    compare_img = cv.drawKeypoints(img, keypoints_final, 0, color=500, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('compare2_img.jpg', compare_img)

    # Calculate hamming distance between all possible watermarks. The one with the highest similarity is selected.
    minimum = len_wm
    w1 = possible_watermarks[0]
    possible_watermarks_final = []
    print(w1)
    print(possible_watermarks[1])
    print(possible_watermarks[2])
    for i in range(len(possible_watermarks) - 1):
        for j in range(i + 1, len(possible_watermarks) - 1):
            curr = sum(c1 != c2 for c1, c2 in zip(possible_watermarks[i], possible_watermarks[j]))
            if curr < 19:
                w1 = possible_watermarks[i]
                w2 = possible_watermarks[j]
                possible_watermarks_final.append(w1)
                possible_watermarks_final.append(w2)
            '''if curr < minimum:
                minimum = curr
                w1 = possible_watermarks[i]
                w2 = possible_watermarks[j]
                print(minimum)
                if minimum < 50:
                    possible_watermarks_final.append(w1)
                    possible_watermarks_final.append(w2)'''
    print(len(possible_watermarks_final))
    for j in possible_watermarks_final:
        bin_blocks = [j[i:i + 7] for i in range(0, len_wm, 7)]
        watermark_rec = ''.join(chr(int(b, 2)) for b in bin_blocks)
        # print('Ist DAS das Wasserzeichen?!?!? omg')
        print(watermark_rec)

    return


# TODO: optimizations, exception handling, watermark extraction
if __name__ == '__main__':
    v = np.array([[860, 1000],
                  [2867, 730],
                  [2395, 3467],
                  [226, 2955]])
    vo = np.array([[0, 0],
                   [3248, 0],
                   [3248, 3258],
                   [0, 3258]])
    # Screenshot resistant for alpha = 0.03
    coord = np.array([[442, 362],
                      [4307, 466],
                      [4411, 2656],
                      [652, 3288]])
    # svd1 = embedding('Img/stones.jpg', 'UniMuensterformr/formr.de.muenster', 0.02)[1]
    # extraction('Img/IMG_20240215_042629199.jpg', svd1, True, coord)

    # Not screenshot resistant for alpha 0.01
    c = np.array([[239, 219],
                  [4402, 391],
                  [4447, 2814],
                  [290, 3121]])
    # svd2 = embedding('Img/stones.jpg', 'UniMuensterformr/formr.de.muenster', 0.01)[1]
    # extraction('Img/IMG_20240215_042802858.jpg', svd2, True, c)

    # Testbild1 sent via messenger app, watermark recovery successfully
    # svd = embedding('Img/Testbild1.png', 'IstDasWasserzeichenJPEGResistent?', 0.01)[1]
    # extraction('Img/jpegtest.jpg', svd)

    james_coords = np.array([[0, 0],
                             [1000, 90],
                             [1000, 600],
                             [0, 500]])

    # projection_transformation('Img/IMG_20240213_182832385.jpg', coord, original_vertices)
    svd = embedding('Img/Testbild5.png', 'uni-muenster.de/formr', 0.03)[1]
    extraction('Img/watermarked.jpg', svd)
    # embedding('Img/Testbild1.png', 'uni-muenster.de/formr', 0.02)
    # extraction('Img/IMG_20240213_053254680.jpg', embedding('Img/james.jpg', 'uni-muenster.de/formr')[1], True, coord)
    # extraction('Img/IMG_20240213_053254680.jpg', embedding('Img/james.jpg', 'uni-muenster.de/formrspodigfpds')[1])
    # james_svd = embedding('Img/Testbild3.png', 'UniMuenster/formr/Lennart', 0.01)[1]
    # extraction('Img/watermarked.jpg', james_svd)
    # extraction('Img/bank.jpg', james_svd, True, james_coords)
