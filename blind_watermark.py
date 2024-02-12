import pathlib
import math
import cv2
import qrcode
import numpy as np

import blind_watermark_core


def embed_watermark(image_path, watermark_text, output_path=None):
    image_path = pathlib.Path(image_path)
    image = cv2.imread(str(image_path), flags=cv2.IMREAD_UNCHANGED)

    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=1, border=0)
    qr.add_data(watermark_text)
    qr_code = qr.make_image(fill_color=(0, 0, 0), back_color=(255, 255, 255))

    watermark = cv2.cvtColor(np.array(qr_code), cv2.COLOR_RGB2GRAY)

    watermark_size = (watermark.shape[0] + 1) * (watermark.shape[1] + 1) * 4
    width = math.sqrt(watermark_size * (image.shape[0] / image.shape[1]))
    height = watermark_size / width
    width = math.ceil(width) * 4 * 2
    height = math.ceil(height) * 4 * 2

    if image.shape[0] < height or image.shape[1] < width:
        image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LINEAR)

    embedded_image = blind_watermark_core.embed_watermark(image, watermark)

    if output_path is None:
        output_path = image_path.parent / "watermark" / image_path.name
    else:
        output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), img=embedded_image)

    return output_path, watermark.shape[0]


def extract_watermark(image_path, watermark_key, output_path=None):
    image_path = pathlib.Path(image_path)
    image = cv2.imread(str(image_path), flags=cv2.IMREAD_UNCHANGED)

    watermark_shape = (watermark_key, watermark_key)

    watermark = blind_watermark_core.extract_watermark(image, watermark_shape)

    watermark = np.where(watermark > 127, 255, 0).astype(np.uint8)
    watermark = np.pad(watermark, 4, constant_values=255)
    watermark = np.repeat(watermark, 10, axis=0)
    watermark = np.repeat(watermark, 10, axis=1)

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img=watermark)

    detector = cv2.QRCodeDetector()
    data, vertices_array, binary_qrcode = detector.detectAndDecode(watermark)

    return data, output_path
