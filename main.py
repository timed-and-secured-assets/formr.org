import cv2

from watermark import embed_watermark, extract_watermark


def main():
    # Embed watermark
    image = cv2.imread("images/input.jpg", flags=cv2.IMREAD_UNCHANGED)
    watermark = cv2.imread("images/watermark.jpg", flags=cv2.IMREAD_GRAYSCALE)
    embedded_image = embed_watermark(image, watermark)
    cv2.imwrite("images/output.png", img=embedded_image)

    # Extract watermark
    embedded_image = cv2.imread("images/output.png", flags=cv2.IMREAD_COLOR)
    watermark = extract_watermark(embedded_image, (50, 50))
    cv2.imwrite("images/extracted-watermark.png", img=watermark)


if __name__ == "__main__":
    main()
