from blind_watermark import WaterMark


def main():
    # Embed watermark
    wm = WaterMark()
    wm.read_img("images/input.jpg")
    wm.read_wm("images/watermark.jpg")
    wm.embed("images/output.png")

    # Extract watermark
    wm = WaterMark()
    wm.extract(filename="images/output.png", wm_shape=(50, 50), out_wm_name="images/extracted-watermark.png", )


if __name__ == "__main__":
    main()
