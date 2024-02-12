import argparse
import sys


def embed(args):
    sys.exit(0)


def extract(args):
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Embed and extract watermarks from images.")
    subparsers = parser.add_subparsers(help="commands")

    embed_parser = subparsers.add_parser("embed", help="embed a watermark into an image")
    embed_parser.add_argument("-i", "--image_path", type=str, required=True,
                              help="path to the image to add the watermark to")
    embed_parser.add_argument("-o", "--output_path", type=str, required=False,
                              help="path where to save the watermarked image to")
    embed_parser.add_argument("-w", "--watermark", type=str, required=False,
                              help="watermark text to embed in the image")
    embed_parser.add_argument("-m", "--method", type=str, required=True,
                              help="method to use for the embedding")
    embed_parser.set_defaults(func=embed)

    extract_parser = subparsers.add_parser("extract", help="extract a watermark from an image")
    extract_parser.add_argument("-i", "--image_path", type=str, required=True,
                                help="path to the image to extract the watermark from")
    extract_parser.add_argument("-m", "--method", type=str, required=True,
                                help="method to use for the extraction")
    extract_parser.add_argument("-o", "--output_path", type=str, required=False,
                                help="path where to save the qr code to")
    extract_parser.set_defaults(func=extract)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
