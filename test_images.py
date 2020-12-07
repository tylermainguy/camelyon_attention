from PIL import Image
from pathlib import Path


def main():

    image_dir = "data/train/downsampled"

    image_dir = Path(image_dir)

    im_shapes = []
    n_images = 0
    for file_path in image_dir.rglob("*.tif"):
        n_images += 1
        im = Image.open(file_path)

        print(im.size)

        im_shapes.append(im.size)

    print("Total number of images: {}".format(n_images))


if __name__ == "__main__":
    main()
