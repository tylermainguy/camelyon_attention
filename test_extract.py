import openslide
from openslide import open_slide
from PIL import Image
from matplotlib import pyplot as plt


def test_extract():

    img_loc = "data/train/original/tumor/tumor_001.tif"

    slide = open_slide(img_loc)

    # print(slide.properties)
    dims = slide.dimensions
    print(dims)

    # x, y = dims[0] // 2, dims[1] // 3

    # im = slide.read_region((70000, 200000), 10, (512, 512))

    im = slide.get_thumbnail((512, 512))
    plt.imshow(im)
    plt.show()


def main():
    test_extract()


if __name__ == "__main__":
    main()
