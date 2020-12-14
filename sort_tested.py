import numpy as np
import os
import pandas as pd
from pathlib import Path


def sort_tested():
    """
    Sort tested data into the normal or tumorous folders based on the
    csv provided for this challenge.
    """
    csv_path = "data/reference.csv"

    data_path = Path("data/test/")
    tumour_save_path = "data/test/tumour/"
    normal_save_path = "data/test/normal/"

    df = pd.read_csv(csv_path, header=None)

    name = df.iloc[:, 0]
    # print(name)
    label = df.iloc[:, 1]

    for file_path in data_path.rglob("*.tif"):
        # filename = "/".join(str(file_path).split("/")[-2:])
        filename = str(file_path).split("/")[-1]
        filename_noext = filename.split(".")[0]

        row = df.loc[df.iloc[:, 0] == filename_noext]

        print("FNAME: {}".format(filename))

        label = row.iloc[:, 1].values

        if type(label) != "str":
            label = label[0]

        if label == "Normal":
            savepath = normal_save_path + filename
        else:
            savepath = tumour_save_path + filename

        os.rename(file_path, savepath)


def rename_stupid():
    fpath = Path("data/test/")

    for file_path in fpath.rglob("*.tif"):
        spl = str(file_path).split("normal")

        if len(spl) > 1:
            fname = spl[1]

            old_path = "/".join(str(file_path).split("/")[:-1]) + "/"

            os.rename(file_path, old_path + fname)


def main():
    sort_tested()
    # rename_stupid()


if __name__ == "__main__":
    main()
