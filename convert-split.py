from pathlib import Path
from shutil import move
from typing import List, Union, Tuple, Generator

import numpy as np  # type: ignore[import]
from multiprocessing import Pool
from skimage import io  # type: ignore[import]
import tqdm  # type: ignore[import]

from hci import HCI


class Engine(object):
    """Workaround for passing parameters to HCI class for multiprocessing"""
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, file):
        return HCI(file, *self.parameters).preprocess()


def get_max_mask_size(files: Union[Generator[Path, None, None], List[str]]) -> Tuple[int, int]:

    xsize = 0
    ysize = 0

    for png in files:
        mask = io.imread(png)

        [rows, cols] = np.where(mask)
        row1 = min(rows)
        row2 = max(rows)
        col1 = min(cols)
        col2 = max(cols)
        rdiff = row2 - row1
        cdiff = col2 - col1
        if rdiff > xsize:
            xsize = rdiff
        if cdiff > ysize:
            ysize = cdiff

    return (xsize, ysize)


ROI_folder = Path("data/aus_data/ROIs")
pngs = ROI_folder.glob("*.png")

bkg_file = Path("data/Background/23102019_CAL_WASH_1.mat")
cal_file = Path("data/Background/19102019_CAL_A11_1.mat")

img_folder = Path("data/aus_data/3d-expanded")
img_files = list(img_folder.glob("*.mat"))

xsize, ysize = get_max_mask_size(pngs)

print("Background")
bkg_HCI = HCI(bkg_file)
bkg_HCI.preprocess()
bkg_img = bkg_HCI.img

print("\nCalibration")
cal_HCI = HCI(cal_file, bkg_img=bkg_img)
cal_HCI.preprocess()
cal_img = cal_HCI.img

cores = 16
pool = Pool(cores)

print("\nImage")
engine = Engine([ROI_folder, True, bkg_img, cal_img, 69, 32, 32, True, (xsize, ysize), True, "3d-patches", img_folder])
for _ in tqdm.tqdm(pool.imap_unordered(engine, img_files), total=len(img_files)):
    pass
pool.close()
pool.join()

# move files to correct folders.
filesToMove = list(img_folder.glob("*.npy"))

train = list((img_folder / ".." / "train").resolve().glob("*.mat"))
val = (img_folder / ".." / "val").resolve().glob("*.mat")
test = (img_folder / ".." / "test").resolve().glob("*.mat")

train_folder = img_folder / "train/"
val_folder = img_folder / "val/"
test_folder = img_folder / "test/"

for ofile in train:
    for file in filesToMove:
        if str(ofile.stem) == str(file.stem).split("_")[0]:
            dest = train_folder / file.name
            move(file, dest)  # type: ignore[arg-type]

for ofile in val:
    for file in filesToMove:
        if str(ofile.stem) == str(file.stem).split("_")[0]:
            dest = val_folder / file.name
            move(file, dest)  # type: ignore[arg-type]

for ofile in test:
    for file in filesToMove:
        if str(ofile.stem) == str(file.stem).split("_")[0]:
            dest = test_folder / file.name
            move(file, dest)  # type: ignore[arg-type]
