# VAPORIZE ALL HUMANS! :gun: :alien:

Have you ever had your beautiful pictures ruined by an overabundance of those little pesky humans?
Wish you could just turn them all to **cosmic dust** with a simple click?
Worry no more! 

<p align="center">
  <img src="/data/gif/vaporizer.gif?raw=true" width="1200px">
</p>



The human vaporizer is still being calibrated: some ingenious humans might still escape it, and some traces of vaporized humans might be left of in the final picture.

# Install

Starting to vaporize humans could not be any simpler! Just run the following in your terminal. It assumes that you have already `git` and `conda` installed.

```shell
# Download the repo
git clone git@github.com:AlbertoParravicini/vaporize-all-humans.git
# Create a conda environment
conda create --name vaporize_all_humans -y python=3.10
conda activate vaporize_all_humans
# Download poetry (skip this if you have it already)
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
# Install dependencies
poetry install
# Download the big-lama model (~300MB, it's big!)
cd external/lama
echo A | curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip
rm big-lama.zip
cd -
```

# Vaporize

Ok, time to start **vaporizing humans**! First, get pumped up with a little demo
```shell
python tools/vaporize.py --demo
```

All good? Now start vaporizing humans in **your** pictures, like this
```shell
python tools/vaporize.py -i data/samples/P1000908.jpg -o data/samples/output
```

If you want to have more control over what's going on, check out all the options as 
```shell
python tools/vaporize.py -h
```

Here's a list of the most useful ones
* `-i`, `--input`: path to one or more images from which to erase humans.
* `-o`, `--output`: path to the directory where the output images are stored. If not present, store them in the same folder as the input images.
* `-s`, `--patch-size`: size of patches used by the vaporizer. Higher values give better quality, but it takes more time to vaporize humans. Recommended values are in the range [128, 512].
* `--show-patches`: if present, show all the inpainted patches, along with the original image patch and the segmentation mask. This is useful if you want to precisely see what has been vaporized, and how

<!-- 

Hidden notes about how to setup the repo from scratch, 
they might be useful again at some point in the future

## Install YOLOv5

We use YOLOv5 to detect human bounding boxes

```
pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies
```

## Install LAMA

Standalone environment for LAMA

```
conda create -n lama
conda activate lama
conda install pytorch torchvision  -c pytorch -y 
pip install pytorch-lightning==1.2.9
pip install -r requirements.txt  # This will probably fail
pip install opencv-python
pip install hydra-core
pip install albumentations==0.5.2
pip install kornia==0.5.0
pip install webdataset
pip install easydict
pip install pandas
```

Integrate it with existing enviroment

```
pip3 install wldhx.yadisk-direct
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip
curl -L $(yadisk-direct https://disk.yandex.ru/d/EgqaSnLohjuzAg) -o lama-models.zip
unzip lama-models.zip
curl -L $(yadisk-direct https://disk.yandex.ru/d/xKQJZeVRk5vLlQ) -o LaMa_test_images.zip
unzip LaMa_test_images.zip
``` -->
