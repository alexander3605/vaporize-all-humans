# VAPORIZE ALL HUMANS! :gun:

```
conda create --name human_vaporizer
conda activate human_vaporizer
poetry install
```

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
```

batch["image"].shape
torch.Size([1, 3, 1000, 1504])
batch["mask"].shape
torch.Size([1, 1, 1000, 1504])
batch["unpad_to_size"]
[tensor([996]), tensor([1499])]
predict_config.out_key
'inpainted'

predict_config.model.path
'/Users/albertoparravicini/Documents/repositories/human-vaporizer/external/lama/big-lama'
predict_config.model.checkpoint
'best.ckpt'