# Virtual-to-Real-Pedestrian-Detection

This is the official implementation for our paper:

**[Virtual to Real adaptation of Pedestrian Detectors](https://www.mdpi.com/1424-8220/20/18/5250)**

[Luca Ciampi](https://scholar.google.it/citations?user=dCjyf-8AAAAJ&hl=it), [Nicola Messina](https://scholar.google.it/citations?user=g-UGCd8AAAAJ&hl=it), [Fabrizio Falchi](https://scholar.google.it/citations?user=4Vr1dSQAAAAJ&hl=it), [Claudio Gennaro](https://scholar.google.it/citations?user=sbFBI4IAAAAJ&hl=it), [Giuseppe Amato](https://scholar.google.it/citations?user=dXcskhIAAAAJ&hl=it)

published at *Sensors - Special Issue on Visual Sensors for Object Tracking and Recognition*

We introduce **[ViPeD](http://aimir.isti.cnr.it/viped/)** (Virtual Pedestrian Dataset), a new synthetically generated 
set of images collected with the highly photo-realistic graphical engine of the video game GTA V (Grand Theft Auto V) 
that extends the [JTA](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=25) dataset, where annotations are 
automatically acquired, suitable for the pedestrian detection task. However, when training solely on the synthetic 
dataset, the model experiences a Synthetic2Real domain shift leading to a performance drop when applied to real-world 
images. To mitigate this gap, we propose two different domain adaptation techniques suitable for the pedestrian 
detection task, but possibly applicable to general object detection. Experiments show that the network trained with 
ViPeD can generalize over unseen real-world scenarios better than the detector trained over real-world data, 
exploiting the variety of our synthetic dataset. Furthermore, we demonstrate that with our domain adaptation techniques, 
we can reduce the Synthetic2Real domain shift, making the two domains closer and obtaining a performance improvement 
when testing the network over the real-world images.

<p align="center">
  <img src="images/repo_image.png">
</p>


## Setup

1. Clone the repo and move into it:
```
git clone https://github.com/ciampluca/Virtual-to-Real-Pedestrian-Detection.git
cd Virtual-to-Real-Pedestrian-Detection
```

2. Setup python environment using virtualenv:
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Data Preparation

- ViPeD Dataset (**[ViPeD - Virtual Pedestrian Dataset](http://aimir.isti.cnr.it/viped/)**) - Note that we provide pre-trained models
already trained exploiting our dataset.
```
wget http://datino.isti.cnr.it/viped.zip
unzip viped.zip -d data
rm viped.zip
chmod -R 755 data/ViPeD
```
- MOT17Det Dataset (**[MOT17Det](https://motchallenge.net/data/MOT17Det/)**)
```
wget https://motchallenge.net/data/MOT17Det.zip
unzip MOT17Det.zip -d data/MOT17Det
rm MOT17Det.zip
python scripts_prepare_data/prepare_mot_datasets.py data/MOT17Det/
```
- MOT20Det Dataset (**[MOT20Det](https://motchallenge.net/data/MOT20Det/)**)
```
wget https://motchallenge.net/data/MOT20Det.zip
unzip MOT20Det.zip -d data
rm MOT20Det.zip
python scripts_prepare_data/prepare_mot_datasets.py data/MOT20Det/ --mot_dataset MOT20
```

You should now have 3 folders in the `data` directory, corresponding to 3 different datasets (ViPeD, MOT17Det and MOT20Det). They should have a common structure: `imgs` containing images and `bbs` containing the associated txt 
files of the annotations. 
Annotations of the bounding boxes are in the format *[x_center, y_center, height, width]* relative to the image size.

If you want, you can put the datasets in a different folder than the `data` one. In this case, you have to modify the 
config.yaml file accordingly.


## Train
In order to train the model using the ViPeD dataset and validate over the two real-world datasets MOT17Det and MOT20Det, 
issue the following command:
```
python train_val.py --cfg-file cfg/viped_training_resnet50.yaml
```
If you want to train the model using the Mixed-Batch Domain Adaptation Technique, for example using the ViPeD and the
MOT17Det datasets, issue the following command:
```
python train_val.py --cfg-file cfg/viped_training_mb_mot17_resnet50.yaml
```
If you want to fine-tune a model already trained with our ViPeD, exploiting for example the MOT20Det 
dataset, issue the following command:
```
python train_val.py --cfg-file cfg/mot20_trainingFromViped_resnet50.yaml
```
Many other cfg files are available, see the `cfg` folder. 

You can also create your personal cfg customizing the available options. 

Note that we provide many pre-trained models (see next section).


## Evaluate
You can evaluate the pre-trained models created exploiting the two domain adaptation techniques over
the two real-world datasets MOT17Det and MOT20Det. 

For example, if you want to evaluate the pre-trained model using the Mixed Batch technique over the 
MOT2Det dataset, issue the following command:
```
python train_val.py --cfg-file cfg/mot20_trainingFromViped_resnet50.yaml --load_model mot20det_mb_viped_resnet_50
```
Another cfg file for the MOT17Det dataset is available, see the `cfg` folder.
Furthermore, other options are available, for example you can validate using the coco metrics instead
of the mot ones. Just properly modify the related cfg file.


## Citations
If you find this work or code useful for your research, please cite:

```
@article{ciampi2020virtual,
  title={Virtual to Real Adaptation of Pedestrian Detectors},
  author={Ciampi, Luca and Messina, Nicola and Falchi, Fabrizio and Gennaro, Claudio and Amato, Giuseppe},
  journal={Sensors},
  volume={20},
  number={18},
  pages={5250},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

```
@inproceedings{amato2019learning,
  title={Learning pedestrian detection from virtual worlds},
  author={Amato, Giuseppe and Ciampi, Luca and Falchi, Fabrizio and Gennaro, Claudio and Messina, Nicola},
  booktitle={International Conference on Image Analysis and Processing},
  pages={302--312},
  year={2019},
  organization={Springer}
}
```
