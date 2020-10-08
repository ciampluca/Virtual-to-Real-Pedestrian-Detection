# Virtual-to-Real-Pedestrian-Detection

This is the official implementation for our paper published at *Sensors - Special Issue on Visual Sensors for Object Tracking and Recognition*:

**[Virtual to Real adaptation of Pedestrian Detectors](https://www.mdpi.com/1424-8220/20/18/5250)**

[Luca Ciampi](https://scholar.google.it/citations?user=dCjyf-8AAAAJ&hl=it), [Nicola Messina](https://scholar.google.it/citations?user=g-UGCd8AAAAJ&hl=it), [Fabrizio Falchi](https://scholar.google.it/citations?user=4Vr1dSQAAAAJ&hl=it), [Claudio Gennaro](https://scholar.google.it/citations?user=sbFBI4IAAAAJ&hl=it), [Giuseppe Amato](https://scholar.google.it/citations?user=dXcskhIAAAAJ&hl=it)


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

- ViPeD Dataset (**[ViPeD - Virtual Pedestrian Dataset](http://aimir.isti.cnr.it/viped/)**) - Just if you want to 
re-train the model
```
wget http://datino.isti.cnr.it/viped.zip
unzip viped.zip -d data
rm viped.zip
chmod -R 755 data/ViPeD
```
- MOT17Det Dataset (**[MOT17Det](https://motchallenge.net/data/MOT17Det/)**)
```
wget https://motchallenge.net/data/MOT17Det.zip
unzip MOT17Det.zip -d data
```
- MOT20Det Dataset (**[MOT20Det](https://motchallenge.net/data/MOT20Det/)**)
```
wget https://motchallenge.net/data/MOT20Det.zip
unzip MOT20Det.zip -d data
```
You should now have 3 folders in the data directory, corresponding to 3 different datasets (ViPeD, MOT17Det and 
MOT20Det). They should have a common structure: imgs containing images and bbs containing the associated txt files of 
the annotations. Annotations of the bounding boxes are in the format [x_center, y_center, height, width] relative to the 
image size.

If you want, you can put the datasets in a different folder than the data one. In this case, you have to modify the 
train_val.py and test.py files accordingly.


## Evaluate


## Train
In order to train the model using the ViPeD dataset and validate over all the real-world datasets, issue the following 
command:
```
python train_val.py --train-on viped --validate-on all --tensorboard-file-name train_viped_validate_all
```
If you want to train the model using the Mixed-Batch Domain Adaptation Technique, for example using the ViPeD and the
MOT17Det datasets, issue the following command:
```
python train_val.py --train-on viped,MOT17Det --validate-on all --tensorboard-file-name train_mixedBatch_vipedAndMOT17DET_validate_all
```
Many other options are available. Issue the following command to see them:
```
python train_val.py --help
```


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
