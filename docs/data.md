# Data Preparation

Create a new directory `data` to store all the datasets.

## Ref-COCO

Download the dataset from the official website [COCO](https://cocodataset.org/#download).   
RefCOCO/+/g use the COCO2014 train split.
Download the annotation files from [github](https://github.com/lichengunc/refer).

Convert the annotation files:

```
python3 tools/data/convert_refexp_to_coco.py
```

Finally, we expect the directory structure to be the following:

```
MUTR
├── data
│   ├── coco
│   │   ├── train2014
│   │   ├── refcoco
│   │   │   ├── instances_refcoco_train.json
│   │   │   ├── instances_refcoco_val.json
│   │   ├── refcoco+
│   │   │   ├── instances_refcoco+_train.json
│   │   │   ├── instances_refcoco+_val.json
│   │   ├── refcocog
│   │   │   ├── instances_refcocog_train.json
│   │   │   ├── instances_refcocog_val.json
```


## Ref-YouTube-VOS

Download the dataset from the competition's website [here](https://competitions.codalab.org/competitions/29139#participate-get_data).
Then, extract and organize the file. We expect the directory structure to be the following:

```
MUTR
├── data
│   ├── ref-youtube-vos
│   │   ├── meta_expressions
│   │   ├── train
│   │   │   ├── JPEGImages
│   │   │   ├── Annotations
│   │   │   ├── meta.json
│   │   ├── valid
│   │   │   ├── JPEGImages
```

## Ref-DAVIS 2017

Downlaod the DAVIS2017 dataset from the [website](https://davischallenge.org/davis2017/code.html). Note that you only need to download the two zip files `DAVIS-2017-Unsupervised-trainval-480p.zip` and `DAVIS-2017_semantics-480p.zip`.
Download the text annotations from the [website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions).
Then, put the zip files in the directory as follows.


```
MUTR
├── data
│   ├── ref-davis
│   │   ├── DAVIS-2017_semantics-480p.zip
│   │   ├── DAVIS-2017-Unsupervised-trainval-480p.zip
│   │   ├── davis_text_annotations.zip
```

Unzip these zip files.
```
unzip -o davis_text_annotations.zip
unzip -o DAVIS-2017_semantics-480p.zip
unzip -o DAVIS-2017-Unsupervised-trainval-480p.zip
```

Preprocess the dataset to Ref-Youtube-VOS format. (Make sure you are in the main directory)

```
python tools/data/convert_davis_to_ytvos.py
```

Finally, unzip the file `DAVIS-2017-Unsupervised-trainval-480p.zip` again (since we use `mv` in preprocess for efficiency).

```
unzip -o DAVIS-2017-Unsupervised-trainval-480p.zip
```
