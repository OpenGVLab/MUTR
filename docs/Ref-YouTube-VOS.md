### Ref-YouTube-VOS

To evaluate the results, please upload the zip file to the [competition server](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-submit_results).


| Backbone| J&F | J | F | Model | Submission | 
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet-50 | 61.9 | 60.4 | 63.4 | [model](https://drive.google.com/file/d/1VEgmQgdsV5fIAkb8zcJ2PVb7k_coZqZe/view?usp=sharing) | [link]() |
| ResNet-101 | 63.6 | 61.8 | 65.4 | [model](https://drive.google.com/file/d/1SM3SxwJvlENAl6F2vVzc8vtW1jrZiPaq/view?usp=sharing) | [link]() |
| Swin-L | 68.4 | 66.4 | 70.4 | [model](https://drive.google.com/file/d/1TSoZLADW6JQhOanFn8yV_ZnIOBn1kqh6/view?usp=sharing) | [link]() |
| Video-Swin-T | 64.0 | 62.2 | 65.8 | [model](https://drive.google.com/file/d/1q9yAJx68UwtGRtjTXECw0dqWPgt2JZnQ/view?usp=sharing) | [link]() |
| Video-Swin-S | 65.1 | 63.0 | 67.1 | [model](https://drive.google.com/file/d/1-lYh3trE9TcaKtes-ETWUCCalieudg1t/view?usp=sharing) | [link]() |
| Video-Swin-B | 67.5 | 65.4 | 69.6 | [model](https://drive.google.com/file/d/1_btRlsRMOpwGceKTCeeW5v7S14Zq7LGX/view?usp=sharing) | [link]() |
| ConvNext-L | 66.7 | 64.8 | 68.7 | [model](https://drive.google.com/file/d/1sYH6JpsqdB0kwW4qkOIcQGAdHsyWyXEL/view?usp=sharing) | [link]() |
| ConvMAE-B | 66.9 | 64.7 | 69.1 | [model](https://drive.google.com/file/d/1kC2052ao_49P_ecKWib5oZEQpxeTvRjq/view?usp=sharing) | [link]() |

### Training

```
./scripts/dist_train.sh  --backbone [backbone] --backbone_pretrained [/path/to/backbone_pretrained_weight] [other args]
```

For example, training the Video-Swin-Tiny model, run the following command:

```
./scripts/dist_train.sh --backbone video_swin_t_p4w7 --backbone_pretrained video_swin_pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth
```

### Inference & Evaluation


First, inference using the trained model.

```
./scripts/dist_test_ytvos.sh [backbone] 
```
For example, evaluating the Swin-Large model, run the following command:

```
./scripts/dist_test_ytvos.sh swin_l_p4w7
```


To evaluate the results, please upload the zip file to the [competition server](https://competitions.codalab.org/competitions/29139#participate-submit_results).


