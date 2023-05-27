### Ref-YouTube-VOS

To evaluate the results, please upload the zip file to the [competition server](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-submit_results).


| Backbone| J&F | J | F | Model | Submission | 
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet-50 | 61.9 | 60.4 | 63.4 | [model](https://drive.google.com/file/d/1bNkR4n7be3hYwtaYp75c2WNrbmyh-Ik2/view?usp=sharing) | [link]() |
| ResNet-101 | 63.6 | 61.8 | 65.4 | [model](https://drive.google.com/file/d/1ZOev9AZM_GRpnsKjg0_gpRFKGrJcm0S5/view?usp=sharing) | [link]() |
| Swin-L | 68.4 | 66.4 | 70.4 | [model](https://drive.google.com/file/d/1e2-BXV3HGxPxWFKO-z34PZDBShCzEmz9/view?usp=sharing) | [link]() |
| Video-Swin-T | 64.0 | 62.2 | 65.8 | [model](https://drive.google.com/file/d/1-TkdQksTrmB253ao99NgnmsrsQkous2V/view?usp=sharing) | [link]() |
| Video-Swin-S | 65.1 | 63.0 | 67.1 | [model](https://drive.google.com/file/d/1gVeOE20nmZzONTQSdBhPHBg_hBZnXgxI/view?usp=sharing) | [link]() |
| Video-Swin-B | 67.5 | 65.4 | 69.6 | [model](https://drive.google.com/file/d/11poAYPbJDB2R_DlsDhRrSYvgOzaihpTN/view?usp=sharing) | [link]() |
| ConvNext-L | 66.7 | 64.8 | 68.7 | [model](https://drive.google.com/file/d/1d6C73EmSpQZBIuhBDu1gnzibDXYxCYDz/view?usp=sharing) | [link]() |
| ConvMAE-B | 66.9 | 64.7 | 69.1 | [model](https://drive.google.com/file/d/1kM9VLjdzl_YKN09WD6iSzmvtxVYU_NiE/view?usp=sharing) | [link]() |

### Training

```
./scripts/dist_train.sh  --backbone [backbone] --backbone_pretrained [/path/to/backbone_pretrained_weight] [other args]
```

For example, training the Video-Swin-Tiny model, run the following command:

```
./scripts/dist_train.sh --backbone video_swin_t_p4w7 --backbone_pretrained video_swin_pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth
```

### Inference & Evaluation

Inference using the trained model.
```
./scripts/dist_test_ytvos.sh [backbone] 
```

For example, evaluating the Swin-Large model, run the following command:

```
./scripts/dist_test_ytvos.sh swin_l_p4w7
```

To evaluate the results, please upload the zip file to the [competition server](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-submit_results).

Note that, if you use the weights we provide, you should put the weights in the corresponding path.  ./results/[backbone]/ckpt/backbone_weight.pth


