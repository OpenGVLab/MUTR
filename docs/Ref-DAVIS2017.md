## Ref-DAVIS 2017

### Model Zoo

As described in the paper, we report the results using the model trained on Ref-Youtube-VOS without finetune.

| Backbone| J&F | J | F | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| ResNet-50 | 65.3 | 62.4 | 68.2 | [model](https://drive.google.com/file/d/1bNkR4n7be3hYwtaYp75c2WNrbmyh-Ik2/view?usp=sharing) | 
| ResNet-101 | 65.3 | 61.9 | 68.6 | [model](https://drive.google.com/file/d/1ZOev9AZM_GRpnsKjg0_gpRFKGrJcm0S5/view?usp=sharing) |
| Swin-L | 68.0 | 64.8 | 71.3 | [model](https://drive.google.com/file/d/1e2-BXV3HGxPxWFKO-z34PZDBShCzEmz9/view?usp=sharing) |
| Video-Swin-T | 66.5 | 63.0 | 70.0 | [model](https://drive.google.com/file/d/1-TkdQksTrmB253ao99NgnmsrsQkous2V/view?usp=sharing) |
| Video-Swin-S | 66.1 | 62.6 | 69.8 | [model](https://drive.google.com/file/d/1gVeOE20nmZzONTQSdBhPHBg_hBZnXgxI/view?usp=sharing)  |
| Video-Swin-B | 66.4 | 62.8 | 70.0 | [model](https://drive.google.com/file/d/11poAYPbJDB2R_DlsDhRrSYvgOzaihpTN/view?usp=sharing) |
| ConvNext-L | 69.0 | 65.6 | 72.4 | [model](https://drive.google.com/file/d/1d6C73EmSpQZBIuhBDu1gnzibDXYxCYDz/view?usp=sharing) | 
| ConvMAE-B | 69.2 | 65.6 | 72.8 | [model](https://drive.google.com/file/d/1kM9VLjdzl_YKN09WD6iSzmvtxVYU_NiE/view?usp=sharing) |


### Inference & Evaluation

```
./scripts/dist_test_davis.sh --backbone [backbone]
```

For example, evaluating the Swin-Large model, run the following command:

```
./scripts/dist_test_davis.sh --backbone swin_l_p4w7
```
Note that, if you use the weights we provide, you should put the weights in the corresponding path.  ./results/[backbone]/ckpt/backbone_weight.pth
