# Attention IoU: Examining Biases in CelebA using Attention Maps

**This repository provides the code and data for our CVPR 2025 paper " Attention IoU: Examining Biases in CelebA using Attention Maps".**

### [Aaron Serianni](https://aaronserianni.com/), [Tyler Zhu](https://tylerzhu.com/), [Olga Russakovsky](https://www.cs.princeton.edu/~olgarus/), [Vikram V. Ramaswamy](https://www.cs.princeton.edu/~vr23/)
> **Abstract:** *Computer vision models have been shown to exhibit and amplify biases across a wide array of datasets and tasks. Existing methods for quantifying bias in classification models primarily focus on dataset distribution and model performance on subgroups, overlooking the internal workings of a model. We introduce the Attention-IoU (Attention Intersection over Union) metric and related scores, which use attention maps to reveal biases within a model's internal representations and identify image features potentially causing the biases. First, we validate Attention-loU on the synthetic Waterbirds dataset, showing that the metric accurately measures model bias. We then analyze the CelebA dataset, finding that Attention-loU uncovers correlations beyond accuracy disparities. Through an investigation of individual attributes through the protected attribute of Male, we examine the distinct ways biases are represented in CelebA. Lastly, by subsampling the training set to change attribute correlations, we demonstrate that Attention-loU reveals potential confounding variables not present in dataset labels.*

## Dataset Setup
To download and create the Waterbirds dataset, follow the instructions provided in [https://github.com/kohpangwei/group_DRO/tree/master](https://github.com/kohpangwei/group_DRO/tree/master). Use the `dataset_scripts/generate_waterbirds.py` script, specifying the dataset bias (`.5, .6, .7, .8, .9, .95, 1` for our experiments) under the `confounder_strength` variable (line 20).

Follow the instructions in [https://github.com/switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) to download the CelebAMask-HQ dataset.

## Python Requirements
Use conda or Mamba to install the required packages into an environment using the provided [`bias.yml`](bias.yml) file. We use Python 3.12.2.

## Main Experiments

### Waterbirds
To train models for each bias percentage, run:

```bash
python run_trainer.py --waterbirds --data_dir {WATERBIRDS_DATASET_DIRECTORY} --model resnet18 --save_model models/ --train_multiple 20 --waterbirds_percentage {DATASET_BIAS}
```

To calculate the mask score for each dataset bias percentage, run:
```bash
python run_bias.py --waterbirds --data_dir {WATERBIRDS_DATASET_DIRECTORY} --model resnet18 --load_model models/ --score_sample --score mask --no-resize_attentions --waterbirds_percentage {DATASET_BIAS}
```
To get accuracies and average maps for each dataset bias percentage, run:
```bash
python run_evaluations.py --waterbirds --data_dir {WATERBIRDS_DATASET_DIRECTORY} --load_model models/ --score_sample --average_maps --test_accuracy --waterbirds_percentage {DATASET_BIAS}
```

### CelebA
To train models, run:
```bash
python run_trainer.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --save_model models/ --train_multiple 20
```

To get raw evaluations and and run average heatmaps on all attributes, run:
```bash
python run_evaluations.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --load_model models/ --score_sample --average_maps --average_masks --evaluate_model --test_accuracy
```

To calculate heatmap scores for specified target label, comparing to the `Male` heatmap, run:
```bash
python run_bias.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --load_model models/ --score_sample --score heatmap --heatmap_score_target {TARGET_ATTRIBUTE} Male
```

To calculate mask scores for specified target label, run:
```bash
python run_bias.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --load_model models/ --score_sample --score mask --no-resize_attentions --score_sample --mask_score_target {TARGET_ATTRIBUTE}
```

To do one-sided training for when feature is or is not present (target attribute was Eyeglasses in our experiments):
```bash
python run_trainer.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --save_model models/ --train_multiple 10 --one_sided_target {TARGET_ATTRIBUTE} --one_sided_train {positive/negative}
```

### CelebA subgroups and one-sided training

To get average heatmaps for when feature is or is not present (target attribute was Eyeglasses in our experiments):
```bash
python run_evaluations.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --results_dir results/one_sided/ --load_model models/ --score_sample --no-resize_attentions --average_maps --one_sided_target {TARGET_ATTRIBUTE} --one_sided_train {positive/negative}
```

To get average heatmaps for dataset subgroups (target attribute was Mustache in our experiments):
```bash
python run_evaluations.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --load_model models/ --no-resize_attentions --average_maps --average_maps_groups {TARGET_ATTRIBUTE} Male
```

To generate commands to run the group subsampling experiments (MCC was -0.3 for Blond_Hair, and -0.31 for Wavy_Hair in our experiments), use the following script:
```bash
python run_subgroups.py {TARGET_ATTRIBUTE} {ORIGINAL_MCC} --data_dir {CELEB_DATASET_DIRECTORY}
```

## Figures
Code to create all figures are in [`make_figures.ipynb`](make_figures.ipynb) Jupyter notebook, and saved to `figures/`.

## Results
All of our raw results are provided in the [`results/`](results/) folder.

## Additional Experiments

### Running with EfficientNetV2-S

To train Waterbirds models with EfficientNet, run:
```bash
python run_trainer.py --waterbirds --data_dir {WATERBIRDS_DATASET_DIRECTORY} --model efficientnet --save_model models/ --train_multiple 10 --waterbirds_percentage {DATASET_BIAS}
```

To calculate the Waterbirds mask score with EfficientNet, run:
```bash
python run_bias.py --waterbirds --data_dir {WATERBIRDS_DATASET_DIRECTORY} --model efficientnet --load_model models/ --score_sample --score mask --no-resize_attentions --waterbirds_percentage {DATASET_BIAS}
```

To train CelebA models with EfficientNet, run:
```bash
python run_trainer.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --model efficientnet --save_model models/ --train_multiple 10
```

To get raw CelebA evaluations on all attributes with EfficientNet, run:
```bash
python run_evaluations.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --model efficientnet --load_model models/ --score_sample --evaluate_model
```

To calculate CelebA heatmap scores for specified target label with EfficientNet, comparing to the `Male` heatmap, run:
```bash
python run_bias.py --celeb --data_dir {CELEB_DATASET_DIRECTORY} --model efficientnet --load_model models/ --score_sample --score heatmap --heatmap_score_target {TARGET_ATTRIBUTE} Male
```

## Acknowledgements
We acknowledge support from the Princeton SEAS Innovation Grant to VVR, and from the Princeton University's Office of Undergraduate Research Undergraduate Fund for Academic Conferences through the Hewlett Foundation Fund to AS. This material is based upon work supported by the National Science Foundation under grant No. 2145198 to OR. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. The experiments presented in this work were performed on computational resources managed and supported by Princeton Research Computing, a consortium of groups including the Princeton Institute for Computational Science and Engineering PICSciE and Research Computing at Princeton University.