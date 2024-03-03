# Preparing SODA-A Dataset

## Download SODA-A dataset

The SODA-A dataset can be downloaded from [here](https://shaunyuan22.github.io/SODA/).

The data structure is as follows:

```none
mmrotate
├── mmrotate
├── tools
├── configs
├── data
│   ├── sodaa
│   │   ├── sodaa_split.py
│   │   ├── split_configs
│   │   │   ├── split_train.json
│   │   │   ├── split_val.json
│   │   │   ├── split_test.json
│   │   ├── Images
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── Annotations
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
```

## Split SODA-A dataset

The original images will be cropped to 800\*800 patches with the stride of 150.

```shell
python tools/data/sodaa/sodaa_split.py --base-json sodaa_train.json 
```

Please change `img_dirs`, `ann_dirs` and `save_dir` in json files before run the script. This script is forked from [BboxToolkit](https://github.com/jbwang1997/BboxToolkit), more details please refer to the project page.

## About evaluation

With regard to the evaluation, we'd like to bring two important points to your attention:
 - The evaluation is performed on the original images (**NOT ON** the splitted images).
 - The `ignore` regions will not be used in the evaluation phase.

Hence you need to filter `ignore` annotations of the original json files in the rawData directory (i.e., `Annotations`) to get available json files stored in `AnnsWoIgnore` for final performance evaluation. This can be finished by running `generate_wo_ignore.py`. Finally, you may have the following folder sturcture:

```none
SODA-A
├── rawData
│   ├── train
│   │   ├── Images
│   │   ├── Annotations
│   │   ├── AnnsWoIgnore
│   ├── val
│   │   ├── Images
│   │   ├── Annotations
│   │   ├── AnnsWoIgnore
│   ├── test
│   │   ├── Images
│   │   ├── Annotations
│   │   ├── AnnsWoIgnore
├── divData
│   ├── train
│   │   ├── Images
│   │   ├── Annotations
│   ├── val
│   │   ├── Images
│   │   ├── Annotations
│   ├── test
│   │   ├── Images
│   │   ├── Annotations
```
