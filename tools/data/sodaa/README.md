# Preparing SODA-A Dataset

<!-- [DATASET] -->

```bibtex
@article{cheng2022towards,
  title={Towards large-scale small object detection: Survey and benchmarks},
  author={Cheng, Gong and Yuan, Xiang and Yao, Xiwen and Yan, Kebing and Zeng, Qinghua and Han, Junwei},
  journal={arXiv preprint arXiv:2207.14096},
  year={2022}
}
```


## download SODA-A dataset

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

## split SODA-A dataset

The original images will be cropped to 800\*800 patches with the stride of 150.

```shell
python tools/data/sodaa/sodaa_split.py --base-json sodaa_train.json 
```

This script is forded from [BboxToolkit](https://github.com/jbwang1997/BboxToolkit), more details please refer to the project page.

## change configurations in split json files

Please change `img_dirs`, `ann_dirs` and `save_dir` in json files before run the script.

