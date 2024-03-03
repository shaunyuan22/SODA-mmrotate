## Introduction

This project is based on the open source object detection toolbox [MMRotate](https://github.com/open-mmlab/mmrotate), please refer to [Installation](https://mmrotate.readthedocs.io/en/latest/install.html) for installation instructions first.

The benchmark experiments work with **Python 3.8**, **PyTorch 1.10** and **mmrotate 0.3.0**, and corresponding configs can be found at `sodaa-benchmarks`. 

## Data preparation
Please refer to [tools/data/sodaa](https://github.com/shaunyuan22/SODA-mmrotate/tree/main/tools/data/sodaa) for more details.

## Citation

If you use our benchmark in your research, please cite this project.


```bibtex
@ARTICLE{SODA,
  author={Cheng, Gong and Yuan, Xiang and Yao, Xiwen and Yan, Kebing and Zeng, Qinghua and Xie, Xingxing and Han, Junwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards Large-Scale Small Object Detection: Survey and Benchmarks}, 
  year={2023},
  volume={45},
  number={11},
  pages={13467-13488}
}

```

```bibtex
@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```
