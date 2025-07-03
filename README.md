## :star2::star2:Update:star2::star2:
We **STRONGLY** recommend updating the evaluation codes ([sodaa.py](https://github.com/shaunyuan22/SODA-mmrotate/blob/main/mmrotate/datasets/sodaa.py) and [sodaa_eval.py](https://github.com/shaunyuan22/SODA-mmrotate/blob/main/mmrotate/datasets/sodaa_eval/sodaa_eval.py)), which now support multi-processing and multi-GPU parallelism and significantly boosts the evaluation efficiency 🚀

For intuitive comparison, we show the overall time cost before and after applying the updated evaluation script as follows.
| Step | Before | After |
|:--------------------:|:----------:|:----------:|
| Merge | ~20m | ~25s |
| IoU Calculation | ~1h30m | ~105s  |
| Per Image Evaluation | ~2h20m  | ~135s |
| Accumulation | ~17s | ~17s |
| Overall| ~4h40m | ~280s  |

The results are tested on four RTX2080Ti GPUs with Rotated FCOS.

## Introduction

This project is based on the open source object detection toolbox [MMRotate](https://github.com/open-mmlab/mmrotate), please refer to [Installation](https://mmrotate.readthedocs.io/en/latest/install.html) for installation instructions first.

The benchmark experiments work with **Python 3.8**, **PyTorch 1.10** and **mmrotate 0.3.0**, and corresponding configs can be found at `sodaa-benchmarks`. 

## Data preparation
Please refer to [tools/data/sodaa](https://github.com/shaunyuan22/SODA-mmrotate/tree/main/tools/data/sodaa) for more details.

 ## **License**
Our SODA-A dataset is licensed under [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/), which means it is freely available for **academic use only**, and any **commercial use is prohibited**.

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
