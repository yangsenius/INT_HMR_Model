## Capturing the Motion of Every Joint: 3D Human Pose and Shape Estimation with Independent Tokens

[[project]](https://yangsenius.github.io/INT_HMR_Model/) [[arxiv]](https://arxiv.org/abs/2303.00298) [[paper]](https://openreview.net/forum?id=0Vv4H4Ch0la)[[examples]](https://yangsenius.github.io/INT_HMR_Model/)

<img src="doc/dance5_.gif" width="30%"> <img src="doc/micheal2.gif" width="40%">  <img src="doc/out3.gif" width="19%"> 

*The multi-person videos above are based on the VIBE detection and tracking framework.*


> [Capturing the motion of every joint: 3D human pose and shape estimation with independent tokens](https://openreview.net/pdf?id=0Vv4H4Ch0la),
>
> [Sen Yang](https://scholar.google.com/citations?user=z5O3DLcAAAAJ&hl=zh-CN), [Wen Heng](), [Gang Liu](https://scholar.google.com/citations?user=ZyzfB9sAAAAJ&hl=zh-CN&authuser=1), [Guozhong Luo](https://github.com/guozhongluo), [Wankou Yang](https://scholar.google.com/citations?user=inPYAuYAAAAJ&hl=zh-CN), [Gang Yu](https://www.skicyyu.org/),
>
> *ICLR 2023 (spotlight)*


## Getting Started


This repo is based on the enviroment of `python>=3.6` and `PyTorch>=1.8`. It's better to use the virtual enironment of `conda`

```
conda create -n int_hmr python=3.6 && conda activate int_hmr
```

Install `PyTorch` following the steps of the official guide on [PyTorch website](https://pytorch.org/get-started/locally/).

The models in the paper were trained using the distributed training framework `Horovod`. If you want to train the model distributedly using this code, please install the `Horovod` following the [website](https://horovod.readthedocs.io/en/stable/), we use the version of horovod:0.3.3.

And install the dependencies using `conda`:

```
pip install -r requirements.txt
```

## Data preparation

We follow the steps of [MAED](https://github.com/ziniuwan/maed) repo to prepare the training data. Please refer to [data.md](doc/data.md)

## Training 


To run on a machine with 4 GPUs:

```
sh hvd_start.sh 4 localhost:4
```

To run on 4 machines with 4 GPUs each

```
sh hvd_start.sh 16 server1_ip:4,server2_ip:4,server3_ip:4,server4_ip:4
```
Here we show the training commands of using a single machine with 4 GPUs for the proposed scheme of progressive 3-stage training.

1.Image based pre-training:
```
sh exp/phase1/hvd_start.sh 4 localhost:4
``` 
2.Image/Video based pre-training:
```
sh exp/phase2/hvd_start.sh 4 localhost:4
``` 
3.Fine-tuning:
```
sh exp/phase3/hvd_start.sh 4 localhost:4
``` 

## Evaluation

```
sh exp/eval/hvd_start.sh 4 localhost:4
```

## Citation
If you find this repository useful please give it a star 🌟 or consider citing our work:

```
@inproceedings{
yang2023capturing,
title={Capturing the Motion of Every Joint: 3D Human Pose and Shape Estimation with Independent Tokens},
author={Sen Yang and Wen Heng and Gang Liu and GUOZHONG LUO and Wankou Yang and Gang YU},
booktitle={The Eleventh International Conference on Learning Representations (ICLR) },
year={2023},
url={https://openreview.net/forum?id=0Vv4H4Ch0la}
}
```

## Credit
Thanks for the great open-source codes of [MAED](https://github.com/ziniuwan/maed) and [VIBE](https://github.com/mkocabas/VIBE)

