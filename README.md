# BSDE-Gen

## Overview
We propose a novel deep generative model, called BSDE-Gen, which combines the flexibility of backward stochastic differential equations (BSDEs) with the power of deep neural networks for generating high-dimensional complex target data, particularly in the field of image generation.

The paper can be find on arXiv: 

Xu, Xingcheng. "Deep Generative Modeling with Backward Stochastic Differential Equations." arXiv preprint arXiv:2304.xxxxx (2023).

## Method: From source

1. Clone this repository and navigate to the BSDE-Gen folder
```bash
git clone https://github.com/xingchengxu/BSDE-Gen.git
cd BSDE-Gen
```

2. Training Run

Training on a single device: GPU/CPU
```bash
python bsde_gen_model_single_device_training.py
```

Training on a machine with multiple GPUs using the command: 
```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nproc_per_node=8 bsde_gen_model_ddp_training.py
```

or the following command:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch \ 
        --nproc_per_node=8 bsde_gen_model_ddp_training.py
```

3. Generate images/Inference

Inference on a single device: GPU/CPU
```bash
python bsde_gen_model_inference.py
```

## Citation

Please cite the paper/repo if you use the idea or code in this paper/repo.

```
@misc{BSDE-Gen,
  author = {Xingcheng Xu},
  title = {Deep Generative Modeling with Backward Stochastic Differential Equations},
  year = {2023},
  publisher = {arXiv},
  journal = {arXiv preprint},
  howpublished = {\url{https://arxiv.org/abs/2304.xxxxx}},
}
```

```
@misc{BSDE-Gen,
  author = {Xingcheng Xu},
  title = {Deep Generative Modeling with Backward Stochastic Differential Equations},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xingchengxu/BSDE-Gen}},
}
```

