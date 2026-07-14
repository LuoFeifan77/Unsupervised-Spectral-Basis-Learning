# From Feature Learning to Spectral Basis Learning: A Unifying and Flexible Framework for Efficient and Robust Shape Matching has been accepted by CVPR2026!
The code will be released as soon as possible.

# [From Feature Learning to Spectral Basis Learning: A Unifying and Flexible Framework for Efficient and Robust Shape Matching [CVPR 2026]](https://luofeifan77.github.io/publications/)
 [![PDF](https://img.shields.io/badge/PDF-Download-blue)](https://arxiv.org/pdf/2603.23383)
<!--[![ArXiv](https://img.shields.io/badge/arXiv-2312.03678-b31b1b.svg)](https://arxiv.org/abs/2312.03678)-->


## Installation
```bash 
conda create -n contrastivefmnet python=3.8 # create new viertual environment
conda activate contrastivefmnet
conda install pytorch cudatoolkit -c pytorch # install pytorch, cuda==11.8!
pip install -r requirements.txt # install other necessary libraries via pip
```

## Dataset
To train and test datasets used in this paper, please download the datasets from [DongliangCao](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link) and put all datasets under ../data/
```Shell
├── data
    ├── FAUST_r
    ├── FAUST_a
    ├── SCAPE_r
    ├── SCAPE_a
    ├── SHREC19_r
    ├── TOPKIDS
    ├── SMAL_r
    ├── DT4D_r
    ├── SHREC20
    ├── SHREC16
    ├── SHREC16_test
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.

## Data precomputation
```python
python preprocess_dataset.py  
```

## Train
To train the model on a specified dataset.
```python
python train.py --opt options/train/faust.yaml 
```
You can visualize the training process in tensorboard.
```bash
tensorboard --logdir experiments/
```

## Test
To test the model on a specified dataset.
```python
python test.py --opt options/test/faust.yaml 
```
The qualitative and quantitative results will be saved in [results](results) folder.

## Texture Transfer
An example of texture transfer is provided in *[texture_transfer.py](texture_transfer.py)*
```python
python texture_transfer.py
```

## Pretrained models
You can find all pre-trained models in [checkpoints](checkpoints) for reproducibility.

## Results
You can find all matching results in [results](results).

## Acknowledgement
The framework implementation is adapted from [Unsupervised Learning of Robust Spectral Shape Matching](https://github.com/dongliangcao/Unsupervised-Learning-of-Robust-Spectral-Shape-Matching/tree/main?tab=readme-ov-file).\
The feature learning network implementation is adapted from [DiffusionNet](https://github.com/nmwsharp/diffusion-net)


## Attribution
Please cite our paper when using the code. You can use the following bibtex
```
@article{luo2026feature,
  title={From Feature Learning to Spectral Basis Learning: A Unifying and Flexible Framework for Efficient and Robust Shape Matching},
  author={Luo, Feifan and Chen, Hongyang},
  journal={arXiv preprint arXiv:2603.23383},
  year={2026}
}

```

## Contact
If you have any questions, please feel free to contact me via [email](luoff@zju.edu.cn) without any hesitation.
