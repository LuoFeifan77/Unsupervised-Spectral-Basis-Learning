# From Feature Learning to Spectral Basis Learning: A Unifying and Flexible Framework for Efficient and Robust Shape Matching has been accepted by CVPR2026!
The code will be released as soon as possible.


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
@inproceedings{luo2026feature,
  title={From Feature Learning to Spectral Basis Learning: A Unifying and Flexible Framework for Efficient and Robust Shape Matching},
  author={Luo, Feifan and Chen, Hongyang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={31377--31388},
  year={2026}
}

```
