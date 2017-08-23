# pix2pix-chainer

chainer implementation of [Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf).

- 論文のソースコード：[pix2pix](https://phillipi.github.io/pix2pix/).
- 影響を受けたコード：[pix2pix](https://github.com/mrzhu-cool/pix2pix-pytorch/).
- 影響を受けたコード：[pix2pix](https://github.com/pfnet-research/chainer-pix2pix/).


## Prerequisites

+ Linux
+ Python with numpy
+ chainer
+ todo:gpu対応

## Getting Started

+ Clone this repo:

```
git clone https://github.com/miyamotok0105/pix2pix-chainer.git
cd pix2pix-chainer
```

+ Get dataset

```
unzip dataset/facades.zip
mv facades/ dataset
```


+ Train the model:

```
python train.py
or
python train.py -g 0
```


+ Test the model:

```
TODO
```



