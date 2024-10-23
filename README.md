# Vision Transformer - MLX

---

Implementation of [Vision Transformer](https://openreview.net/pdf?id=YicbFdNTTy) in [MLX](https://github.com/ml-explore/mlx). For further explanation and details on the ViT Architecture check out [Yannic Kilcher's](https://www.youtube.com/watch?v=TrdevFK_am4) video.

## Install

TODO

## Usage

### Original ViT

```python
import mlx.core as mx
from vit_mlx import ViT

v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

img = mx.random.normal((1, 3, 256, 256))
preds = v(img) # (1, 1000)
```

## Parameters

- `image_size`: int | tuple[int, int].
  Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
- `patch_size`: int.
  Size of patches. `image_size` must be divisible by `patch_size`.
  The number of patches is: ` n = (image_size // patch_size) ** 2` and `n` **must be greater than 16**.
- `num_classes`: int.
  Number of classes to classify.
- `dim`: int.
  Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.
- `depth`: int.
  Number of Transformer blocks.
- `heads`: int.
  Number of heads in Multi-head Attention layer.
- `mlp_dim`: int.
  Dimension of the MLP (FeedForward) layer.
- `channels`: int, default `3`.
  Number of image's channels.
- `dropout`: float between `[0, 1]`, default `0.`.
  Dropout rate.
- `emb_dropout`: float between `[0, 1]`, default `0`.
  Embedding dropout rate.
- `pool`: string, either `cls` token pooling or `mean` pooling

## Simple ViT

In an [updated Version](https://arxiv.org/abs/2205.01580) the authors introduced a simplified version of the ViT.
They used a fixed 2d sinusoidal positional encoding instead of learning the positional encoding. They also introduced global average pooling, removed the dropout, increased the batch sizes to 4096, and used of RandAugment and MixUp augmentations.

```python
import mlx.core as mx
from vit_mlx import ViT

v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
)

img = mx.random.normal((1, 3, 256, 256))
preds = v(img) # (1, 1000)
```

## Acknowledgment

---

The original Pytorch implementation from [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains) can be found [here](https://github.com/lucidrains/vit-pytorch)
