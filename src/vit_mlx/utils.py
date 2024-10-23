import mlx.core as mx


def pos_embed_2d(h: int, w: int, dim: int, temp: int = 10000):
    y, x = mx.meshgrid(mx.arange(h), mx.arange(w))
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos embedding"

    omega = mx.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temp**omega)

    y = y.transpose().flatten()[:, None] * omega[None, :]
    x = x.transpose().flatten()[:, None] * omega[None, :]
    pe = mx.stack([x.sin(), x.cos(), y.sin(), y.cos()], axis=1)
    return pe.reshape(h * w, dim)
