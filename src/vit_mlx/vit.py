import mlx.core as mx
import mlx.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def __call__(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dim, heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x):
        x = self.norm(x)
        x = self.attention(x, x, x)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0.0
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = []
        for _ in range(depth):
            self.layers.append(
                [
                    MultiHeadAttention(dim, heads, dropout=dropout),
                    FeedForwardNet(dim, mlp_dim, dropout=dropout),
                ]
            )

    def __call__(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x

        return self.norm(x)


class Rearrange:
    def __init__(self, patch_size: tuple[int, int]):
        self.patch_size = patch_size

    def __call__(self, x):
        patch_height, patch_width = self.patch_size
        b, c, hp1, wp2 = x.shape

        h = hp1 // patch_height
        w = wp2 // patch_width

        # Reshape to separate patch dimensions
        x = x.reshape(b, c, h, patch_height, w, patch_width)
        # Step 2: Permute to bring (p1, p2, c) together
        # Permute from (b, c, h, p1, w, p2) -> (b, h, w, p1, p2, c)
        x = x.transpose([0, 2, 4, 3, 5, 1])
        # Step 3: Reshape to merge (h, w) and (p1, p2, c)
        x = x.reshape(b, h * w, patch_height * patch_width * c)

        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: tuple[int, int] | int,
        patch_size: tuple[int, int] | int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        channels: int = 3,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()

        image_height, image_width = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        patch_height, patch_width = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "image dimensions must be divisible by patch dimensions"
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange((patch_height, patch_width)),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = mx.random.normal((1, num_patches + 1, dim))
        self.cls_token = mx.random.normal((1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, mlp_dim, dropout)

        self.pool = pool
        self.mlp_head = nn.Linear(dim, num_classes)

    def __call__(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = mx.repeat(self.cls_token, b, axis=0)
        x = mx.concatenate([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.encoder(x)

        if self.pool == "cls":
            x = x[:, 0]
        elif self.pool == "mean":
            x = x.mean(axis=1)

        return self.mlp_head(x)
