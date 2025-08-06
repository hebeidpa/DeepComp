import torch
from torch import nn
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, ff_dropout=0.1, attn_dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads, dim_head, attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout)))
            ]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class FTTransformer(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous,
        dim=128,
        depth=3,
        heads=4,
        dim_head=32,
        dim_out=1,
        num_special_tokens=2,
        ff_dropout=0.1,
        attn_dropout=0.1
    ):
        super().__init__()
        self.num_categories   = len(categories)
        self.num_continuous   = num_continuous
        self.dim              = dim
        self.num_special_tokens = num_special_tokens

        # Embedding categorical (none used here)
        if self.num_categories > 0:
            self.categorical_embeds = nn.ModuleList([
                nn.Embedding(card, dim) for card in categories
            ])

        # Project continuous
        if self.num_continuous > 0:
            self.numerical_proj = nn.Sequential(
                nn.Linear(num_continuous, dim),
                nn.LayerNorm(dim)
            )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer body
        self.transformer = Transformer(dim, depth, heads, dim_head, ff_dropout, attn_dropout)

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_cont):
        # x_categ ignored in our use-case (pure continuous)
        xs = []
        if self.num_categories > 0 and x_categ is not None:
            embeds = [emb(x_categ[:, i]) for i, emb in enumerate(self.categorical_embeds)]
            x_cat = torch.stack(embeds, dim=1)
            xs.append(x_cat)

        if self.num_continuous > 0:
            x_num = self.numerical_proj(x_cont).unsqueeze(1)
            xs.append(x_num)

        x = torch.cat(xs, dim=1)
        # prepend CLS
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)
        cls_out = x[:, 0]
        return self.head(cls_out)
