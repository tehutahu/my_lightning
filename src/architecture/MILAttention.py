import torch
import torch.nn as nn


class InputLayer(nn.Module):
    def __init__(self, input_dim: int, num_instances: int, embed_dim=384):
        """_summary_

        Args:
            input_dim (int): _description_
            num_instances (int): _description_
            embed_dim (int, optional): _description_. Defaults to 384.
        """
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_instances + 1, embed_dim)
        )

    def forward(self, x):
        x = self.embed(x)
        x = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0), 1, 1)), x], dim=1
        )
        x = x + self.pos_emb
        return x


class MHSAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=8, bias=False, dropout=0.0):
        """_summary_

        Args:
            input_dim (int): _description_
            num_instances (int): _description_
            embed_dim (int, optional): _description_. Defaults to 384.
        """
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x):
        x = x.transpose(1, 0)
        x, attn_weight = self.multi_head_attention(x, x, x)
        x = x.transpose(1, 0)
        return x, attn_weight


class ViTEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        hidden_dim=384 * 4,
        num_heads=8,
        bias=False,
        dropout=0.0,
    ):
        """_summary_

        Args:
            embed_dim (int, optional): _description_. Defaults to 384.
            hidden_dim (_type_, optional): _description_. Defaults to 384*4.
            num_heads (int, optional): _description_. Defaults to 8.
            bias (bool, optional): _description_. Defaults to False.
            dropout (_type_, optional): _description_. Defaults to 0..
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)

        self.multi_head_attention = MHSAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_res, attn_weight = self.multi_head_attention(self.ln1(x))
        x = x + x_res
        x_res = self.mlp(self.ln2(x))
        x = x + x_res
        return x, attn_weight


class MILAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_instances: int,
        embed_dim=384,
        output_dim=2,
        num_blocks=4,
        hidden_dim=384 * 4,
        num_heads=8,
        bias=False,
        dropout=0.0,
    ):
        """_summary_

        Args:
            input_dim (int): _description_
            num_instances (int): _description_
            embed_dim (int, optional): _description_. Defaults to 384.
            hidden_dim (_type_, optional): _description_. Defaults to 384*4.
            num_heads (int, optional): _description_. Defaults to 8.
            bias (bool, optional): _description_. Defaults to False.
            dropout (_type_, optional): _description_. Defaults to 0..
        """
        super().__init__()
        self.input_layer = InputLayer(
            input_dim=input_dim,
            num_instances=num_instances,
            embed_dim=embed_dim,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                ViTEncoderBlock(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    bias=bias,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x):
        B, N = x.shape[:2]
        x = x.view(B, N, -1)
        x = self.input_layer(x)

        attn_weights = []
        for encoder_block in self.encoder_blocks:
            x, w = encoder_block(x)
            attn_weights.append(w)

        x = x[:, 0, :]
        x = self.mlp_head(x)
        return x, attn_weights
