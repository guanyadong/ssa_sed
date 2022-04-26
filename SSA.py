from sparsemax import Sparsemax
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import constant_


class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.2, normal_func="sparsemax", sparsity=1.):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, normal_func=normal_func,sparsity=sparsity)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src: Tensor) -> Tensor:
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.,normal_func=None, sparsity=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.normal_func = normal_func
        self.sparsity = sparsity
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value):

        return multi_head_attention_forward(
            query, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            out_proj_weight = self.out_proj.weight,
            out_proj_bias = self.out_proj.bias,
            normal_func=self.normal_func,
            sparsity=self.sparsity,
        )



def multi_head_attention_forward(query,  # type: Tensor
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 normal_func="sparsemax",
                                 sparsity=1,
                                 ):
    tgt_len, bsz, embed_dim = query.size()

    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5

    # self-attention
    q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))

    if normal_func == "sparsemax":
        sparsemax = Sparsemax(dim=2)
        attn_output_weights = sparsemax(attn_output_weights / 1.3)

    elif normal_func == "softmax":
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output
