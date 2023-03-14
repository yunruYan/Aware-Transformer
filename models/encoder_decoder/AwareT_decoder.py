import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.config import cfg


# 位置嵌入矩阵
def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=512,
            depth=3,
            num_heads=8,
            dropout=0.1,
            ff_dropout=0.1,
            use_gx=False
    ):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        self.use_gx = use_gx
        for i in range(depth):
            sublayer = DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                ff_dropout=ff_dropout,
                use_gx=use_gx
            )
            self.layers.append(sublayer)

        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED)

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.pos_embed = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(100, self.embed_dim, 0), freeze=True
        )

        self.generator = nn.Linear(self.embed_dim, self.vocab_size, bias=True)

        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.seq_len = 0
        for layer in self.layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        for layer in self.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        for layer in self.layers:
            layer.apply_to_states(fn)

    def precompute(self, encoder_out):
        p_att_feats = []
        for layer in self.layers:
            key, value2 = layer.precompute(encoder_out)
            p_att_feats.append((key, value2))
        return p_att_feats

    def forward(self, gx, seq,s_att, seq_mask=None, att_mask=None):
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)  # [B, 1, M]

        seq_len = seq.size()[1]
        pos_indx = torch.arange(1, seq_len + 1, device='cuda').view(1, -1)
        if self.seq_len is not None:
            seq_len = self.seq_len + seq_len
            self.seq_len = seq_len
            pos_indx = torch.arange(seq_len, seq_len + 1, device='cuda').view(1, -1)

        # 词汇嵌入 + 位置嵌入
        # [B, seq_len, C] for training or [B, 1, C] for inference
        x = self.embed_scale * self.word_embed(seq) + self.pos_embed(pos_indx)

        for layer in self.layers:
            x = layer(gx, x, s_att, seq_mask, att_mask)

        x = self.dropout(x)
        out = self.generator(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ff_dropout=0.1, use_gx=False):
        super(DecoderLayer, self).__init__()
        self.word_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.cross_att = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ff_layer = FeedForward(
            embed_dim=embed_dim,
            ffn_embed_dim=embed_dim * 4,
            relu_dropout=ff_dropout
        )
        self.layer_norm3 = torch.nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.use_gx = use_gx
        if self.use_gx:
            # 方式2，concat接Linear / Linear+GLU
            self.fuse_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.fuse_layer_norm = nn.LayerNorm(embed_dim)
        # self.cov_layer=nn.Conv1d(24,out_channels=144,kernel_size=1)
        #     self.fuse_layer_2 = nn.Sequential(
        #         nn.Linear(embed_dim, embed_dim*2),
        #         nn.ReLU(),
        #         nn.Linear(embed_dim*2,embed_dim),
        #         nn.ReLU(),
        #         nn.Dropout(0.1)
        #     )
        #     self.fuse_layer_norm_2 = nn.LayerNorm(embed_dim)

    def apply_to_states(self, fn):
        self.word_attn.apply_to_states(fn)

    def init_buffer(self, batch_size):
        self.word_attn.init_buffer(batch_size)

    def clear_buffer(self):
        self.word_attn.clear_buffer()

    def precompute(self, encoder_out):
        # key, value2 = self.cross_att.precompute(encoder_out, encoder_out)
        # return key, value2
        pass

    def forward(self, gx, x,s_att, seq_mask, att_mask=None):
        # 单词嵌入自注意力
        # short_cut = x
        # 在单词嵌入自注意力阶段，嵌入图像的全局特征
        # 方式2:concat接Linear+GLU / Linear
        '''
        此处为新加的代码，language_x记录融合后前的语义信息
        '''
        # language_x=x

        if self.use_gx:
            x_cat = torch.cat([x, gx.unsqueeze(1).expand_as(x)], dim=-1)
            x = self.fuse_layer(x_cat) + x
            x = self.fuse_layer_norm(x)
        short_cut = x
        '''
        此处为新加的代码
        '''

        # language_x=self.cov_layer(language_x)
        # encoder_out = torch.cat([encoder_out, language_x], dim=-2)
        # encoder_out = self.fuse_layer_2(encoder_out)
        # encoder_out = self.fuse_layer_norm_2(encoder_out)

        x = self.word_attn(
            q=x,
            k=x,
            v=x,
            mask=seq_mask
        )
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)

        # 单词嵌入与图像特征（可包含全局特征）cross 注意力
        short_cut = x
        if self.use_gx:
            '''这里有很多种选择：spatialchannel并行,channel-spatial,spatial-channel'''
            k = torch.cat([s_att, gx.unsqueeze(1)], 1)
            v = torch.cat([s_att, gx.unsqueeze(1)], 1)
            # kv = torch.cat([encoder_out, gx.unsqueeze(1),language_x], 1)  #改动点，加入了未经处理的语义信息
            if att_mask is not None:
                # [B, 1, M+1]，对于grid特征，直接设置为None亦可
                _att_mask = torch.cat(
                    [att_mask, torch.ones(att_mask.size(0), device='cuda').unsqueeze(1).unsqueeze(1)], 2
                ).long()
                # _att_mask = torch.cat(
                #     [att_mask, torch.ones([att_mask.size(0), 31], device='cuda').unsqueeze(1)], 2
                # ).long()
            else:
                _att_mask = None
        else:
            k = s_att
            v = s_att
            _att_mask = att_mask

        x = self.cross_att(
            q=x,
            k=k,
            v=v,
            mask=_att_mask,
            # precompute=False
        )
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)

        # Feedforward
        short_cut = x
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = self.layer_norm3(x + short_cut)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.o_linear = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(-1)

        self.clear_buffer()

    def init_buffer(self, batch_size):
        # [B, nH, 0, C/nH]
        self.buffer_key = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')
        self.buffer_value = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')

    def clear_buffer(self):
        self.buffer_key = None
        self.buffer_value = None

    def apply_to_states(self, fn):
        self.buffer_key = fn(self.buffer_key)
        self.buffer_value = fn(self.buffer_value)

    def forward(self, q, k, v, mask):
        """
        Decoder部分有两部分进行注意力：
            1）单词嵌入自注意力，q/k/v大小均为[B, L, D]
            2）单词嵌入与图像特征（包含全局特征）的cross attention，q的大小为[B, L, D]
               k/v的大小为[B, M+1, D]
        输出的维度大小只与q的维度大小相关
        """
        B_, N, C = q.size()
        # 线性变换
        q = self.q_linear(q).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 存储buffer，用于inference时单词嵌入自注意力
        if self.buffer_key is not None and self.buffer_value is not None:
            self.buffer_key = torch.cat([self.buffer_key, k], dim=2)
            self.buffer_value = torch.cat([self.buffer_value, v], dim=2)
            k = self.buffer_key
            v = self.buffer_value

        # 注意力核心操作
        # [B, nH, L, L] or [B, nH, L, M+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 计算注意力权重
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.o_linear(out)
        return out


'''
带有记忆向量的多头注意力机制
'''


class MemoryMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_memory=30):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_memory = num_memory
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.o_linear = nn.Linear(embed_dim, embed_dim)
        self.k_memory = Variable(torch.normal(0, 1 / self.head_dim, size=(1, self.num_memory, self.embed_dim)).cuda(),
                                 requires_grad=True)
        self.v_memory = Variable(torch.normal(0, 1 / self.num_memory, size=(1, self.num_memory, self.embed_dim)).cuda(),
                                 requires_grad=True)
        self.k_w = nn.Linear(embed_dim, embed_dim)
        self.v_w = nn.Linear(embed_dim, embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(-1)
        self.clear_buffer()

    def init_buffer(self, batch_size):
        # [B, nH, 0, C/nH]
        self.buffer_key = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')
        self.buffer_value = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')

    def clear_buffer(self):
        self.buffer_key = None
        self.buffer_value = None

    def apply_to_states(self, fn):
        self.buffer_key = fn(self.buffer_key)
        self.buffer_value = fn(self.buffer_value)

    def forward(self, q, k, v, mask):
        """
        Decoder部分有两部分进行注意力：
            1）单词嵌入自注意力，q/k/v大小均为[B, L, D]
            2）单词嵌入与图像特征（包含全局特征）的cross attention，q的大小为[B, L, D]
               k/v的大小为[B, M+1, D]
        输出的维度大小只与q的维度大小相关
        """
        B_, N, C = q.size()
        # 记忆向量的增加
        k_memory = self.k_w(self.k_memory)
        v_memory = self.v_w(self.v_memory)
        k = torch.cat([k, k_memory.repeat(B_, 1, 1)], dim=1)  # (batch_size,seq_len_k+num_memory,d_model)
        v = torch.cat([v, v_memory.repeat(B_, 1, 1)], dim=1)  # (batch_size,seq_len_k+num_memory,d_model)
        # 线性变换
        q = self.q_linear(q).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 存储buffer，用于inference时单词嵌入自注意力
        if self.buffer_key is not None and self.buffer_value is not None:
            self.buffer_key = torch.cat([self.buffer_key, k], dim=2)
            self.buffer_value = torch.cat([self.buffer_value, v], dim=2)
            k = self.buffer_key
            v = self.buffer_value

        # 注意力核心操作
        # [B, nH, L, L] or [B, nH, L, M+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 计算注意力权重
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.o_linear(out)
        return out


# 不包含残差连接和LayerNorm
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x