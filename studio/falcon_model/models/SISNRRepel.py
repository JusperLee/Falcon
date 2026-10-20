import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from sru import SRU
from .base_model import BaseModel
from ..drop_path import DropPath
from ..layers import normalizations, activations
#from fla.layers import Comba as FLAComba

class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        pre_norm_type: str = None,
        pre_act_type: str = None,
        norm_type: str = None,
        act_type: str = None,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        causal: bool = False,  
    ):
        super(ConvNormAct, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan if kernel_size > 0 else self.in_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.pre_norm_type = pre_norm_type
        self.pre_act_type = pre_act_type
        self.norm_type = norm_type
        self.act_type = act_type
        self.xavier_init = xavier_init
        self.bias = bias
        self.is2d = is2d
        self.causal = causal 

        if self.padding is None:
            self.padding = dilation * (kernel_size - 1) // 2
        
        if self.causal:
            self.causal_padding = dilation * (kernel_size - 1)

            if kernel_size > 0:
                conv_class = nn.Conv2d if is2d else nn.Conv1d
                
                self.conv = conv_class(
                    in_channels=self.in_chan,
                    out_channels=self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.causal_padding if not self.is2d else (self.causal_padding, self.padding),
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                )
                if self.xavier_init:
                    nn.init.xavier_uniform_(self.conv.weight)
            else:
                self.conv = nn.Identity()
        else:
            if kernel_size > 0:
                conv_class = nn.Conv2d if is2d else nn.Conv1d
                
                self.conv = conv_class(
                    in_channels=self.in_chan,
                    out_channels=self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                )
                if self.xavier_init:
                    nn.init.xavier_uniform_(self.conv.weight)
            else:
                self.conv = nn.Identity()

        self.pre_norm = normalizations.get(self.pre_norm_type)(self.in_chan)
        self.pre_act = activations.get(self.pre_act_type)()
        self.norm = normalizations.get(self.norm_type)(self.out_chan)
        self.act = activations.get(self.act_type)()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): B, C, T, (F)
        """
        x = self.pre_norm(x)
        x = self.pre_act(x)
        x = self.conv(x)
        if self.causal:
            if self.is2d:
                x = x[:, :, :-self.causal_padding, :]
            else:
                x = x[:, :, :-self.causal_padding]
        x = self.norm(x)
        x = self.act(x)
        return x
    
class ConvActNorm(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        n_freqs: int = -1,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        causal: bool = False,  
    ):
        super(ConvActNorm, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias
        self.is2d = is2d
        self.causal = causal  

        if self.padding is None:
            self.padding = dilation * (kernel_size - 1) // 2
        
        if self.causal:
            self.causal_padding = dilation * (kernel_size - 1)

            if kernel_size > 0:
                conv_class = nn.Conv2d if is2d else nn.Conv1d
                
                self.conv = conv_class(
                    in_channels=self.in_chan,
                    out_channels=self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.causal_padding if not self.is2d else (self.causal_padding, self.padding),
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                )
                if self.xavier_init:
                    nn.init.xavier_uniform_(self.conv.weight)
            else:
                self.conv = nn.Identity()
        else:
            if kernel_size > 0:
                conv_class = nn.Conv2d if is2d else nn.Conv1d
                
                self.conv = conv_class(
                    in_channels=self.in_chan,
                    out_channels=self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                )
                if self.xavier_init:
                    nn.init.xavier_uniform_(self.conv.weight)
            else:
                self.conv = nn.Identity()

        self.act = activations.get(self.act_type)()
        self.norm = normalizations.get(self.norm_type)(
            (self.out_chan, self.n_freqs) if self.norm_type == "LayerNormalization4D" else self.out_chan
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): B, C, T, (F)
        """
        output = self.conv(x)
        if self.causal:
            if self.is2d:
                output = output[:, :, :-self.causal_padding, :]
            else:
                output = output[:, :, :-self.causal_padding]
        output = self.act(output)
        output = self.norm(output)
        return output

# class FiLMBlock(nn.Module):
#     """您提供的 FiLM 风格融合模块"""
#     def __init__(
#         self,
#         in_chan_a: int,
#         in_chan_b: int,
#         kernel_size: int = 1,
#         is2d: bool = False
#     ):
#         super(FiLMBlock, self).__init__()
#         self.in_chan_a = in_chan_a
#         self.in_chan_b = in_chan_b
#         self.kernel_size = kernel_size
#         self.upsampling_depth = 2
#         self.stride = 2
#         self.is2d = is2d

#         # 注意：这里 groups=in_chan_b 要求 2*in_chan_a 能被 in_chan_b 整除。
#         # 您的配置中 in_chan_a=256, in_chan_b=512，2*256=512，完美整除！
#         self.param_gen = nn.Conv1d(in_chan_b, 2 * in_chan_a, kernel_size=1, groups=in_chan_b)
        
#     def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
#         B_, C_, T_, F_= tensor_a.shape
#         gamma_beta = self.param_gen(tensor_b)             # [B, 2*C, T_v]
#         gamma_beta = F.interpolate(gamma_beta, size=T_, mode="nearest")

#         gamma, beta = torch.split(gamma_beta, C_, dim=1)  # [B, C, T_v]
        
#         gamma = gamma.unsqueeze(-1)  # [B, C, T_a, 1]
#         beta  = beta.unsqueeze(-1)   # [B, C, T_a, 1]
        
#         output = tensor_a * gamma + beta  # [B, C, T_a, F]
#         return output

class NeuroSCNetFusion(nn.Module):
    def __init__(self, ain_chan, vin_chan):
        super().__init__()
        if ain_chan != vin_chan:
            groups = math.gcd(vin_chan, ain_chan)
            self.v_proj = nn.Conv1d(vin_chan, ain_chan, kernel_size=1, groups=groups)
        else:
            self.v_proj = nn.Identity()

    def forward(self, audio, video):
        # audio: [B, C_a, T_a, F_]
        # video: [B, C_v, T_v]
        B, C_a, T_a, F_ = audio.shape
        
        # 1. 投影视频通道以匹配音频通道
        v_feat = self.v_proj(video)  # [B, C_a, T_v]
        
        # 2. 对齐时间维度 (如果 T_v 和 T_a 不一致)
        if v_feat.shape[-1] != T_a:
            v_feat = F.interpolate(v_feat, size=T_a, mode="nearest")
            
        # 3. 增加频率维度，使其形状变为 [B, C_a, T_a, 1]
        v_feat = v_feat.unsqueeze(-1)
        
        # 4. 直接相加融合 (利用 PyTorch 的广播机制自动适配 F_ 维度)
        Z_sel = audio + v_feat
        return Z_sel

class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        in_chan: int,
        kernel_size: int,
        norm_type: str = "gLN",
        is2d: bool = False,
        causal: bool = False, 
    ):
        super(InjectionMultiSum, self).__init__()
        self.in_chan = in_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.is2d = is2d
        self.causal = causal  

        self.local_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
            causal=self.causal,  
        )
        self.global_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
            causal=self.causal,  
        )
        self.global_gate = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            act_type="Sigmoid",
            bias=False,
            is2d=self.is2d,
            causal=self.causal, 
        )
    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor):
        old_shape = global_features.shape[-(len(local_features.shape) // 2) :]
        new_shape = local_features.shape[-(len(local_features.shape) // 2) :]

        local_emb = self.local_embedding(local_features)
        if torch.prod(torch.tensor(new_shape)) > torch.prod(torch.tensor(old_shape)):
            global_emb = F.interpolate(self.global_embedding(global_features), size=new_shape, mode="nearest")
            gate = F.interpolate(self.global_gate(global_features), size=new_shape, mode="nearest")
        else:
            g_interp = F.interpolate(global_features, size=new_shape, mode="nearest")
            global_emb =  F.interpolate(self.global_embedding(g_interp), size=new_shape, mode="nearest")
            gate = F.interpolate(self.global_gate(g_interp), size=new_shape, mode="nearest")

        injection_sum = local_emb * gate + global_emb

        return injection_sum

class DualPathRNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        dim: int,
        kernel_size: int = 8,
        stride: int = 1,
        rnn_type: str = "Comba", 
        num_layers: int = 1,
        norm_type: str = "LayerNormalization4D",
        act_type: str = "Tanh",
        bidirectional: bool = True,
        causal: bool = False,
        num_heads: int = 4, 
        head_dim: int = 12, 
        expand_v: float = 2.0,
        use_output_gate: bool = True,
        use_short_conv: bool = True,
        use_output_correction: bool = True,
        use_inner_decay: bool = True,
        **kwargs 
    ):
        super(DualPathRNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.act_type = act_type
        self.bidirectional = bidirectional
        self.causal = causal  

        self.num_direction = int(bidirectional) + 1
        self.rnn_out_chan = self.hid_chan * self.num_direction

        self.norm = normalizations.get(self.norm_type)((self.in_chan, 1) if self.norm_type == "LayerNormalization4D" else self.in_chan)

        self.rnn = getattr(nn, self.rnn_type)(
                input_size=self.in_chan,
                hidden_size=self.hid_chan,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
            )

        # projection back to in_chan when rnn output dim != in_chan
        self.out_proj = (
            nn.Linear(self.rnn_out_chan, self.in_chan, bias=False)
            if self.rnn_out_chan != self.in_chan else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        B, C, T, F_ = x.shape
        residual = x
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous().view(B * F_, C, T)

        if "Comba" in self.rnn_type:
            x = x.permute(0, 2, 1).contiguous()
            x, _, _ = self.rnn(x)
            x = x.permute(0, 2, 1).contiguous()
        else:
            x = x.permute(2, 0, 1).contiguous()
            x, _ = self.rnn(x)
            x = x.permute(1, 2, 0).contiguous()  # [B*F_, rnn_out_chan, T]

        # project rnn output back to in_chan if needed: [B*F_, rnn_out_chan, T] -> [B*F_, C, T]
        x = x.permute(0, 2, 1).contiguous()       # [B*F_, T, rnn_out_chan]
        x = self.out_proj(x)                       # [B*F_, T, C]
        x = x.permute(0, 2, 1).contiguous()        # [B*F_, C, T]

        x = x.view([B, F_, C, T])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + residual

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x
     
class MultiHeadSelfAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "PReLU",
        norm_type: str = "LayerNormalization4D",
        dim: int = 3,
        causal: bool = False,  
    ):
        super(MultiHeadSelfAttention2D, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim
        self.causal = causal  

        assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                    causal=False
                )
            )
            self.Keys.append(
                ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                    causal=False
                )
            )
            self.Values.append(
                ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                    causal=False
                )
            )

        self.attn_concat_proj = ConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
            causal=False
        )

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        batch_size, _, time, freq = x.size()
        residual = x
        
        all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
        all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head

        #attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        #attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]

        if self.causal:
            # Using diagonal mask in streaming case
            T = Q.size(1)
            mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        attn_mat = F.softmax(attn_scores, dim=-1)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B', C/n_head, T, F]
        emb_dim = V.shape[1]  # C/n_head

        x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
        x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]

        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
        x = self.attn_concat_proj(x)  # [B, C, T, F]

        x = x + residual

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_len: int = 10000, *args, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.channels = channels
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.channels, requires_grad=False)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.channels, 2).float() * -(torch.log(torch.tensor(self.max_len).float()) / self.channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_head: int = 8,
        dropout: int = 0.1,
        positional_encoding: bool = True,
        batch_first=True,
        causal: bool = False,  
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_chan = in_chan
        self.n_head = n_head
        self.dropout = dropout
        self.positional_encoding = positional_encoding
        self.batch_first = batch_first
        self.causal = causal  

        assert self.in_chan % self.n_head == 0, "In channels: {} must be divisible by the number of heads: {}".format(
            self.in_chan, self.n_head
        )

        self.norm1 = nn.LayerNorm(self.in_chan)
        self.pos_enc = PositionalEncoding(self.in_chan) if self.positional_encoding else nn.Identity()
        self.attention = nn.MultiheadAttention(self.in_chan, self.n_head, self.dropout, batch_first=self.batch_first)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.in_chan)
        self.drop_path_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        res = x
        if self.batch_first:
            x = x.transpose(1, 2)  # B, C, T -> B, T, C

        x = self.norm1(x)
        x = self.pos_enc(x)
        residual = x

        if self.causal:
            # In the case of streaming, use causal mask
            seq_len = x.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.attention(x, x, x, attn_mask=attn_mask, is_causal=True)[0]  
        else:
            x = self.attention(x, x, x)[0]

        x = self.dropout_layer(x) + residual
        x = self.norm2(x)

        if self.batch_first:
            x = x.transpose(2, 1)  # B, T, C -> B, C, T

        x = self.drop_path_layer(x) + res
        return x 
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        norm_type: str = "gLN",
        act_type: str = "ReLU",
        dropout: float = 0,
        is2d: bool = False,
        causal: bool = False,  
    ):
        super(FeedForwardNetwork, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.act_type = act_type
        self.dropout = dropout
        self.is2d = is2d
        self.causal = causal  

        self.encoder = ConvNormAct(self.in_chan, self.hid_chan, 1, norm_type=self.norm_type, bias=False, is2d=self.is2d, causal=False)  # FC 1
        self.refiner = ConvNormAct(
            self.hid_chan,
            self.hid_chan,
            self.kernel_size,
            groups=self.hid_chan,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=self.causal,  
        )  
        self.decoder = ConvNormAct(self.hid_chan, self.in_chan, 1, norm_type=self.norm_type, bias=False, is2d=self.is2d, causal=False)  # FC 2
        self.dropout_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        res = x
        x = self.encoder(x)
        x = self.refiner(x)
        x = self.dropout_layer(x)
        x = self.decoder(x)
        x = self.dropout_layer(x) + res
        return x

class GlobalAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        pos_enc: bool = True,
        norm_type: str = "gLN",
        causal: bool = False,  
    ):
        super(GlobalAttention, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.pos_enc = pos_enc
        self.causal = causal 

        self.MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc, causal=self.causal,)
        self.FFN = FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=self.dropout, causal=self.causal, norm_type=norm_type)

    def forward(self, x: torch.Tensor):
        x = self.MHSA(x)
        x = self.FFN(x)
        return x

class AudioBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        layers: dict = dict(),
        is2d: bool = False,
        causal: bool = False,
        n_freqs: int = 129, # 新增参数：需要知道初始的频点数来计算后续层的维度
    ):
        super(AudioBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.layers = layers
        self.is2d = is2d
        self.causal = causal  

        self.gateway = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            groups=self.in_chan,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=False,  
        )
        self.projection = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=False, 
        )
        
        # --- 修改 1: 定义 ModuleList 替代单个模块 ---
        self.downsample_layers = nn.ModuleList([])
        self.process_f_layers = nn.ModuleList([])
        self.process_t_layers = nn.ModuleList([])
        self.process_att_layers = nn.ModuleList([])

        # 模拟计算 F 维度的变化
        # 假设 padding = (kernel_size - 1) // 2 (ConvNormAct默认行为)
        # H_out = floor((H_in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
        # 简化后: stride=1 -> F; stride=2 -> floor((F-1)/2 + 1)
        current_f = n_freqs 

        for i in range(self.upsampling_depth):
            # 1. Downsample Layer
            cur_stride = 1 if i == 0 else self.stride
            
            # Set kernel_size to 3 for the first layer, otherwise use the default (5)
            cur_kernel_size = 3 if i == 0 else self.kernel_size 
            
            self.downsample_layers.append(
                ConvNormAct(
                    in_chan=self.hid_chan,
                    out_chan=self.hid_chan,
                    kernel_size=cur_kernel_size,  # Updated to use the conditional variable
                    stride=cur_stride,
                    groups=self.hid_chan,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=False, 
                )
            )
            
            # 更新 F 维度
            if cur_stride > 1:
                # 对应 PyTorch Conv2d 的输出尺寸计算公式
                # ConvNormAct 默认 padding 是 kernel_size // 2 (Same padding for stride 1)
                # 当 stride=2 时，尺寸大约减半
                current_f = (current_f - 1) // cur_stride + 1

            # 2. Process Layers (F, T, Attn) 为每一层实例化
            self.process_f_layers.append(
                DualPathRNN(in_chan=self.hid_chan, **self.layers["layer_1"])
            )
            self.process_t_layers.append(
                DualPathRNN(in_chan=self.hid_chan, causal=self.causal, **self.layers["layer_2"])
            )
            
            # 注意：Layer 3 (Attention) 需要更新 n_freqs 参数
            att_config = self.layers["layer_3"].copy()
            att_config["n_freqs"] = current_f # 传入当前层的 F 维度
            
            self.process_att_layers.append(
                MultiHeadSelfAttention2D(in_chan=self.hid_chan, **att_config)
            )

        # top down fusion 
        self.fusion_layers = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            self.fusion_layers.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=False, 
                )
            )
        
        # Upsampling layers
        self.concat_layers = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            self.concat_layers.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=False, 
                )
            )        
        
        self.residual_conv = ConvNormAct(
            in_chan=self.hid_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            is2d=self.is2d,
            causal=False,  
        )

    def forward(self, x: torch.Tensor):
        # x: B, C, T, (F)
        residual = self.gateway(x)
        x_enc = self.projection(residual)
        
        # ---------------------------
        # 1. Encoder (下采样 + 处理)
        # ---------------------------
        downsampled_outputs = []
        current_input = x_enc

        for i in range(self.upsampling_depth):
            # 下采样
            out = self.downsample_layers[i](current_input)
            
            # 仅在最后一次下采样（即最深层的 Bottleneck）后进行处理
            if i == self.upsampling_depth - 1:
                out = self.process_f_layers[i](out)
                out = self.process_t_layers[i](out)
                out = self.process_att_layers[i](out)
            
            downsampled_outputs.append(out)
            current_input = out

        # ---------------------------
        # 2. Preparation (全局上下文融合)
        # ---------------------------
        # 取最深层的输出作为 Global Context
        global_features = downsampled_outputs[-1]
        
        # 利用 fusion_layers 将全局信息注入到每一层的 skip connection 中
        # x_fused[i] 的形状与 downsampled_outputs[i] 保持一致
        x_fused = [self.fusion_layers[i](downsampled_outputs[i], global_features) for i in range(self.upsampling_depth)]

        # ---------------------------
        # 3. Decoder (上采样 + 共享处理)
        # ---------------------------
        
        # [Step 1: 初始融合] 
        # 将最底层的两层 (Level N-1 和 Level N-2) 进行融合
        # downsampled_outputs[-1] 是最深层, downsampled_outputs[-2] 是倒数第二层
        # 融合后的分辨率将回到 Level N-2
        expanded = self.concat_layers[-1](x_fused[-2], x_fused[-1]) + downsampled_outputs[-2]
        
        # [Step 1: 共享处理]
        # 当前 expanded 处于 Level N-2，因此使用对应层级 (upsampling_depth - 2) 的处理模块
        idx = self.upsampling_depth - 2
        expanded = self.process_f_layers[idx](expanded)
        expanded = self.process_t_layers[idx](expanded)
        expanded = self.process_att_layers[idx](expanded)

        # [Step 2: 逐级上采样]
        # 从 Level N-3 向上遍历到 Level 0
        for i in range(self.upsampling_depth - 3, -1, -1):
            # 融合：将当前 expanded (来自深层) 融合到 Level i (skip connection)
            # expanded 经过插值后分辨率变大，与 x_fused[i] 一致
            expanded = self.concat_layers[i](x_fused[i], expanded) + downsampled_outputs[i]
            
            # [Step 2: 共享处理]
            # 当前 expanded 已恢复到 Level i 的分辨率，再次使用第 i 层的共享模块进行处理
            expanded = self.process_f_layers[i](expanded)
            expanded = self.process_t_layers[i](expanded)
            expanded = self.process_att_layers[i](expanded)

        # ---------------------------
        # 4. Output Projection
        # ---------------------------
        expanded = F.interpolate(expanded, size=residual.shape[-(len(residual.shape) // 2) :], mode="nearest")
        out = self.residual_conv(expanded) + residual

        return out
    
class VideoBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        layers: dict = dict(),
        is2d: bool = False,
        causal: bool = False,  
        num_heads: int = 4, 
        head_dim: int = 12, 
        expand_v: float = 2.0,
        use_output_gate: bool = True,
        use_short_conv: bool = True,
        use_output_correction: bool = True,
        use_inner_decay: bool = True,
    ):
        super(VideoBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.layers = layers
        self.is2d = is2d
        self.causal = causal  

        self.gateway = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            groups=self.in_chan,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=False,  
        )
        self.projection = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=False,  
        )

        self.rnn = nn.LSTM(
                input_size=self.hid_chan,
                hidden_size=self.hid_chan,
                bidirectional=False,
            )
        
        self.residual_conv = ConvNormAct(
            in_chan=self.hid_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            is2d=self.is2d,
            causal=False, 
        )

    def forward(self, x: torch.Tensor):
        residual = self.gateway(x)
        x_enc = self.projection(residual)
        
        x_enc = x_enc.transpose(1, 2).contiguous()
        x_rnn, _ = self.rnn(x_enc)
        x_rnn = x_rnn.transpose(1, 2).contiguous()

        out = self.residual_conv(x_rnn) + residual

        return out
    
class Separator(nn.Module):
    def __init__(
        self,
        audio_params: dict,
        video_params: dict,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_params: dict,
    ):
        super(Separator, self).__init__()
        self.audio_params = audio_params
        self.video_params = video_params
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        
        self.fusion_repeats = self.video_params.pop("repeats", 0)
        self.audio_repeats = self.audio_params.pop("repeats", 0) - self.fusion_repeats
        self.video_repeat = self.fusion_repeats
        
        self.audio_shared = self.audio_params.pop("shared", False)
        self.video_shared = self.video_params.pop("shared", False)
        self.fusion_shared = fusion_params.get("fusion_shared", False)
        
        # Video net and audio net
        if self.audio_shared:
            self.audio_net = AudioBlock(**self.audio_params, in_chan=self.audio_bn_chan)
        else:
            self.audio_net = nn.ModuleList([
                AudioBlock(**self.audio_params, in_chan=self.audio_bn_chan) 
                for _ in range(self.fusion_repeats + self.audio_repeats)
            ])
            
        if self.video_shared:
            self.video_net = VideoBlock(**self.video_params, in_chan=self.video_bn_chan)
        else:
            self.video_net = nn.ModuleList([
                VideoBlock(**self.video_params, in_chan=self.video_bn_chan) 
                for _ in range(self.video_repeat)
            ])
            
        # Simplified CAF Setup
        if self.fusion_shared:
            self.caf = NeuroSCNetFusion(self.audio_bn_chan, self.video_bn_chan)
        else:
            self.caf = nn.ModuleList([
                NeuroSCNetFusion(self.audio_bn_chan, self.video_bn_chan) 
                for _ in range(self.fusion_repeats)
            ])
        
    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio_residual = audio
        video_residual = video
        
        # CAF
        for i in range(self.fusion_repeats):
            if self.audio_shared:
                audio = self.audio_net(audio + audio_residual if i > 0 else audio)
            else:
                audio = self.audio_net[i](audio + audio_residual if i > 0 else audio)
                
            if self.video_shared:
                video = self.video_net(video + video_residual if i > 0 else video)
            else:
                video = self.video_net[i](video + video_residual if i > 0 else video)
                
            fusion_block = self.caf if self.fusion_shared else self.caf[i]
            audio = fusion_block(audio, video)
        
        # RTFSNet
        for j in range(self.audio_repeats):
            i = j + self.fusion_repeats
            if self.audio_shared:
                audio = self.audio_net(audio + audio_residual if i > 0 else audio)
            else:
                audio = self.audio_net[i](audio + audio_residual if i > 0 else audio)
                
        return audio

class STFTEncoder(nn.Module):
    def __init__(
        self,
        win: int,
        hop_length: int,
        out_chan: int = 2,
        kernel_size: int = -1,
        stride: int = 1,
        act_type: str = "ReLU",
        norm_type: str = "gLN",
        bias: bool = False,
        causal: bool = False,  
    ):
        super(STFTEncoder, self).__init__()

        self.win = win
        self.hop_length = hop_length
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_type = act_type
        self.norm_type = norm_type
        self.bias = bias
        self.causal = causal  
        self.eps = torch.finfo(torch.float32).eps  

        self.conv = ConvNormAct(
            in_chan=3,
            out_chan=self.out_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            act_type=self.act_type,
            norm_type=self.norm_type,
            xavier_init=True,
            bias=self.bias,
            is2d=True,
            causal=self.causal,  
        )

        self.register_buffer("window", torch.hann_window(self.win), False)

    def unsqueeze_to_2D(self, x: torch.Tensor):
        if x.ndim == 1:
            return x.reshape(1, -1)
        elif len(s := x.shape) == 3:
            assert s[1] == 1
            return x.reshape(s[0], -1)
        else:
            return x
        
    def get_out_chan(self):
        return self.out_chan
    
    def forward(self, x: torch.Tensor):
        x = self.unsqueeze_to_2D(x)

        # Metal has no general complex STFT support; preserve the exact transform on CPU.
        compute_device = x.device
        stft_x = x.cpu() if x.device.type == "mps" else x
        spec = torch.stft(
            stft_x,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(stft_x.device),
            return_complex=True,
        )
        this_spec = (spec.abs().pow(2) + self.eps).sqrt()  #[1, 1, 251]
        spec = torch.stack([spec.real, spec.imag, this_spec], 1).transpose(2, 3).contiguous()  # B, 3, T, F
        spec = spec.to(compute_device)
        spec_feature_map = self.conv(spec)  # B, C, T, F

        return spec_feature_map
    
class STFTDecoder(nn.Module):
    def __init__(
        self,
        win: int,
        hop_length: int,
        in_chan: int,
        n_src: int,
        kernel_size: int = -1,
        stride: int = 1,
        bias: bool = False,
        causal: bool = False,  
        *args,
        **kwargs,
    ):
        super(STFTDecoder, self).__init__()
        self.win = win
        self.hop_length = hop_length
        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.stride = stride
        self.bias = bias
        self.causal = causal  

        if self.kernel_size > 0:
            self.decoder = nn.ConvTranspose2d(
                in_channels=self.in_chan,
                out_channels=2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            )
            torch.nn.init.xavier_uniform_(self.decoder.weight)
        else:
            self.decoder = nn.Identity()

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x: torch.Tensor, input_shape: torch.Size):
        # B, n_src, N, T, F

        batch_size, length = input_shape[0], input_shape[-1]

        x = x.view(batch_size * self.n_src, self.in_chan, *x.shape[-2:])  # B, n_src, N, T, F -> # B * n_src, N, T, F

        if self.causal:
            pad = (0, 0, self.kernel_size - 1, 0)
            x = F.pad(x, pad)

        decoded_separated_audio = self.decoder(x)  # B * n_src, N, T, F - > B * n_src, 2, T, F

        if self.causal:
            decoded_separated_audio = decoded_separated_audio[:, :, :-self.padding, :]

        # Move real channels before constructing a complex tensor on Metal.
        output_device = decoded_separated_audio.device
        if output_device.type == "mps":
            decoded_separated_audio = decoded_separated_audio.cpu()
        spec = torch.complex(decoded_separated_audio[:, 0], decoded_separated_audio[:, 1])  # B*n_src, T, F
        # spec = torch.stack([spec.real, spec.imag], dim=-1)  # B*n_src, T, F
        spec = spec.transpose(1, 2).contiguous()  # B*n_src, F, T
        output = torch.istft(
            spec,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(spec.device),
            length=length,
        )  # B*n_src, L

        output = output.to(output_device)
        output = output.view(batch_size, self.n_src, length)  # B, n_src, L

        return output
    
class MaskGenerator(nn.Module):
    def __init__(
        self,
        n_src: int,
        audio_emb_dim: int,
        bottleneck_chan: int,
        kernel_size: int = 1,
        mask_act: str = "ReLU",
        RI_split: bool = False,
        output_gate: bool = False,
        dw_gate: bool = False,
        direct: bool = False,
        is2d: bool = False,
        causal: bool = False,  
        *args,
        **kwargs,
    ):
        super(MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.mask_act = mask_act
        self.output_gate = output_gate
        self.dw_gate = dw_gate
        self.RI_split = RI_split
        self.direct = direct
        self.is2d = is2d
        self.causal = causal 

        if not self.direct:
            mask_output_chan = self.n_src * self.in_chan

            self.mask_generator = nn.Sequential(
                nn.PReLU(),
                ConvNormAct(
                    self.bottleneck_chan,
                    mask_output_chan,
                    self.kernel_size,
                    act_type=self.mask_act,
                    is2d=self.is2d,
                    causal=self.causal,  
                ),
            )

            if self.output_gate:
                groups = mask_output_chan if self.dw_gate else 1
                self.output = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Tanh", is2d=self.is2d, groups=groups, causal=self.causal,)
                self.gate = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Sigmoid", is2d=self.is2d, groups=groups, causal=self.causal,)

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        batch_size = audio_mixture_embedding.size(0)
        dims = audio_mixture_embedding.shape[-(len(audio_mixture_embedding.shape) // 2) :]
        if self.RI_split:
            masks = masks.view(batch_size, self.n_src, 2, self.in_chan // 2, *dims)
            audio_mixture_embedding = audio_mixture_embedding.view(batch_size, 2, self.in_chan // 2, *dims)

            mask_real = masks[:, :, 0]  # B, n_src, C/2, T, (F)
            mask_imag = masks[:, :, 1]  # B, n_src, C/2, T, (F)
            emb_real = audio_mixture_embedding[:, 0].unsqueeze(1)  # B, 1, C/2, T, (F)
            emb_imag = audio_mixture_embedding[:, 1].unsqueeze(1)  # B, 1, C/2, T, (F)

            est_spec_real = emb_real * mask_real - emb_imag * mask_imag  # B, n_src, C/2, T, (F)
            est_spec_imag = emb_real * mask_imag + emb_imag * mask_real  # B, n_src, C/2, T, (F)

            separated_audio_embedding = torch.cat([est_spec_real, est_spec_imag], 2)  # B, n_src, C, T, (F)
        else:
            masks = masks.view(batch_size, self.n_src, self.in_chan, *dims)
            separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)

        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        if self.direct:
            return refined_features
        else:
            masks = self.mask_generator(refined_features)
            if self.output_gate:
                masks = self.output(masks) * self.gate(masks)
            separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)

            return separated_audio_embedding

class SISNRRepel(BaseModel):
    def __init__(
        self,
        n_src: int,
        enc_dec_params: dict,
        audio_bn_params: dict,
        audio_params: dict,
        mask_generation_params: dict,
        pretrained_vout_chan: int = -1,
        video_bn_params: dict = dict(),
        video_params: dict = dict(),
        fusion_params: dict = dict(),
        sample_rate=16000,
        causal: bool = False,
        mdcl_margin: float = 0.0,    # SI-SNR threshold (dB): penalise if SI-SNR(ŝ, s_int) > margin
        mdcl_alpha: float = 0.15,    # weight for contrastive repulsion loss
    ):
        super(SISNRRepel, self).__init__(sample_rate=sample_rate)
        self.n_src = n_src
        self.pretrained_vout_chan = pretrained_vout_chan
        self.audio_bn_params = audio_bn_params
        self.video_bn_params = video_bn_params
        self.enc_dec_params = enc_dec_params
        self.audio_params = audio_params
        self.video_params = video_params
        self.fusion_params = fusion_params
        self.mask_generation_params = mask_generation_params
        self.causal = causal
        self.mdcl_margin = mdcl_margin
        self.mdcl_alpha = mdcl_alpha

        self.encoder = STFTEncoder(**self.enc_dec_params)
        encoder_out_chan = self.encoder.get_out_chan()

        self.audio_bottleneck = ConvNormAct(**self.audio_bn_params, in_chan=encoder_out_chan)
        self.video_bottleneck = ConvNormAct(**self.video_bn_params, in_chan=self.pretrained_vout_chan)

        self.audio_params['causal'] = self.causal
        self.video_params['causal'] = self.causal

        self.separator = Separator(
            audio_params=self.audio_params,
            video_params=self.video_params,
            audio_bn_chan=self.audio_bn_params["out_chan"],
            video_bn_chan=self.pretrained_vout_chan,
            fusion_params=self.fusion_params,
        )

        self.mask_generator = MaskGenerator(
            **self.mask_generation_params,
            n_src=self.n_src,
            audio_emb_dim=encoder_out_chan,
            bottleneck_chan=self.audio_bn_params["out_chan"],
        )

        self.decoder = STFTDecoder(
            **self.enc_dec_params,
            in_chan=encoder_out_chan * self.n_src,
            n_src=self.n_src,
        )

        # Hann window for MDCL STFT, registered as non-persistent buffer
        self.register_buffer(
            "mdcl_window",
            torch.hann_window(self.enc_dec_params["win"]),
            persistent=False,
        )

    def _encode_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """
        把任意波形 wav [B, T] 编码为 MaskGenerator 的输入 refined_features，
        使用零视觉特征（视觉分支对 near 音频没有意义）。

        Returns: refined_features [B, C, T', F']
        """
        emb = self.encoder(wav)                        # [B, N, T', F']
        audio_feat = self.audio_bottleneck(emb)        # [B, C, T', F']
        # 视觉侧用全零占位（near 音频没有口型）
        dummy_video = self.video_bottleneck(
            wav.new_zeros(wav.size(0), self.pretrained_vout_chan, 1)
        )                                              # [B, C_v, 1] or [B, C_v, 1]
        refined = self.separator(audio_feat, dummy_video)
        return refined

    @staticmethod
    def _sisnr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        SI-SNR between est and ref, both [B, T].
        Returns [B] in dB (higher = more similar).
        """
        est = est - est.mean(dim=-1, keepdim=True)
        ref = ref - ref.mean(dim=-1, keepdim=True)
        dot = (est * ref).sum(dim=-1, keepdim=True)
        ref_energy = (ref ** 2).sum(dim=-1, keepdim=True) + eps
        proj = dot * ref / ref_energy                    # scaled target
        noise = est - proj
        sisnr = 10 * torch.log10(
            (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
        )
        return sisnr                                     # [B]

    def _mag_sim(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Phase-agnostic similarity using STFT magnitude spectra.

        Replaces waveform SI-SNR for MDCL when near-speaker audio is reverberant.
        Reverb primarily distorts phase; magnitude spectra retain speaker-identity
        features while suppressing room-acoustic correlation artifacts.

        Applies the same SI-SNR formula to flattened magnitude spectrograms,
        so the dB scale and margin semantics are preserved.

        Args:
            est: [B, T]
            ref: [B, T]
        Returns:
            [B] scores in dB (higher = estimate more spectrally similar to ref).
        """
        win = self.enc_dec_params["win"]
        hop = self.enc_dec_params["hop_length"]

        def get_mag(wav):  # [B, T] → [B, F*T_frames]
            spec = torch.stft(
                wav,
                n_fft=win,
                hop_length=hop,
                window=self.mdcl_window,
                return_complex=True,
            )  # [B, F, T_frames]
            return spec.abs().flatten(start_dim=1)  # [B, F*T_frames]

        return self._sisnr(get_mag(est), get_mag(ref))  # [B]

    def _mdcl_loss_hard(
        self,
        est_target: torch.Tensor,
        near_audio,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Hard Negative MDCL — magnitude-domain repulsion with importance weighting.

        Uses STFT magnitude similarity (_mag_sim) instead of waveform SI-SNR to
        avoid gradient conflicts caused by room-acoustic correlation when near
        speakers are reverberant (same RIR as target → shared spectral coloring
        → waveform SI-SNR gradient conflicts with PIT gradient).

        L = Σ_k q_β(k) · relu( mag_sim(ŝ, s_int^k) - margin )
        where  q_β(k) = softmax_k( β · mag_sim(ŝ, s_int^k) )

        Special cases:
          K=0 → zero loss  (no near speakers)
          K=1 → single interferer, weight=1
          β=0 → uniform weights

        Args:
            est_target:  [B, T]  – estimated target waveform (has grad)
            near_audio:  list of [B, T]  – near-speaker waveforms (may be reverberant)
            beta:        concentration parameter (≥0); larger = harder focus

        Returns:
            Scalar repulsion loss.
        """
        interferers = near_audio if isinstance(near_audio, list) else [near_audio]
        K = len(interferers)

        if K == 0:
            return est_target.new_zeros(1)

        if K == 1:
            sim = self._mag_sim(est_target, interferers[0])        # [B]
            return F.relu(sim - self.mdcl_margin).mean()

        # ── K ≥ 2: hard-negative weighting ──────────────────────────────────
        # sim_scores: [K, B]  — magnitude-domain similarity scores
        sim_scores = torch.stack(
            [self._mag_sim(est_target, itf) for itf in interferers], dim=0
        )

        # q_β(k) = softmax along K dim; β=0 → uniform (1/K)
        weights = torch.softmax(beta * sim_scores, dim=0)          # [K, B]

        penalties = F.relu(sim_scores - self.mdcl_margin)          # [K, B]

        # weighted sum over K, then mean over B
        return (weights * penalties).sum(dim=0).mean()

    def forward(self, audio_mixture: torch.Tensor, mouth_embedding: torch.Tensor = None,
                target_audio: torch.Tensor = None, near_audio: torch.Tensor = None):
        audio_mixture_embedding = self.encoder(audio_mixture)  # B, 1, L -> B, N, T, (F)

        audio = self.audio_bottleneck(audio_mixture_embedding)  # B, C, T, (F)
        video = self.video_bottleneck(mouth_embedding)  # B, N2, T2, (F2) -> B, C2, T2, (F2)

        refined_features = self.separator(audio, video)  # B, C, T, (F)

        separated_audio_embeddings = self.mask_generator(refined_features, audio_mixture_embedding)  # B, n_src, N, T, (F)
        separated_audios = self.decoder(separated_audio_embeddings, audio_mixture.shape)  # B, n_src, L

        # MDCL repulsion loss – only during training when near audio is provided
        if self.training and near_audio is not None:
            # est for the single target source [B, T]
            est_target = separated_audios[:, 0, :]
            mdcl_loss = self.mdcl_alpha * self._mdcl_loss(est_target, near_audio)
            return separated_audios, mdcl_loss

        return separated_audios

    def get_model_args(self):
        model_args = {"n_src": self.n_src}
        return model_args
