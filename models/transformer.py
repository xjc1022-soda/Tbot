import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = self.pe[:,:x.size(1)]
        return x



def get_attn_pad_mask(seq_q, seq_k): # Mask Pad 
    '''
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    #eq(zero) is Pad token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) #[batch_size,1, len_k] # True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)#[batch_size, len_q, len_k]



class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_head, len_q, d_k]
        K: [batch_size, n_head, len_k, d_k]
        V: [batch_size, n_head, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_head, seq_len, seq_len]
        '''
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_head, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_head, len_q, d_v]
        return torch.matmul(attn, V), attn
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # print(batch_size)
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1,2)

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        output = self.fc(context)  
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]
    
class EncoderHier(nn.Module):
    def __init__(self, n_layers, d_model, d_k, d_v, d_ff, n_head):
        super(EncoderHier, self).__init__()
        self.n_layers = n_layers
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        attns = []
        for _ in range(self.n_layers):
            enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_outputs: [batch_size, src_len, d_model]
            attns.append(attn)
            enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
            enc_inputs = enc_outputs
        return enc_outputs, attn

class Merger(nn.Module):
    def __init__(self, d_model):
        super(Merger, self).__init__()
        self.fc = nn.Linear(2 * d_model, d_model, bias=False)
    def forward(self, enc_outputs):
        n_patch = enc_outputs.size(1)
        if n_patch == 1:
            return enc_outputs
        else:
            if n_patch % 2 == 0:
                idx_1 = torch.arange(0, n_patch, 2)
                idx_2 = torch.arange(1, n_patch, 2)
                assert idx_1.size(0) == idx_2.size(0)
                enc_outputs_1 = enc_outputs[:, idx_1, :]
                enc_outputs_2 = enc_outputs[:, idx_2, :]
                merge_outputs = self.fc(torch.cat([enc_outputs_1, enc_outputs_2], dim=-1))
            else:
                idx_1 = torch.arange(0, n_patch-1, 2)
                idx_2 = torch.arange(1, n_patch-1, 2)
                assert idx_1.size(0) == idx_2.size(0)
                enc_outputs_1 = torch.cat([enc_outputs[:, idx_1, :], enc_outputs[:, -1, :].unsqueeze(1)], dim=1)
                enc_outputs_2 = torch.cat([enc_outputs[:, idx_2, :], enc_outputs[:, -1, :].unsqueeze(1)], dim=1)
                merge_outputs = self.fc(torch.cat([enc_outputs_1, enc_outputs_2], dim=-1))
            return merge_outputs # [batch_size, src_len, d_model]

class Merger_pool(nn.Module):
    def __init__(self):
        super(Merger_pool, self).__init__()
        self.pool = nn.AvgPool1d(2, padding=0)
    def forward(self, enc_outputs):
        return self.pool(enc_outputs.transpose(1, 2)).transpose(1, 2)

class TranEncoder(nn.Module):
    def __init__(self, patch_size, d_model, n_hierarchy, n_layers, d_k, d_v, d_ff, n_head, dropout):
        super(TranEncoder, self).__init__()
        self.value_embedding = nn.Linear(patch_size, d_model, bias=False)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.hiers = nn.ModuleList([EncoderHier(n_layers, d_model, d_k, d_v, d_ff, n_head) for _ in range(n_hierarchy)])
        self.merger = Merger_pool()
    def forward(self, embbings): # enc_inputs: [batch_size, src_len]
        embbings = self.value_embedding(embbings) + self.position_embedding(embbings)
        embbings = self.dropout(embbings)
        enc_self_attns = []
        enc_outputs_list = []
        for hier in self.hiers:
            embbings, enc_self_attn = hier(embbings, None)
            enc_self_attns.append(enc_self_attn)
            enc_outputs_list.append(embbings)
            # print(embbings.size())
            embbings = self.merger(embbings)
        return enc_outputs_list, enc_self_attns 

class TranDecoder(nn.Module):
    def __init__(self, patch_size, d_model, n_hierarchy, d_k, d_v, d_ff, n_head):
        super(TranDecoder, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.dec_cross_attn = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.project = nn.Linear(d_model, patch_size, bias=False)
        self.merger = Merger_pool()
        self.n_hierarchy = n_hierarchy
    def forward(self, m_query, enc_outputs_list): # dec_inputs: [batch_size, tgt_len]
        z_list = []
        pred_list = []
        for i in range(self.n_hierarchy):
            K, V = enc_outputs_list[i], enc_outputs_list[i]
            z , _ = self.dec_cross_attn(m_query, K, V, None) # [batch_size, tgt_len, d_model]
            z = self.pos_ffn(z)
            z , _ = self.dec_self_attn(z, z, z, None) # [batch_size, tgt_len, d_model]
            z = self.pos_ffn(z)
            pred = self.project(z)
            z_list.append(z)
            pred_list.append(pred)
            m_query = self.merger(m_query)
        return z_list, pred_list
    

class Patching(nn.Module):
    def __init__(self, patch_size, stride, padding):
        super(Patching, self).__init__()
        # Patching
        self.patch_len = patch_size
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        # self.value_embedding = nn.Linear(patch_size, d_model, bias=False)

        # # Positional embedding
        # self.position_embedding = PositionalEncoding(d_model)

        # # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        x = self.padding_patch_layer(x.transpose(1, 2)).transpose(1, 2)
        # print(x.size())
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride).transpose(2,3)
        # print(x.size())
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        return x