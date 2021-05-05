#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]
        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class CA_module(nn.Module):
    def __init__(self, hid_dim, pf_dim, n_heads, dropout):
        super(CA_module, self).__init__()
        self.n_hidden = hid_dim
        self.multi_head_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.FFN = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.layer_norm = nn.LayerNorm(hid_dim)

    def forward(self, inputs):

        z_hat = self.CR_2Datt(inputs)
        z = self.layer_norm(z_hat + inputs)
        outputs_hat = self.FFN(z)
        outputs = self.layer_norm(outputs_hat + z)
        return outputs

    def row_2Datt(self, inputs):
        max_len = inputs.size(1)
        z_row = inputs.reshape(-1, max_len, self.n_hidden)
        z_row, _ = self.multi_head_attention(z_row, z_row, z_row)
        return z_row.reshape(-1, max_len, max_len, self.n_hidden)

    def col_2Datt(self, inputs):
        z_col = inputs.permute(0, 2, 1, 3)
        z_col = self.row_2Datt(z_col)
        z_col = z_col.permute(0, 2, 1, 3)
        return z_col

    def CR_2Datt(self, inputs):
        z_row = self.row_2Datt(inputs)
        z_col = self.col_2Datt(inputs)
        outputs = (z_row + z_col) / 2.
        return outputs