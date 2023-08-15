# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional as F
from fastNLP import seq_len_to_mask
from basic import PositionwiseFeedForward, PositionalEncoding, TimeEncoding, max_pooling, mean_pooling
import random


def time_activation(x):
    return F.softplus(x)
    # return torch.sigmoid(x)
    # return torch.relu(x)


class IdentityActivation(nn.Module):
    def forward(self, x):
        return x


class Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(Embedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.pos = TimeEncoding(hidden_dim)

    def forward(self, x, times):
        x = self.fc(x)
        return self.dropout(x + self.pos(times))


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.embed = Embedding(input_dim, hidden_dim, dropout)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, layers,
                          batch_first=True, bidirectional=False, dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * layers, hidden_dim * layers)
        self.final = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

    def forward(self, statics, dynamics, priv, nex, mask, times, seq_len):
        bs, max_len, _ = dynamics.size()
        # bs, max_len, statics_dim
        x = statics.unsqueeze(1).expand(-1, max_len, -1)
        # bs, max_len, dynamics_dim*3 + statics_dim
        x = torch.cat([x, dynamics, priv, mask], dim=-1)
        x = self.embed(x, times)  # bs, max_len, hidden_dim
        # x = dynamics

        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)  # h num_layers, bs, hidden_dim
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True)  # bs, max_len, hidden_dim
        # h, c = h
        # layers, 1, bs, hidden_dim
        h = h.view(self.layers, -1, bs, self.hidden_dim)
        h1, _ = max_pooling(out, seq_len)  # bs, hidden_dim
        h2 = mean_pooling(out, seq_len)  # bs, hidden_dim
        h3 = h[-1].view(bs, -1)  # bs, hidden_dim
        glob = torch.cat([h1, h2, h3], dim=-1)  # bs, hidden_dim*3
        # bs, hidden_dim [s in paper]
        glob = self.final(self.fc(self.drop(glob)))

        # hf = h[:,0]
        # hb = h[:,1]
        # lasth = self.final(self.fc1(torch.cat([hf,hb], dim=-1)))
        lasth = h.view(-1, bs, self.hidden_dim)  # layers, bs, hidden_dim

        lasth = lasth.permute(1, 0, 2).contiguous().view(
            bs, -1)  # bs, layers*hidden_dim
        lasth = self.final(self.fc1(self.drop(lasth)))
        # bs, hidden_dim*(layers+1)   r = [s, {h} in paper]
        hidden = torch.cat([glob, lasth], dim=-1)
        return hidden


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, layers, dropout):
        super(VariationalEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.embed = Embedding(input_dim, hidden_dim, dropout)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, layers,
                          batch_first=True, bidirectional=False, dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * layers, hidden_dim * layers)
        self.final = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.encoder_mu = nn.Linear(hidden_dim*4, hidden_dim*4)
        self.encoder_logvar = nn.Linear(hidden_dim*4, hidden_dim*4)

    def forward(self, statics, dynamics, priv, nex, mask, times, seq_len):
        bs, max_len, _ = dynamics.size()
        # bs, max_len, statics_dim
        # bs, max_len, statics_dim
        x = statics.unsqueeze(1).expand(-1, max_len, -1)
        # bs, max_len, dynamics_dim*3 + statics_dim
        x = torch.cat([x, dynamics, priv, mask], dim=-1)
        x = self.embed(x, times)  # bs, max_len, hidden_dim
        # x = dynamics

        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)  # h num_layers, bs, hidden_dim
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True)  # bs, max_len, hidden_dim
        # h, c = h
        # layers, 1, bs, hidden_dim
        h = h.view(self.layers, -1, bs, self.hidden_dim)
        h1, _ = max_pooling(out, seq_len)  # bs, hidden_dim
        h2 = mean_pooling(out, seq_len)  # bs, hidden_dim
        h3 = h[-1].view(bs, -1)  # bs, hidden_dim
        glob = torch.cat([h1, h2, h3], dim=-1)  # bs, hidden_dim*3
        # bs, hidden_dim [s in paper]
        glob = self.final(self.fc(self.drop(glob)))

        # hf = h[:,0]
        # hb = h[:,1]
        # lasth = self.final(self.fc1(torch.cat([hf,hb], dim=-1)))
        lasth = h.view(-1, bs, self.hidden_dim)  # layers, bs, hidden_dim

        lasth = lasth.permute(1, 0, 2).contiguous().view(
            bs, -1)  # bs, layers*hidden_dim
        lasth = self.final(self.fc1(self.drop(lasth)))
        # bs, hidden_dim*(layers+1)   r = [s, {h} in paper]
        hidden = torch.cat([glob, lasth], dim=-1)

        mu = self.encoder_mu(hidden)
        logvar = self.encoder_logvar(hidden)
        return mu, logvar


class TransformerVariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, layers, dropout):
        super(TransformerVariationalEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.embed = Embedding(input_dim, hidden_dim, dropout)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, layers,
                          batch_first=True, bidirectional=False, dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * layers, hidden_dim * layers)
        self.final = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.encoder_mu = nn.Linear(hidden_dim*4, hidden_dim*4)
        self.encoder_logvar = nn.Linear(hidden_dim*4, hidden_dim*4)

        num_heads = 4
        num_layers = 2
        z_dim = 128
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout,            batch_first=True
            ),
            num_layers=num_layers,
        )
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim*4)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim*4)

    def forward(self, statics, dynamics, priv, nex, mask, times, seq_len, dt=None):
        bs, max_len, _ = dynamics.size()
        # bs, max_len, statics_dim
        # bs, max_len, statics_dim
        x = statics.unsqueeze(1).expand(-1, max_len, -1)
        mask_len = seq_len_to_mask(seq_len).unsqueeze(-1)  # False means masked
        # bs, max_len, dynamics_dim*3 + statics_dim
        if dt is not None:
            # priv = dt.expand([-1, -1, 2])

            # dt1(t1-(-1.7)) | dt2(t2-t1) .... dt_{L}
            # temp = times.diff(axis=1)
            temp = times.diff(
                axis=1, prepend=-1.7*torch.ones((times.shape[0], 1, 1), device=times.device))
            # shifted_mask = torch.cat((mask_len[:, 1:], torch.zeros(mask_len.shape[0], 1,1,device=mask_len.device)), dim=1)

            priv = (mask_len*temp).expand([-1, -1, dynamics.shape[-1]])

            # CHECK out.sum()==0
            # out = (   (times[:,:-1,:]+priv[:,:,0:1])-times[:,1:,:]   )*mask_len[:,1:,:]

            # concat_times = torch.cat([-1.7*torch.ones((times.shape[0],1,1),device=times.device), times[:,:-1,:]], dim=1)
            # out = (   (concat_times+priv[:,:,0:1])-times   )*mask_len
        x = torch.cat([x, dynamics, priv, mask], dim=-1)
        x = self.embed(x, times)  # bs, max_len, hidden_dim
        # x = dynamics

        # packed = nn.utils.rnn.pack_padded_sequence(
        #     x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        # out, h = self.rnn(packed)  # h num_layers, bs, hidden_dim
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #     out, batch_first=True)  # bs, max_len, hidden_dim

        # mask: the mask for the src sequence (optional).
        #     src_key_padding_mask: the mask for the src keys per batch (optional).

        max_seq_len = torch.max(seq_len)
        src_key_padding_mask = torch.arange(max_seq_len, device=seq_len.device)[
            None, :] >= seq_len[:, None]

        transformer_output = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask)
        # Aggregate over sequence dimension
        mu = self.fc_mu(transformer_output.mean(dim=1))
        # Aggregate over sequence dimension
        logvar = self.fc_logvar(transformer_output.mean(dim=1))

        # # h, c = h
        # # layers, 1, bs, hidden_dim
        # h = h.view(self.layers, -1, bs, self.hidden_dim)
        # h1, _ = max_pooling(out, seq_len)  # bs, hidden_dim
        # h2 = mean_pooling(out, seq_len)  # bs, hidden_dim
        # h3 = h[-1].view(bs, -1)  # bs, hidden_dim
        # glob = torch.cat([h1, h2, h3], dim=-1)  # bs, hidden_dim*3
        # # bs, hidden_dim [s in paper]
        # glob = self.final(self.fc(self.drop(glob)))

        # # hf = h[:,0]
        # # hb = h[:,1]
        # # lasth = self.final(self.fc1(torch.cat([hf,hb], dim=-1)))
        # lasth = h.view(-1, bs, self.hidden_dim)  # layers, bs, hidden_dim

        # lasth = lasth.permute(1, 0, 2).contiguous().view(
        #     bs, -1)  # bs, layers*hidden_dim
        # lasth = self.final(self.fc1(self.drop(lasth)))
        # # bs, hidden_dim*(layers+1)   r = [s, {h} in paper]
        # hidden = torch.cat([glob, lasth], dim=-1)

        # mu = self.encoder_mu(hidden)
        # logvar = self.encoder_logvar(hidden)
        return mu, logvar


def apply_activation(processors, x):
    data = []
    st = 0
    for model in processors.models:
        if model.name == processors.use_pri:
            continue
        ed = model.tgt_len + st
        if model.which == 'categorical':
            if not model.missing or processors.use_pri:
                data.append(torch.softmax(x[:, st:ed], dim=-1))
            else:
                data.append(torch.softmax(x[:, st:ed-1], dim=-1))
                data.append(torch.sigmoid(x[:, ed-1:ed]))
            st = ed
        elif model.which == 'binary':
            data.append(torch.sigmoid(x[:, st:ed]))
            # applying no activation to the last layer
            # data.append(x[:, st:ed])
        else:
            # data.append(torch.sigmoid(x[:, st:ed]))
            if model.missing:
                data.append(x[:, ed-1:ed])
                data.append(torch.sigmoid(x[:, st:ed-1]))
            else:
                data.append(x[:, st:ed])
        st = ed
    assert ed == x.size(1)
    return torch.cat(data, dim=-1)


def apply_activation2(processors, x):
    data = []
    st = 0
    for model in processors.models:
        if model.name == processors.use_pri:
            continue
        ed = model.tgt_len + st
        if model.which == 'categorical':
            if not model.missing or processors.use_pri:
                data.append(torch.softmax(x[:, st:ed], dim=-1))
            else:
                data.append(torch.softmax(x[:, st:ed-1], dim=-1))
                data.append(torch.sigmoid(x[:, ed-1:ed]))
            st = ed
        else:
            # data.append(torch.sigmoid(x[:, st:ed]))
            # applying no activation to the last layer
            data.append(x[:, st:ed])
            st = ed
    assert ed == x.size(1)
    return torch.cat(data, dim=-1)


def pad_zero(x):
    input_x = torch.zeros_like(x[:, 0:1, :])
    input_x = torch.cat([input_x, x[:, :-1, :]], dim=1)
    return input_x


class Decoder(nn.Module):
    def __init__(self, processors, hidden_dim, layers, dropout):
        super(Decoder, self).__init__()
        self.s_P, self.d_P = processors
        self.hidden_dim = hidden_dim
        statics_dim, dynamics_dim = self.s_P.tgt_dim, self.d_P.tgt_dim
        self.dynamics_dim = dynamics_dim
        self.miss_dim = self.d_P.miss_dim
        self.s_dim = sum([x.tgt_len for x in self.d_P.models if x.missing])
        self.layers = layers
        self.embed = Embedding(dynamics_dim + self.s_dim +
                               statics_dim + self.miss_dim, hidden_dim, dropout)
        self.rnn = nn.GRU(hidden_dim * 2, hidden_dim, 1,
                          batch_first=True, dropout=dropout)
        self.miss_rnn = nn.GRU(hidden_dim, hidden_dim,
                               layers-1, batch_first=True, dropout=dropout)

        self.statics_fc = nn.Linear(hidden_dim, statics_dim)
        self.dynamics_fc = nn.Linear(hidden_dim, dynamics_dim)
        self.decay = nn.Linear(self.miss_dim * 2, hidden_dim)

        self.miss_fc = nn.Linear(hidden_dim, self.miss_dim)
        self.time_fc = nn.Linear(hidden_dim, 1)

    # embed: bs, hidden_dim*(layers+1)
    def forward(self, embed, sta, dynamics, lag, mask, priv, times, seq_len, forcing=11):
        # glob is s in the paper [bs, hidden_dim]
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        statics_x = self.statics_fc(glob)  # bs, statics_dim
        gen_sta = apply_activation(self.s_P, statics_x)  # bs, statics_dim

        bs, max_len, _ = dynamics.size()
        hidden = hidden.view(bs, self.layers, -1).permute(1,
                                                          0, 2).contiguous()  # layers, bs, hidden_dim
        hidden, finh = hidden[:-1], hidden[-1:]

        if forcing >= 1:
            # bs, max_len, dynamics_dim (right shift of dynamics across second dim)
            pad_dynamics = pad_zero(dynamics)
            pad_mask = pad_zero(mask)
            pad_times = pad_zero(times)
            pad_priv = pad_zero(priv)
            # bs, max_len, statics_dim
            sta_expand = sta.unsqueeze(1).expand(-1, max_len, -1)
            glob_expand = glob.unsqueeze(
                1).expand(-1, max_len, -1)  # bs, max_len, hidden_dim
            # bs, max_len, dynamics_dim*3 + statics_dim
            x = torch.cat([sta_expand, pad_dynamics,
                          pad_priv, pad_mask], dim=-1)
            x = self.embed(x, pad_times)  # bs, max_len, hidden_dim

            packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            # h layers-1, bs, hidden_dim
            out, h = self.miss_rnn(packed, hidden)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True)  # bs, max_len, hidden_dim

            gen_times = torch.sigmoid(self.time_fc(
                out)) + pad_times  # bs, max_len, 1
            # bs, max_len, miss_dim
            gen_mask = torch.sigmoid(self.miss_fc(out))

            # bs, max_len, latent_dim
            beta = torch.exp(-torch.relu(
                self.decay(torch.cat([mask, lag], dim=-1))))
            y = beta * out

            # bs, max_len, hidden_dim*2
            y = torch.cat([y, glob_expand], dim=-1)
            packed = nn.utils.rnn.pack_padded_sequence(
                y, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            out1, finh = self.rnn(packed, finh)  # finh 1, bs, hidden_dim
            out1, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out1, batch_first=True)  # bs, max_len, hidden_dim

            dyn = self.dynamics_fc(out1)  # bs, max_len, dynamics_dim
            dyn = apply_activation2(
                self.d_P, dyn.view(-1, self.dynamics_dim)).view(bs, -1, self.dynamics_dim)
        else:
            true_sta = sta.unsqueeze(1)
            gsta = gen_sta.detach().unsqueeze(1)
            glob = glob.unsqueeze(1)
            dyn = []
            gen_mask = []
            gen_times = []
            cur_x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
            gen_p = [torch.zeros((bs, 1, self.s_dim)).to(embed.device)]
            cur_mask = torch.zeros((bs, 1, self.miss_dim)).to(embed.device)
            cur_time = torch.zeros((bs, 1, 1)).to(embed.device)
            thr = torch.Tensor(
                [model.threshold for model in self.d_P.models if model.missing]).to(embed.device)
            thr = thr.view(1, 1, self.miss_dim).expand(bs, -1, -1)
            force = True
            for i in range(max_len):
                force = random.random() < forcing
                if i == 0 or not force:
                    sta = gsta
                    pre_x = cur_x
                    pre_mask = cur_mask.detach()
                    pre_time = cur_time.detach()
                else:
                    sta = true_sta
                    pre_x = dynamics[:, i-1:i]
                    pre_mask = mask[:, i-1:i]
                    pre_time = times[:, i-1:i]

                j = 0
                st = 0
                np = gen_p[-1].detach()
                for model in self.d_P.models:
                    if model.name == self.d_P.use_pri:
                        continue
                    if model.missing:
                        np[:, :, st:st+model.tgt_len] = np[:, :, st:st+model.tgt_len] * (
                            1-pre_mask[:, :, j:j+1]) + pre_x[:, :, st:st+model.tgt_len]*pre_mask[:, :, j:j+1]
                        j += 1
                    st += model.tgt_len
                gen_p.append(np)

                in_x = torch.cat([sta, pre_x, gen_p[i], pre_mask], dim=-1)
                in_x = self.embed(in_x, pre_time)

                out, hidden = self.miss_rnn(in_x, hidden)
                cur_time = torch.sigmoid(self.time_fc(out))

                if i == 0:
                    lg = cur_time.expand(-1, -1, self.miss_dim).detach()
                else:
                    lg = (1 - pre_mask) * lg + cur_time.detach()
                if i > 0:
                    gen_times.append(cur_time + pre_time)
                else:
                    gen_times.append(cur_time)
                cur_mask = torch.sigmoid(self.miss_fc(out))
                gen_mask.append(cur_mask)
                if force:
                    use_mask = mask[:, i:i+1]
                else:
                    use_mask = cur_mask.detach()

                beta = torch.exp(-torch.relu(self.decay(
                    torch.cat([use_mask, lg], dim=-1))))
                y = torch.cat([out * beta, glob], dim=-1)

                out, finh = self.rnn(y, finh)
                out = self.dynamics_fc(out)
                out = apply_activation(self.d_P, out.squeeze(1)).unsqueeze(1)
                dyn.append(out)

                x = out.detach()
                x = self.d_P.re_transform(
                    x.squeeze(1).cpu().numpy(), use_mask.squeeze(1).cpu().numpy())
                cur_x = torch.FloatTensor(x).to(embed.device).unsqueeze(1)
                cur_time = gen_times[-1].detach()
                cur_mask = (cur_mask > thr).detach().float()

            dyn = torch.cat(dyn, dim=1)
            gen_mask = torch.cat(gen_mask, dim=1)
            gen_times = torch.cat(gen_times, dim=1)

        return gen_sta, dyn, gen_mask, gen_times

    def generate_statics(self, embed):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        statics = self.statics_fc(glob)
        statics = apply_activation(self.s_P, statics)
        return statics.detach()

    def generate_dynamics(self, embed, sta, max_len):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        glob = glob.unsqueeze(1)
        sta = sta.unsqueeze(1)
        bs = glob.size(0)
        x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
        priv = [torch.zeros((bs, 1, self.s_dim)).to(
            embed.device), torch.zeros((bs, 1, self.s_dim)).to(embed.device)]
        mask = torch.zeros((bs, 1, self.miss_dim)).to(embed.device)
        t = torch.zeros((bs, 1, 1)).to(embed.device)
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()
        hidden, regress_hidden = hidden[:-1], hidden[-1:]
        dyn = []
        gen_mask = []
        gen_times = []
        thr = torch.Tensor(
            [model.threshold for model in self.d_P.models if model.missing]).to(embed.device)
        thr = thr.view(1, 1, self.miss_dim).expand(bs, -1, -1)
        for i in range(max_len):
            in_x = torch.cat([sta, x, priv[i], mask], dim=-1)
            in_x = self.embed(in_x, t)
            out, hidden = self.miss_rnn(in_x, hidden)
            cur_times = torch.sigmoid(self.time_fc(out))

            if i == 0:
                lg = cur_times.expand(-1, -1, self.miss_dim)
            else:
                lg = (1 - mask) * lg + cur_times
            if i > 0:
                gen_times.append(cur_times + gen_times[-1])
            else:
                gen_times.append(cur_times)
            mask = torch.sigmoid(self.miss_fc(out))
            gen_mask.append(mask)

            beta = torch.exp(-torch.relu(
                self.decay(torch.cat([mask, lg], dim=-1))))
            y = torch.cat([out * beta, glob], dim=-1)

            out, regress_hidden = self.rnn(y, regress_hidden)
            out = self.dynamics_fc(out)
            out = apply_activation(self.d_P, out.squeeze(1)).unsqueeze(1)
            dyn.append(out)

            j = 0
            x = out.detach()
            x = self.d_P.re_transform(
                x.squeeze(1).cpu().numpy(), mask.detach().squeeze(1).cpu().numpy())
            x = torch.FloatTensor(x).to(embed.device).unsqueeze(1)
            mask = (mask > thr).float()
            st = 0
            np = priv[-1].detach()
            for model in self.d_P.models:
                if model.name == self.d_P.use_pri:
                    continue
                if model.missing:
                    # x[:, :, st:st+model.tgt_len] *= mask[:,:,j:j+1]
                    np[:, :, st:st+model.tgt_len] = np[:, :, st:st+model.tgt_len] * \
                        (1-mask[:, :, j:j+1]) + x[:, :, st:st +
                                                  model.tgt_len] * mask[:, :, j:j+1]
                    j += 1
                st += model.tgt_len
            priv.append(np)
            t = gen_times[-1].detach()
            mask = mask.detach()

        dyn = torch.cat(dyn, dim=1)
        gen_mask = torch.cat(gen_mask, dim=1)
        gen_times = torch.cat(gen_times, dim=1)
        return dyn, gen_mask, gen_times


class TransformerDecoder(nn.Module):
    def __init__(self, processors, hidden_dim, layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.s_P, self.d_P = processors
        self.hidden_dim = hidden_dim
        statics_dim, dynamics_dim = self.s_P.tgt_dim, self.d_P.tgt_dim
        self.dynamics_dim = dynamics_dim
        self.miss_dim = self.d_P.miss_dim
        self.s_dim = sum([x.tgt_len for x in self.d_P.models if x.missing])
        self.layers = layers
        self.embed = Embedding(dynamics_dim + self.s_dim +
                               statics_dim + self.miss_dim, hidden_dim, dropout)
        self.rnn = nn.GRU(hidden_dim * 1, hidden_dim, 1,
                          batch_first=True, dropout=dropout)
        self.miss_rnn = nn.GRU(hidden_dim, hidden_dim,
                               layers-1, batch_first=True, dropout=dropout)

        self.statics_fc = nn.Linear((layers+1)*hidden_dim, statics_dim)
        self.dynamics_fc = nn.Linear(hidden_dim, dynamics_dim)
        self.decay = nn.Linear(self.miss_dim * 2, hidden_dim)

        self.miss_fc = nn.Linear(hidden_dim, self.miss_dim)
        self.time_fc = nn.Linear(hidden_dim, 1)

        num_heads = 4
        num_layers = 2
        z_dim = 128
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout,            batch_first=True
            ),
            num_layers=num_layers,
        )
        self.fc_finh = nn.Linear((layers+1)*hidden_dim, hidden_dim)

        self.fc = nn.Linear(4*hidden_dim, hidden_dim)
        mdn_num_components = 3
        self.fc_mdn = nn.Linear(hidden_dim, mdn_num_components*3)
        # self.fc_mdn = nn.Linear(hidden_dim, 8)
        # self.fc_mu = nn.Linear(hidden_dim, hidden_dim*4)
        # self.fc_logvar = nn.Linear(hidden_dim, hidden_dim*4)

    # embed: bs, hidden_dim*(layers+1)

    # embed.shape = bs, hidden_dim*(layers+1)
    def forward(self, embed, sta, dynamics, lag, mask, priv, times, seq_len, dt=None, forcing=11):
        # glob is s in the paper [bs, hidden_dim]
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        statics_x = self.statics_fc(embed)  # bs, statics_dim
        gen_sta = apply_activation(self.s_P, statics_x)  # bs, statics_dim

        bs, max_len, _ = dynamics.size()
        hidden = embed.view(bs, self.layers+1, -1).permute(1,
                                                           0, 2).contiguous()  # layers, bs, hidden_dim
        # hidden, finh = hidden[:-1], hidden[-1:]

        if forcing >= 1:
            # bs, max_len, dynamics_dim (right shift of dynamics across second dim)
            pad_dynamics = pad_zero(dynamics)
            pad_mask = pad_zero(mask)
            pad_times = pad_zero(times)
            if dt is not None:
                # pad_priv = pad_zero(dt.expand([-1, -1, 2]))

                # NEW METHOD
                mask_len = seq_len_to_mask(
                    seq_len).unsqueeze(-1)  # False means masked

                temp = times.diff(
                    axis=1, prepend=-1.7*torch.ones((times.shape[0], 1, 1), device=times.device))
                priv = (mask_len*temp).expand([-1, -1, dynamics.shape[-1]])
                pad_priv = pad_zero(priv)
            else:
                pad_priv = pad_zero(priv)
            # bs, max_len, statics_dim
            sta_expand = sta.unsqueeze(1).expand(-1, max_len, -1)
            glob_expand = glob.unsqueeze(
                1).expand(-1, max_len, -1)  # bs, max_len, hidden_dim
            # bs, max_len, dynamics_dim*3 + statics_dim
            x = torch.cat([sta_expand, pad_dynamics,
                          pad_priv, pad_mask], dim=-1)
            x = self.embed(x, pad_times)  # bs, max_len, hidden_dim
            max_seq_len = x.size(1)
            z_expanded = self.fc(embed.unsqueeze(1).repeat(
                1, max_seq_len, 1))  # bs, max_len, hidden_dim

            # Create subsequent mask
            tgt_subsequent_mask = torch.triu(torch.ones(
                max_seq_len, max_seq_len), diagonal=1).bool().to(x.device)
            tgt_key_padding_mask = torch.arange(max_seq_len, device=seq_len.device)[
                None, :] >= seq_len[:, None]

            out = self.transformer_decoder(
                tgt=x,
                memory=z_expanded,
                tgt_mask=tgt_subsequent_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                # memory_mask=subsequent_mask,
            )  # bs, max_len, hidden_dim

            # packed = nn.utils.rnn.pack_padded_sequence(
            #     x, seq_len.cpu(), batch_first=True, enforce_sorted=False)

            # # h layers-1, bs, hidden_dim
            # out, h = self.miss_rnn(packed, hidden)
            # out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            #     out, batch_first=True)  # bs, max_len, hidden_dim
            # diff_times = torch.sigmoid(self.time_fc(out))

            gen_times = time_activation(self.time_fc(
                out)) + pad_times*0  # bs, max_len, 1

            if hasattr(self, 'fc_mdn'):
                # bs, max_len, 3*mdn_num_components
                gen_times = self.fc_mdn(out)

            # gen_times = torch.relu(self.time_fc(out)) + \
            #     pad_times*0  # bs, max_len, 1
            # gen_times = (self.time_fc(out)) + pad_times  # bs, max_len, 1
            # bs, max_len, miss_dim
            gen_mask = torch.sigmoid(self.miss_fc(out))

            # bs, max_len, latent_dim
            beta = torch.exp(-torch.relu(
                self.decay(torch.cat([mask, lag], dim=-1))))
            y = beta * out

            # bs, max_len, hidden_dim*2
            # y = torch.cat([y, glob_expand], dim=-1)
            finh = self.fc_finh(embed)[None, :, :]  # [1,B,h]
            packed = nn.utils.rnn.pack_padded_sequence(
                y, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            out1, finh = self.rnn(packed, finh)  # finh 1, bs, hidden_dim
            out1, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out1, batch_first=True)  # bs, max_len, hidden_dim

            dyn = self.dynamics_fc(out)  # bs, max_len, dynamics_dim
            dyn = apply_activation(
                self.d_P, dyn.view(-1, self.dynamics_dim)).view(bs, -1, self.dynamics_dim)
        else:
            true_sta = sta.unsqueeze(1)
            gsta = gen_sta.detach().unsqueeze(1)
            glob = glob.unsqueeze(1)
            dyn = []
            gen_mask = []
            gen_times = []
            cur_x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
            gen_p = [torch.zeros((bs, 1, self.s_dim)).to(embed.device)]
            cur_mask = torch.zeros((bs, 1, self.miss_dim)).to(embed.device)
            cur_time = torch.zeros((bs, 1, 1)).to(embed.device)
            thr = torch.Tensor(
                [model.threshold for model in self.d_P.models if model.missing]).to(embed.device)
            thr = thr.view(1, 1, self.miss_dim).expand(bs, -1, -1)
            force = True
            for i in range(max_len):
                force = random.random() < forcing
                if i == 0 or not force:
                    sta = gsta
                    pre_x = cur_x
                    pre_mask = cur_mask.detach()
                    pre_time = cur_time.detach()
                else:
                    sta = true_sta
                    pre_x = dynamics[:, i-1:i]
                    pre_mask = mask[:, i-1:i]
                    pre_time = times[:, i-1:i]

                j = 0
                st = 0
                np = gen_p[-1].detach()
                for model in self.d_P.models:
                    if model.name == self.d_P.use_pri:
                        continue
                    if model.missing:
                        np[:, :, st:st+model.tgt_len] = np[:, :, st:st+model.tgt_len] * (
                            1-pre_mask[:, :, j:j+1]) + pre_x[:, :, st:st+model.tgt_len]*pre_mask[:, :, j:j+1]
                        j += 1
                    st += model.tgt_len
                gen_p.append(np)

                in_x = torch.cat([sta, pre_x, gen_p[i], pre_mask], dim=-1)
                in_x = self.embed(in_x, pre_time)

                out, hidden = self.miss_rnn(in_x, hidden)
                cur_time = time_activation(self.time_fc(out))
                # cur_time = torch.relu(self.time_fc(out))
                # cur_time = (self.time_fc(out))  # bs, max_len, 1
                if i == 0:
                    lg = cur_time.expand(-1, -1, self.miss_dim).detach()
                else:
                    lg = (1 - pre_mask) * lg + cur_time.detach()
                if i > 0:
                    gen_times.append(cur_time + pre_time)
                else:
                    gen_times.append(cur_time)
                cur_mask = torch.sigmoid(self.miss_fc(out))
                gen_mask.append(cur_mask)
                if force:
                    use_mask = mask[:, i:i+1]
                else:
                    use_mask = cur_mask.detach()

                beta = torch.exp(-torch.relu(self.decay(
                    torch.cat([use_mask, lg], dim=-1))))
                y = torch.cat([out * beta, glob], dim=-1)

                out, finh = self.rnn(y, finh)
                out = self.dynamics_fc(out)
                out = apply_activation(self.d_P, out.squeeze(1)).unsqueeze(1)
                dyn.append(out)

                x = out.detach()
                x = self.d_P.re_transform(
                    x.squeeze(1).cpu().numpy(), use_mask.squeeze(1).cpu().numpy())
                cur_x = torch.FloatTensor(x).to(embed.device).unsqueeze(1)
                cur_time = gen_times[-1].detach()
                cur_mask = (cur_mask > thr).detach().float()

            dyn = torch.cat(dyn, dim=1)
            gen_mask = torch.cat(gen_mask, dim=1)
            gen_times = torch.cat(gen_times, dim=1)

        return gen_sta, dyn, gen_mask, gen_times

    def generate_statics(self, embed):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        statics = self.statics_fc(embed)
        statics = apply_activation(self.s_P, statics)
        return statics.detach()

    def generate_dynamics(self, embed, sta, max_len, dt=None):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        glob = glob.unsqueeze(1)
        sta = sta.unsqueeze(1)
        bs = glob.size(0)
        x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
        priv = [torch.zeros((bs, 1, self.s_dim)).to(
            embed.device), torch.zeros((bs, 1, self.s_dim)).to(embed.device)]
        mask = torch.zeros((bs, 1, self.miss_dim)).to(embed.device)
        t = torch.zeros((bs, 1, 1)).to(embed.device)
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()
        hidden, regress_hidden = hidden[:-1], hidden[-1:]
        dyn = []
        gen_mask = []
        gen_times = []
        thr = torch.Tensor(
            [model.threshold for model in self.d_P.models if model.missing]).to(embed.device)
        thr = thr.view(1, 1, self.miss_dim).expand(bs, -1, -1)
        for i in range(max_len):
            in_x = torch.cat([sta, x, priv[i], mask], dim=-1)  # bs, 1, s+3*d
            in_x = self.embed(in_x, t)  # bs, 1, hidden_dim
            max_seq_len = in_x.size(1)
            # out, hidden = self.miss_rnn(in_x, hidden)
            z_expanded = self.fc(embed.unsqueeze(1).repeat(1, max_seq_len, 1))

            # # Create subsequent mask
            # tgt_subsequent_mask = torch.triu(torch.ones(
            #     max_seq_len, max_seq_len), diagonal=1).bool().to(x.device)
            # tgt_key_padding_mask = torch.arange(max_seq_len, device=seq_len.device)[
            # None, :] >= seq_len[:, None]

            out = self.transformer_decoder(
                tgt=in_x,
                memory=z_expanded,
                # tgt_mask=subsequent_mask,
            )  # bs, 1, hidden_dim

            cur_times = time_activation(self.time_fc(out))
            # cur_times = torch.relu(self.time_fc(out))
            # cur_times = (self.time_fc(out))  # bs, max_len, 1
            if i == 0:
                lg = cur_times.expand(-1, -1, self.miss_dim)
            else:
                lg = (1 - mask) * lg + cur_times

            if dt is not None:
                gen_times.append(cur_times)
            else:
                if i > 0:
                    gen_times.append(cur_times + gen_times[-1])
                else:
                    gen_times.append(cur_times)
            mask = torch.sigmoid(self.miss_fc(out))
            gen_mask.append(mask)

            beta = torch.exp(-torch.relu(
                self.decay(torch.cat([mask, lg], dim=-1))))
            # y = torch.cat([out * beta, glob], dim=-1)
            y = out * beta  # bs, 1, hidden_dim
            regress_hidden = self.fc_finh(embed)[None, :, :]  # [1,B,h]
            # out, regress_hidden = self.rnn(y, regress_hidden)
            out = self.dynamics_fc(out)
            out = apply_activation(
                self.d_P, out.squeeze(1)).unsqueeze(1)  # bs, 1, d
            dyn.append(out)

            j = 0
            x = out.detach()
            x = self.d_P.re_transform(
                x.squeeze(1).cpu().numpy(), mask.detach().squeeze(1).cpu().numpy())
            x = torch.FloatTensor(x).to(embed.device).unsqueeze(1)
            mask = (mask > thr).float()
            st = 0
            np = priv[-1].detach()
            for model in self.d_P.models:
                if model.name == self.d_P.use_pri:
                    continue
                if model.missing:
                    # x[:, :, st:st+model.tgt_len] *= mask[:,:,j:j+1]
                    np[:, :, st:st+model.tgt_len] = np[:, :, st:st+model.tgt_len] * \
                        (1-mask[:, :, j:j+1]) + x[:, :, st:st +
                                                  model.tgt_len] * mask[:, :, j:j+1]
                    j += 1
                st += model.tgt_len
            priv.append(np)
            t = gen_times[-1].detach()
            mask = mask.detach()

        dyn = torch.cat(dyn, dim=1)
        gen_mask = torch.cat(gen_mask, dim=1)
        gen_times = torch.cat(gen_times, dim=1)
        return dyn, gen_mask, gen_times

    def generate_dynamics2(self, embed, sta, max_len, dt=None):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        glob = glob.unsqueeze(1)
        sta = sta.unsqueeze(1)
        bs = glob.size(0)
        x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
        priv = [torch.zeros((bs, 1, self.s_dim)).to(
            embed.device), torch.zeros((bs, 1, self.s_dim)).to(embed.device)]
        mask = torch.zeros((bs, 1, self.miss_dim)).to(embed.device)
        t = torch.zeros((bs, 1, 1)).to(embed.device)
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()
        hidden, regress_hidden = hidden[:-1], hidden[-1:]
        dyn = [x]
        gen_mask = [mask]
        gen_times = []
        list_dt = [torch.zeros((bs, 1, 1)).to(embed.device)]
        thr = torch.Tensor(
            [model.threshold for model in self.d_P.models if model.missing]).to(embed.device)
        thr = thr.view(1, 1, self.miss_dim).expand(bs, -1, -1)

        for i in range(max_len):
            in_x = torch.cat([sta.expand(-1, i+1, -1),
                              torch.cat(dyn, axis=1),
                              torch.cat(list_dt, axis=1).expand(
                                  [-1, -1, self.dynamics_dim]),
                              torch.cat(gen_mask, axis=1)], dim=-1)  # bs, 1, s+3*d
            in_x = self.embed(in_x, t)  # bs, 1, hidden_dim
            max_seq_len = in_x.size(1)
            # out, hidden = self.miss_rnn(in_x, hidden)
            z_expanded = self.fc(embed.unsqueeze(1).repeat(1, max_seq_len, 1))

            # # Create subsequent mask
            # tgt_subsequent_mask = torch.triu(torch.ones(
            #     max_seq_len, max_seq_len), diagonal=1).bool().to(x.device)
            # tgt_key_padding_mask = torch.arange(max_seq_len, device=seq_len.device)[
            # None, :] >= seq_len[:, None]

            out = self.transformer_decoder(
                tgt=in_x,
                memory=z_expanded,
                # tgt_mask=subsequent_mask,
            )  # bs, 1, hidden_dim

            cur_times = time_activation(self.time_fc(out))[:, -1:, :]
            # cur_times = torch.relu(self.time_fc(out))[:, -1:, :]
            # cur_times = (self.time_fc(out))  # bs, max_len, 1
            if i == 0:
                lg = cur_times.expand(-1, -1, self.miss_dim)
            else:
                lg = (1 - mask) * lg + cur_times

        # if dt is not None:
            list_dt.append(cur_times)
            gen_times.append(cur_times)
        # else:
        #     if i > 0:
        #         gen_times.append(cur_times + gen_times[-1])
        #     else:
        #         gen_times.append(cur_times)
            mask = torch.sigmoid(self.miss_fc(out))[:, -1:, :]
            gen_mask.append(mask)

            beta = torch.exp(-torch.relu(
                self.decay(torch.cat([mask, lg], dim=-1))))
            # y = torch.cat([out * beta, glob], dim=-1)
            y = out * beta  # bs, 1, hidden_dim
            regress_hidden = self.fc_finh(embed)[None, :, :]  # [1,B,h]
            # out, regress_hidden = self.rnn(y, regress_hidden)
            out = self.dynamics_fc(out)[:, -1:, :]
            out = apply_activation(
                self.d_P, out.view(-1, self.dynamics_dim)).view(bs, -1, self.dynamics_dim)  # bs, 1, d
            dyn.append(out)

            j = 0
            x = out.detach()
            x = self.d_P.re_transform(
                x.squeeze(1).cpu().numpy(), mask.detach().squeeze(1).cpu().numpy())
            x = torch.FloatTensor(x).to(embed.device).unsqueeze(1)
            mask = (mask > thr).float()
            st = 0
            np = priv[-1].detach()
            for model in self.d_P.models:
                if model.name == self.d_P.use_pri:
                    continue
                if model.missing:
                    # x[:, :, st:st+model.tgt_len] *= mask[:,:,j:j+1]
                    np[:, :, st:st+model.tgt_len] = np[:, :, st:st+model.tgt_len] * \
                        (1-mask[:, :, j:j+1]) + x[:, :, st:st +
                                                  model.tgt_len] * mask[:, :, j:j+1]
                    j += 1
                st += model.tgt_len
            priv.append(np)
            # list_prives.append(np)

            t = gen_times[-1].detach()
            mask = mask.detach()

        dyn = torch.cat(dyn[1:], dim=1)
        gen_mask = torch.cat(gen_mask[1:], dim=1)
        gen_times = torch.cat(gen_times, dim=1)
        dts = torch.cat(list_dt[1:], dim=1)

        #  value_max = 0.5109656108798026
        # gt_new = 1
        # loss4 = self.time_loss(gt, dt, seq_len)
        # seq_mask = seq_len_to_mask(seq_len)
        # miss_loss2_bl += self.time_loss(gt *
        #                                 0+0.03, dt, seq_len).item()
        # # shift to right and pad the first element
        # # [t1,t1,....,t_{L-1}]
        # temp = torch.cat(
        #     [times[:, :1, :], times[:, :-1, :]], axis=1)*seq_mask.unsqueeze(-1)
        # # predicted [t1,t2,...,t_{L}]
        # times_pred = temp + gt*value_max
        # times_true = times
        # loss_abs += self.time_loss(
        #     times_pred, times_true, seq_len).item()
        return dyn, gen_mask, dts


class Autoencoder(nn.Module):
    def __init__(self, processors, hidden_dim, embed_dim, layers, dropout=0.0):
        super(Autoencoder, self).__init__()
        print(processors[0].tgt_dim, processors[1].tgt_dim,
              processors[1].miss_dim)
        s_dim = sum([x.tgt_len for x in processors[1].models if x.missing])
        self.encoder = Encoder(processors[0].tgt_dim + processors[1].tgt_dim +
                               s_dim + processors[1].miss_dim, hidden_dim, embed_dim, layers, dropout)
        self.decoder = Decoder(processors, hidden_dim, layers, dropout)
        self.decoder.embed = self.encoder.embed

    def forward(self, sta, dyn, lag, mask, priv, nex, times, seq_len, forcing=1):
        hidden = self.encoder(sta, dyn, priv, nex, mask, times, seq_len)
        return self.decoder(hidden, sta, dyn, lag, mask, priv, times, seq_len, forcing=forcing)


class VariationalAutoencoder(nn.Module):
    def __init__(self, processors, hidden_dim, embed_dim, layers, dropout=0.0):
        super(VariationalAutoencoder, self).__init__()
        print(processors[0].tgt_dim, processors[1].tgt_dim,
              processors[1].miss_dim)
        s_dim = sum([x.tgt_len for x in processors[1].models if x.missing])
        self.encoder = TransformerVariationalEncoder(
            processors[0].tgt_dim + processors[1].tgt_dim + s_dim + processors[1].miss_dim, hidden_dim, embed_dim, layers, dropout)
        self.decoder = TransformerDecoder(
            processors, hidden_dim, layers, dropout)
        self.decoder.embed = self.encoder.embed

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, sta, dyn, lag, mask, priv, nex, times, seq_len, dt=None, forcing=1):
        mu, logvar = self.encoder(
            sta, dyn, priv, nex, mask, times, seq_len, dt)
        self.KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        hidden = self.reparameterize(mu, logvar)

        return self.decoder(hidden, sta, dyn, lag, mask, priv, times, seq_len, dt=dt, forcing=forcing)
