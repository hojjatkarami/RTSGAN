# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler
from fastNLP import seq_len_to_mask
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
from basic import max_pooling
from autoencoder import Autoencoder, VariationalAutoencoder
from gan import Generator, Discriminator
import random
from torch import autograd
import time

from tqdm import tqdm

import wandb
from plotly.subplots import make_subplots

import plotly.express as px
import plotly.graph_objects as go
from MulticoreTSNE import MulticoreTSNE as TSNE

TIME_CONST = 0


class AeGAN:
    def __init__(self, processors, params):
        self.params = params
        if self.params.get("force") is None:
            self.params["force"] = ""
        self.device = params["device"]
        self.logger = params["logger"]
        self.static_processor, self.dynamic_processor = processors

        if params["vae"]:
            self.ae = VariationalAutoencoder(
                processors, self.params["hidden_dim"], self.params["embed_dim"], self.params["layers"], dropout=self.params["dropout"])
        else:
            self.ae = Autoencoder(
                processors, self.params["hidden_dim"], self.params["embed_dim"], self.params["layers"], dropout=self.params["dropout"])

        self.ae.to(self.device)
        """
        self.decoder_optm = torch.optim.Adam(
            params=self.ae.decoder.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
        )
        self.encoder_optm = torch.optim.Adam(
            params=self.ae.encoder.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
        )
        """
        self.ae_optm = torch.optim.Adam(
            params=self.ae.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
            weight_decay=self.params["weight_decay"]
        )
        self.ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.ae_optm, T_max=10, eta_min=0.00001)
        self.ae_scheduler = torch.optim.lr_scheduler.StepLR(
            self.ae_optm, 10, gamma=0.5)
        self.loss_con = nn.MSELoss(reduction='none')
        self.loss_dis = nn.NLLLoss(reduction='none')
        self.loss_mis = nn.BCELoss(reduction='none')

        self.generator = Generator(
            self.params["noise_dim"], self.params["hidden_dim"], self.params["layers"]).to(self.device)
        self.discriminator = Discriminator(
            self.params["embed_dim"]).to(self.device)
        self.discriminator_optm = torch.optim.RMSprop(
            params=self.discriminator.parameters(),
            lr=self.params['gan_lr'],
            alpha=self.params['gan_alpha'],
        )
        self.generator_optm = torch.optim.RMSprop(
            params=self.generator.parameters(),
            lr=self.params['gan_lr'],
            alpha=self.params['gan_alpha'],
        )

    def load_ae(self, pretrained_dir=None):
        if pretrained_dir is not None:
            path = pretrained_dir
        else:
            path = '{}/ae.dat'.format(self.params["root_dir"])
        self.logger.info("load: "+path)
        self.ae.load_state_dict(torch.load(path, map_location=self.device))

    def load_generator(self, pretrained_dir=None):
        if pretrained_dir is not None:
            path = pretrained_dir
        else:
            path = '{}/generator.dat'.format(self.params["root_dir"])
        self.logger.info("load: "+path)
        self.generator.load_state_dict(
            torch.load(path, map_location=self.device))

    def sta_loss(self, data, target):
        loss = 0
        n = len(self.static_processor.models)
        st = 0
        # print(data,target)
        for i, model in enumerate(self.static_processor.models):
            ed = st + model.tgt_len - int(model.missing)
            use = 1
            if model.missing:
                loss += 0.1 * \
                    torch.mean(self.loss_mis(data[:, ed], target[:, ed]))
                use = target[:, ed:ed+1]

            if model.which == "categorical":
                loss += torch.mean(use * self.loss_dis((data[:, st:ed]+1e-8).log(
                ), torch.argmax(target[:, st:ed], dim=-1)).unsqueeze(-1))
            elif model.which == "binary":
                loss += torch.mean(use *
                                   self.loss_mis(data[:, st:ed], target[:, st:ed]))
            else:
                loss += torch.mean(use *
                                   self.loss_con(data[:, st:ed], target[:, st:ed]))

            st += model.tgt_len
        assert st == target.size(-1)
        return loss/n

    def dyn_loss(self, data, target, seq_len, mask):
        loss = []
        n = len(self.dynamic_processor.models)
        st = 0
        i = 0
        for model in self.dynamic_processor.models:
            if model.name == self.dynamic_processor.use_pri:
                continue
            ed = st + model.tgt_len
            use = 1
            if model.missing:
                use = mask[:, :, i:i+1]
                i += 1

            if model.which == "categorical":
                x = (data[:, :, st:ed] + 1e-8).log().transpose(1, 2)
                loss.append(
                    use * self.loss_dis(x, torch.argmax(target[:, :, st:ed], dim=-1)).unsqueeze(-1))
            elif model.which == "binary":
                loss.append(
                    use * self.loss_mis(data[:, :, st:ed], target[:, :, st:ed]))
            else:
                loss.append(
                    use * 10 * self.loss_con(data[:, :, st:ed], target[:, :, st:ed]))
            st += model.tgt_len
        assert i == mask.size(-1)
        loss = torch.cat(loss, dim=-1)
        seq_mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))
        return torch.mean(loss)

    def time_loss(self, data, target, seq_len):
        loss = self.loss_con(data, target)  # batch_size, max_len, 1
        seq_mask = seq_len_to_mask(seq_len)  # batch_size, max_len
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))
        return torch.mean(loss)  # scalar

    def mdn_loss(self, mixture_params, target, seq_len):

        seq_mask = seq_len_to_mask(seq_len)  # batch_size, max_len

        # MDN loss function
        # Split mixture parameters into mean, variance, and weight components
        num_components = mixture_params.shape[-1]//3
        mean, log_variance, raw_weight = torch.split(
            mixture_params, num_components, dim=-1)
        # Apply softplus transformation to ensure non-negative weights
        mean = nn.functional.softplus(mean)
        weight = nn.Softmax(dim=-1)(raw_weight)

        variance = torch.exp(log_variance)

        # Calculate the negative log-likelihood of the target given the mixture parameters
        # [bs, max_len, 3]
        log_prob = -0.5 * torch.log(2 * np.pi * variance) - \
            0.5 * ((target - mean) ** 2) / variance
        log_weighted_prob = log_prob + torch.log(weight)
        # log_weighted_prob = log_weighted_prob.masked_fill(~seq_mask.unsqueeze(-1),-1e9)
        loss = -torch.logsumexp(log_weighted_prob,
                                dim=-1)  # [bs, max_len]
        loss = torch.masked_select(loss, seq_mask)

        # gt new
        # mean, _, raw_weight = torch.split(gt, 3, dim=-1)
        max_idx = torch.argmax(log_weighted_prob, dim=-1)
        # point_predictions = mean[torch.arange(mean.size(0)),torch.arange(mean.size(1)), max_idx]
        # [bs, max_len, 1]
        target_point_estimate = torch.gather(mean, 2, max_idx.unsqueeze(-1))

        # # discrete loss function
        # num_classes = mixture_params.shape[-1]

        # # from 0 to 0.02 with step 0.002
        # bin_ranges = torch.tensor([0, 0.002, 0.004, 0.006, 0.008, 0.01,
        #                           0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.025]).to(target.device)

        # bin_ranges = torch.tensor([0.0000, 0.003, 0.0060, 0.0090, 0.0120, 0.0150, 0.0180, 0.020,
        #                            0.025,]).to(target.device)
        # target_cut = torch.bucketize(target, bin_ranges)
        # target_cut[target_cut < 0] = 0
        # target_cut[target_cut >= num_classes] = num_classes-1

        # freqs = torch.bincount(target_cut.flatten())
        # w = freqs.sum()/freqs
        # w[w > 10*w.min()] = 10*w.min()
        # w = w / w.sum()

        # criterion = nn.CrossEntropyLoss(
        #     reduction='none')  # , weight=w.to(target.device)
        # loss = criterion(mixture_params.view(-1, num_classes),
        #                  target_cut.view(-1)).view(target.size())  # [bs*max_len]
        # loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))
        # # now point estimate
        # bin_indices = torch.argmax(mixture_params, dim=-1)
        # midpoints = (bin_ranges[:-1] + bin_ranges[1:]) / 2

        # midpoints = torch.tensor(
        #     [0.0020, 0.0050, 0.0080, 0.0100, 0.014, 0.0160, 0.0190, 0.0210,]).to(target.device)
        # target_point_estimate = midpoints[bin_indices].unsqueeze(-1)

        return torch.mean(loss), target_point_estimate

    def missing_loss(self, data, target, seq_len):  # data (batch_size, seq_len, 35)
        thr = torch.Tensor(
            [model.threshold for model in self.dynamic_processor.models if model.missing]).to(data.device)
        thr = thr.unsqueeze(0).unsqueeze(0)  # 1, 1, 35

        scale = thr * target + (1 - thr) * (1 - target)  # [bs, seq_len, 35]
        # BCE loss with red=none with data.shape
        loss = self.loss_mis(data, target) * scale
        seq_mask = seq_len_to_mask(seq_len)  # batch_size, seq_len
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))

        mx, _ = max_pooling(data, seq_len)  # batch_size, 35
        # batch_size, 35 which features are available during the whole sequence
        gold_mx, _ = torch.max(target, dim=1)
        loss1 = self.loss_mis(mx, gold_mx)  # batch_size, 35
        # + torch.mean(torch.masked_select(loss1, gold_mx == 0))
        return torch.mean(loss)

    def plot_ae(self, dyn, times, mask, seq_len, out_dyn, gt, missing, seq_len_pred):

        # dyn, mask, of shape [bs, seq_len, n_channels]
        # times [bs, seq_len, 1]
        def render(dyn, times, mask, L, i_channel):
            x_values = torch.masked_select(torch.arange(
                out_dyn.shape[1], device=mask.device), mask[0, :, i_channel] == 1)
            x_values = torch.masked_select(
                times[0, :, 0], mask[0, :, i_channel] == 1)
            y_values = torch.masked_select(
                dyn[0, :, i_channel], mask[0, :, i_channel] == 1)

            return x_values.cpu().detach().numpy()[:L], y_values.cpu().detach().numpy()[:L]
        # mask=1 means the value is available
        L_pred = max(seq_len_pred[0].astype(int), 1)
        L_true = seq_len[0]
        fig = go.Figure()
        x_values, y_values = render(dyn, times, mask, L_true, 0)
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode='lines+markers', name='S1', line=dict(color='red')))
        x_values, y_values = render(out_dyn, gt, missing > 0.5, L_pred, 0)
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode='lines+markers', name='S1_rec', line=dict(color='red', dash='dash')))
        x_values, y_values = render(dyn, times, mask, L_true, 1)
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode='lines+markers', name='S2', line=dict(color='blue')))
        x_values, y_values = render(out_dyn, gt, missing > 0.5, L_pred, 1)
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode='lines+markers', name='S2_rec', line=dict(color='blue', dash='dash')))

        # correlation plot
        fig = go.Figure()
        i = [0, 1]
        x_true = torch.masked_select(
            dyn[:, :, i], mask[:, :, i] == 1).cpu().detach().numpy()
        # x_pred = torch.masked_select(out_dyn,missing>0.5)
        x_pred = torch.masked_select(
            out_dyn[:, :, i], mask[:, :, i] == 1).cpu().detach().numpy()
        fig.add_trace(go.Scatter(
            x=x_true, y=x_pred, mode='markers', name='S1', line=dict(color='red')))

        # times_true = torch.masked_select(
        #     times[:, :, 0].cpu(), torch.from_numpy(seq_len_to_mask(seq_len))).cpu().detach().numpy()
        # times_pred = torch.masked_select(
        #     gt[:, :, 0].cpu(), torch.from_numpy(seq_len_to_mask(seq_len))).cpu().detach().numpy()

        mask_len = seq_len_to_mask(
            torch.from_numpy(seq_len).to(times.device)).unsqueeze(-1)  # False means masked

        temp = times.diff(
            axis=1, prepend=TIME_CONST*torch.ones((times.shape[0], 1, 1), device=times.device))

        dt_true = (mask_len*temp)
        dt_true = torch.masked_select(
            dt_true[:, 0:, 0], mask[:, 0:, :].sum(-1) > 0).cpu().detach().numpy()
        # dt_true = torch.masked_select(times.diff(
        #     axis=1)[:, :, 0], mask.sum(-1)[:, 0:] > 0).cpu().detach().numpy()
        dt_pred = torch.masked_select(
            gt[:, 0:, 0], mask[:, 0:, :].sum(-1) > 0).cpu().detach().numpy()
        # dt_pred = torch.masked_select(
        #     gt.diff(axis=1)[:, :, 0], mask.sum(-1)[:, 1:] > 0).cpu().detach().numpy()
        # i = []
        # x_true = torch.masked_select(dyn[:, :, i], mask[:, :, i] == 1)
        # # x_pred = torch.masked_select(out_dyn,missing>0.5)
        # x_pred = torch.masked_select(out_dyn[:, :, i], mask[:, :, i] == 1)
        # fig.add_trace(go.Scatter(
        #     x=x_true.cpu().detach().numpy(), y=x_pred.cpu().detach().numpy(), mode='markers', name='S2', line=dict(color='blue')))
        # fig.update_layout(yaxis_range=[-2, 2])
        # # Set layout properties to make the axis square
        # fig.update_layout(
        #     xaxis=dict(scaleanchor="y", scaleratio=1),
        #     yaxis=dict(scaleanchor="x", scaleratio=1)
        # )

        # a 1 x 3 subplot
        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            "S1", "S2", "Correlation"))
        fig.add_trace(go.Scatter(
            x=x_true, y=(x_true-x_pred), mode='markers', name='x', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=seq_len, y=(seq_len-seq_len_pred), mode='markers', name='seq_len', line=dict(color='red')), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=dt_true, y=(dt_true-dt_pred), mode='markers', name='dt', line=dict(color='red')), row=1, col=3)
        # fig.update_layout(
        #     xaxis=dict(scaleanchor="y", scaleratio=1),
        #     yaxis=dict(scaleanchor="x", scaleratio=1)
        # )
        df = pd.DataFrame({'X': dt_true, 'Y': dt_pred}).round(3)
        df['count'] = df.groupby(['X', 'Y'])['X'].transform('count')
        fig2 = px.scatter(
            df,
            x='X',
            y='Y',
            size='count',
            marginal_x='histogram',
            marginal_y='histogram'
        )
        fig2.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        fig2.add_shape(
            type='line',
            x0=df['X'].min(),
            y0=df['X'].min(),
            x1=df['X'].max(),
            y1=df['X'].max(),
            line=dict(color='red', width=2)
        )
        return fig, fig2

    def train_ae(self, dataset, epochs=800):
        min_loss = 1e15
        best_epsilon = 0
        train_batch = DataSetIter(
            dataset=dataset, batch_size=self.params["ae_batch_size"], sampler=RandomSampler())
        force = 1
        for i in tqdm(range(epochs)):
            self.ae.train()
            tot_loss = 0
            con_loss = 0
            dis_loss = 0
            miss_loss1 = 0
            miss_loss2 = 0
            tot = 0
            t1 = time.time()
            if self.params["force"] == "linear":
                if i >= epochs / 100 and i < epochs / 2:
                    force -= 2 / epochs
                elif i >= epochs / 2:
                    force = -1
            elif self.params["force"] == "constant":
                force = 0.5
            else:
                force = 1
            for batch_x, batch_y in train_batch:
                self.ae.zero_grad()
                sta = batch_x["sta"].to(self.device)
                dyn = batch_x["dyn"].to(self.device)
                mask = batch_x["mask"].to(self.device)
                lag = batch_x["lag"].to(self.device)
                priv = batch_x["priv"].to(self.device)
                nex = batch_x["nex"].to(self.device)
                times = batch_x["times"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)

                out_sta, out_dyn, missing, gt = self.ae(
                    sta, dyn, lag, mask, priv, nex, times, seq_len, forcing=force)
                # [bs, max_len, n_features], [bs, max_len, n_features], [bs]
                loss3 = self.missing_loss(missing, mask, seq_len)
                miss_loss1 += loss3.item()
                loss4 = self.time_loss(gt, times, seq_len)
                miss_loss2 += loss4.item()

                loss1 = self.sta_loss(out_sta, sta)
                loss2 = self.dyn_loss(out_dyn, dyn, seq_len, mask)

                sta_num = len(self.static_processor.models)
                dyn_num = len(self.dynamic_processor.models)
                scale1 = sta_num / (sta_num + dyn_num)
                scale2 = dyn_num / (sta_num + dyn_num)
                scale3 = 0.1

                loss = scale1 * loss1 + scale2 * \
                    (loss2 + loss3) + scale3 * loss4
                # loss = loss1 + loss2 + loss3 + loss4
                loss.backward()
                self.ae_optm.step()

                tot_loss += loss.item()
                con_loss += loss1.item()
                dis_loss += loss2.item()
                tot += 1

            tot_loss /= tot
            wandb.log({"ae_loss": tot_loss}, step=i+1)

            if i % 5 == 0:
                self.logger.info("Epoch:{} {}\t{}\t{}\t{}\t{}\t{}".format(
                    i+1, time.time()-t1, force, con_loss/tot, dis_loss/tot, miss_loss1/tot, miss_loss2/tot))
            if i % 5 == 0:
                fig_AE_rec = self.plot_ae(
                    dyn, times, mask, out_dyn, gt, missing, i)
                wandb.log(
                    {"example_rec": wandb.Plotly(fig_AE_rec)}, step=i)
                # x_values = torch.masked_select(torch.arange(
                #     out_dyn.shape[1], device=mask.device), mask[0, :, 0] == 1)
                # y_values = torch.masked_select(
                #     dyn[0, :, 0], mask[0, :, 0] == 1)
                # times [bs, max_len, 1], dyn [bs, max_len, n_features], out_dyn [bs, max_len, n_features]
            if i % 100 == 99:
                torch.save(self.ae.state_dict(),
                           '{}/ae{}.dat'.format(self.params["root_dir"], i))
                self.generate_ae(dataset[:100])

        torch.save(self.ae.state_dict(),
                   '{}/ae.dat'.format(self.params["root_dir"]))

    def train_gan(self, dataset, iterations=15000, d_update=5):
        self.discriminator.train()
        self.generator.train()
        self.ae.train()
        batch_size = self.params["gan_batch_size"]
        idxs = list(range(len(dataset)))
        batch = DataSetIter(
            dataset=dataset, batch_size=batch_size, sampler=RandomSampler())
        min_loss = 1e15
        for iteration in tqdm(range(iterations)):
            avg_d_loss = 0
            t1 = time.time()
            toggle_grad(self.generator, False)
            toggle_grad(self.discriminator, True)

            for j in range(d_update):
                for batch_x, batch_y in batch:
                    self.discriminator_optm.zero_grad()
                    z = torch.randn(
                        batch_size, self.params['noise_dim']).to(self.device)

                    sta = batch_x["sta"].to(self.device)
                    dyn = batch_x["dyn"].to(self.device)
                    mask = batch_x["mask"].to(self.device)
                    lag = batch_x["lag"].to(self.device)
                    priv = batch_x["priv"].to(self.device)
                    nex = batch_x["nex"].to(self.device)
                    times = batch_x["times"].to(self.device)
                    seq_len = batch_x["seq_len"].to(self.device)

                    real_rep = self.ae.encoder(
                        sta, dyn, priv, nex, mask, times, seq_len)
                    d_real = self.discriminator(real_rep)
                    dloss_real = -d_real.mean()
                    # y = d_real.new_full(size=d_real.size(), fill_value=1)
                    # dloss_real = F.binary_cross_entropy_with_logits(d_real, y)
                    dloss_real.backward()

                    """
                    dloss_real.backward(retain_graph=True)
                    reg = 10 * compute_grad2(d_real, real_rep).mean()
                    reg.backward()
                    """

                    # On fake data
                    with torch.no_grad():
                        x_fake = self.generator(z)

                    x_fake.requires_grad_()
                    d_fake = self.discriminator(x_fake)
                    dloss_fake = d_fake.mean()
                    """
                    y = d_fake.new_full(size=d_fake.size(), fill_value=0)
                    dloss_fake = F.binary_cross_entropy_with_logits(d_fake, y)
                    """
                    dloss_fake.backward()
                    """
                    dloss_fake.backward(retain_graph=True)
                    reg = 10 * compute_grad2(d_fake, x_fake).mean()
                    reg.backward()
                    """
                    reg = 10 * self.wgan_gp_reg(real_rep, x_fake)
                    reg.backward()

                    self.discriminator_optm.step()
                    d_loss = dloss_fake + dloss_real
                    avg_d_loss += d_loss.item()
                    break

            avg_d_loss /= d_update

            toggle_grad(self.generator, True)
            toggle_grad(self.discriminator, False)
            self.generator_optm.zero_grad()
            z = torch.randn(batch_size, self.params['noise_dim']).to(
                self.device)  # batch_size, noise_dim
            fake = self.generator(z)  # batch_size, hidden_dim
            g_loss = -torch.mean(self.discriminator(fake))  # [batch,1]->scalar
            g_loss.backward()
            self.generator_optm.step()

            wandb.log({"d_loss": avg_d_loss, "g_loss": g_loss.item()},
                      step=iteration+1)
            if iteration % 50 == 49:
                self.logger.info('[Iteration %d/%d] [%f] [D loss: %f] [G loss: %f] [%f]' % (
                    iteration, iterations, time.time()-t1, avg_d_loss, g_loss.item(), reg.item()
                ))

            if iteration % 10 == 0:
                # table = self.save_sample_wandb()
                # wandb.log(
                #     {"example_syn": wandb.plot.line(table, "t", "y",
                #                                     title="Example Synthesized Time Series")}, step=iteration+1)

                # plot one generated sample
                # plot = self.save_sample_wandb(
                #     seq_len=batch_x['seq_len'][0].item())
                fig = self.save_sample_wandb()
                wandb.log(
                    {"example_syn": wandb.Plotly(fig)}, step=iteration+1)

                # plot t-SNE of latent space
                plot = self.plot_tsne(real_rep.cpu().detach(
                ).numpy(), x_fake.cpu().detach().numpy())
                wandb.log(
                    {"tsne": plot}, step=iteration+1)

        torch.save(self.generator.state_dict(),
                   '{}/generator.dat'.format(self.params["root_dir"]))

    def train_ae2(self, dataset, epochs=800):
        min_loss = 1e15
        best_epsilon = 0
        train_batch = DataSetIter(
            dataset=dataset, batch_size=self.params["ae_batch_size"], sampler=RandomSampler())
        force = 1
        for i in tqdm(range(epochs)):
            # self.ae_scheduler.step()
            self.ae.train()
            tot_loss = 0
            con_loss = 0
            dis_loss = 0
            KLD_loss = 0
            loss_abs = 0
            loss_abs_bl = 0
            static_loss = 0
            dynamic_loss = 0
            time_loss = 0
            loss_mdn_bl = 0
            miss_loss = 0
            miss_loss1 = 0
            miss_loss2 = 0
            miss_loss2_bl = 0
            tot = 0
            t1 = time.time()
            if self.params["force"] == "linear":
                if i >= epochs / 100 and i < epochs / 2:
                    force -= 2 / epochs
                elif i >= epochs / 2:
                    force = -1
            elif self.params["force"] == "constant":
                force = 0.5
            else:
                force = 1
            for batch_x, batch_y in train_batch:
                self.ae.zero_grad()
                sta = batch_x["sta"].to(self.device)
                dyn = batch_x["dyn"].to(self.device)
                mask = batch_x["mask"].to(self.device)
                lag = batch_x["lag"].to(self.device)
                priv = batch_x["priv"].to(self.device)
                nex = batch_x["nex"].to(self.device)
                times = batch_x["times"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)
                if "dt" in batch_x:
                    dt = batch_x["dt"].to(self.device)
                else:
                    dt = None
                    dt = True

                out_sta, out_dyn, missing, gt = self.ae(
                    sta, dyn, lag, mask, priv, nex, times, seq_len, dt=dt, forcing=force)
                # [bs, max_len, n_features], [bs, max_len, n_features], [bs]
                loss3 = self.missing_loss(missing, mask, seq_len)
                miss_loss1 += loss3.item()
                if ("dt" in batch_x) or True:
                    value_max = 0.5109656108798026
                    gt_new = 1
                    mask_len = seq_len_to_mask(
                        seq_len).unsqueeze(-1)  # False means masked

                    temp = times.diff(
                        axis=1, prepend=TIME_CONST*torch.ones((times.shape[0], 1, 1), device=times.device))
                    mask_len = seq_len_to_mask(
                        seq_len).unsqueeze(-1)  # False means masked
                    dt_true = (mask_len*temp)

                    if hasattr(self.ae.decoder, 'fc_mdn'):
                        # mixture_params = mdn(gt, dt_true, seq_len)
                        mixture_params = torch.zeros_like(gt)
                        loss4, gt = self.mdn_loss(gt, dt_true, seq_len)

                        # baseline mdn (handcrafted)
                        # mixture_params[:, :, 0] = 0.0034
                        # mixture_params[:, :, 1] = 0.0104
                        # mixture_params[:, :, 2] = 0.0208

                        # mixture_params[:, :, 3] = -3.8
                        # mixture_params[:, :, 4] = -3.8
                        # mixture_params[:, :, 5] = -3.8

                        # mixture_params[:, :, [6]] = 1/torch.abs(dt_true-0.0034)
                        # mixture_params[:, :, [7]] = 1/torch.abs(dt_true-0.0104)
                        # mixture_params[:, :, [8]] = 1/torch.abs(dt_true-0.0208)

                        # loss_temp, gt2 = self.mdn_loss(
                        #     mixture_params, dt_true, seq_len)
                        # loss_mdn_bl += loss_temp.item()
                        # # extract pointn estimate
                        # mean, _, raw_weight = torch.split(gt, 3, dim=-1)
                        # _, max_idx = torch.max(raw_weight, dim=-1)
                        # # point_predictions = mean[torch.arange(mean.size(0)),torch.arange(mean.size(1)), max_idx]
                        # # [bs, max_len, 1]
                        # gt = torch.gather(mean, 2, max_idx.unsqueeze(-1))
                    else:
                        loss4 = self.time_loss(gt, dt_true, seq_len)

                    seq_mask = seq_len_to_mask(seq_len)
                    # miss_loss2_bl += self.time_loss(gt *
                    #                                 0+0.03, dt, seq_len).item()
                    # # ### times_pred OLD
                    # # # shift to right and pad the first element
                    # # # [t1,t1,....,t_{L-1}]
                    # # temp = torch.cat(
                    # #     [times[:, :1, :], times[:, :-1, :]], axis=1)*seq_mask.unsqueeze(-1)
                    # # # predicted [t1,t2,...,t_{L}]
                    # # times_pred = temp + gt*value_max
                    # # times_true = times
                    # # loss_abs += self.time_loss(
                    # #     times_pred, times_true, seq_len).item()

                    # # loss_abs_bl
                    # times_pred = temp + 0.03
                    # times_true = times
                    # loss_abs_bl += self.time_loss(
                    #     times_pred, times_true, seq_len).item()

                    # times_pred NEW

                    concat_times = torch.cat(
                        [TIME_CONST*torch.ones((times.shape[0], 1, 1), device=times.device), times[:, :-1, :]], dim=1)
                    times_pred = ((concat_times+gt[:, :, 0:1]))*mask_len
                    loss_abs += self.time_loss(
                        times_pred, times, seq_len).item()

                    # true dt
                    temp = times.diff(
                        axis=1, prepend=TIME_CONST*torch.ones((times.shape[0], 1, 1), device=times.device))
                    priv = (mask_len*temp)
                    dt_mean = torch.masked_select(priv, mask_len).mean()
                    dt_median = torch.masked_select(priv, mask_len).median()

                    # loss_abs_bl
                    loss_abs_bl += self.time_loss(
                        times+dt_median, times, seq_len).item()

                else:
                    times_pred = times
                    loss4 = self.time_loss(gt, times, seq_len)
                miss_loss2 += loss4.item()

                loss1 = self.sta_loss(out_sta, sta)
                loss2 = self.dyn_loss(out_dyn, dyn, seq_len, mask)

                sta_num = len(self.static_processor.models)
                dyn_num = len(self.dynamic_processor.models)
                scale1 = sta_num / (sta_num + dyn_num)
                scale2 = dyn_num / (sta_num + dyn_num)
                scale3 = 0.1

                loss = scale1 * loss1 + scale2 * \
                    (loss2 + loss3) + scale3 * loss4 + self.ae.KLD*0
                # loss = loss1+loss2 + self.ae.KLD*5
                # loss = loss1 + loss2 + loss3 + loss4
                if i > 0:
                    loss.backward()
                    self.ae_optm.step()

                tot_loss += loss.item()
                con_loss += loss1.item()
                dis_loss += loss2.item()
                KLD_loss += self.ae.KLD.item()

                tot += 1

            tot_loss /= tot
            wandb.log({"ae_loss": tot_loss, "KLD loss": KLD_loss/tot,
                      "static_loss": con_loss/tot, "dynamic_loss": dis_loss/tot, "miss_loss": miss_loss1/tot, "time_loss": miss_loss2/tot, "loss_abs": loss_abs/tot, "loss_abs_bl": loss_abs_bl/tot, "miss_loss2_bl": miss_loss2_bl/tot, "loss_mdn_bl": loss_mdn_bl/tot}, step=i)

            if i % 5 == 0:
                self.logger.info("Epoch:{} {}\t{}\t{}\t{}\t{}\t{}".format(
                    i+1, time.time()-t1, force, con_loss/tot, dis_loss/tot, miss_loss1/tot, miss_loss2/tot))
            if i % 5 == 0:
                seq_len_pred = self.static_processor.inverse_transform(
                    out_sta.detach().cpu().numpy()).iloc[:, -1].values

                fig_AE_rec, fig2 = self.plot_ae(
                    dyn, times, mask, seq_len.detach().cpu().numpy(),
                    out_dyn, gt, missing, seq_len_pred)

                # x_values = torch.masked_select(torch.arange(
                #     out_dyn.shape[1], device=mask.device), mask[0, :, 0] == 1)
                # y_values = torch.masked_select(
                #     dyn[0, :, 0], mask[0, :, 0] == 1)
                # times [bs, max_len, 1], dyn [bs, max_len, n_features], out_dyn [bs, max_len, n_features]
                out = self.ae.encoder(
                    sta, dyn, priv, nex, mask, times, seq_len, dt=dt)  # [bs, hidden_dim]
                if isinstance(out, tuple):
                    mu, logvar = out
                    real_rep = self.ae.reparameterize(mu, logvar)

                    # plot mu and logvar
                    fig_vae = go.Figure()
                    fig_vae.add_trace(
                        go.Bar(x=np.arange(mu.shape[1]), y=mu.mean(0).cpu().detach().numpy(), name='mu'))

                    fig_vae.add_trace(
                        go.Bar(x=np.arange(logvar.shape[1]), y=logvar.mean(0).cpu().detach().numpy(), name='logvar'))
                    fig_vae.update_layout(yaxis_range=[-0.1, 0.1])

                    # plot_vae = wandb.Plotly(fig_vae)
                    # wandb.log({"vae": plot_vae}, step=i)

                    # plot AE synth
                    fig_AE_syn = self.save_sample_wandb(from_generator=False, test=(
                        real_rep, sta, dyn, lag, mask, priv, times, seq_len), dt=dt)
                    # wandb.log(
                    #     {"example_AE_syn": wandb.Plotly(fig_AE_syn)}, step=i)
                else:
                    real_rep = out
                # plot t-SNE
                tsne = TSNE(n_components=2, perplexity=30,
                            learning_rate=10, n_jobs=4)
                X = real_rep.cpu().detach().numpy()  # [bs, hidden_dim]
                X_tsne = tsne.fit_transform(X)  # [2*bs, 2]
                fig_tsne = go.Figure()
                fig_tsne.add_trace(go.Scatter(
                    x=X_tsne[:, 0], y=X_tsne[:, 1], mode='markers', name='real'))

                # plot_tsne = wandb.Plotly(fig_tsne)

                wandb.log(
                    {"example_AE_syn": wandb.Plotly(fig_AE_syn),
                     "vae": wandb.Plotly(fig_vae),
                     "example_rec": wandb.Plotly(fig_AE_rec),
                     "dt_corr": wandb.Plotly(fig2),
                     "tsne_AE": wandb.Plotly(fig_tsne)}, step=i)
            if i % 100 == 99:
                torch.save(self.ae.state_dict(),
                           '{}/ae{}.dat'.format(self.params["root_dir"], i))
                self.generate_ae(dataset[:100])

        torch.save(self.ae.state_dict(),
                   '{}/ae.dat'.format(self.params["root_dir"]))

    def train_gan2(self, dataset, iterations=15000, d_update=5):
        self.discriminator.train()
        self.generator.train()
        self.ae.train()
        batch_size = self.params["gan_batch_size"]
        idxs = list(range(len(dataset)))
        batch = DataSetIter(
            dataset=dataset, batch_size=batch_size, sampler=RandomSampler())
        min_loss = 1e15
        for iteration in tqdm(range(iterations)):
            avg_d_loss = 0
            t1 = time.time()
            toggle_grad(self.generator, False)
            toggle_grad(self.discriminator, True)
            bce_loss = nn.BCEWithLogitsLoss().to(self.device)

            for j in range(d_update):
                for batch_x, batch_y in batch:
                    self.discriminator_optm.zero_grad()
                    z = torch.randn(
                        batch_size, self.params['noise_dim']).to(self.device)

                    sta = batch_x["sta"].to(self.device)
                    dyn = batch_x["dyn"].to(self.device)
                    mask = batch_x["mask"].to(self.device)
                    lag = batch_x["lag"].to(self.device)
                    priv = batch_x["priv"].to(self.device)
                    nex = batch_x["nex"].to(self.device)
                    times = batch_x["times"].to(self.device)
                    seq_len = batch_x["seq_len"].to(self.device)

                    out = self.ae.encoder(
                        sta, dyn, priv, nex, mask, times, seq_len)
                    if isinstance(out, tuple):
                        mu, logvar = out
                        real_rep = self.ae.reparameterize(mu, logvar)
                    else:
                        real_rep = out
                    d_real = self.discriminator(real_rep)
                    real_labels = torch.ones_like(d_real)
                    # dloss_real = -d_real.mean()
                    d_loss_real = bce_loss(d_real, real_labels)
                    # dloss_real.backward()

                    """
                    dloss_real.backward(retain_graph=True)
                    reg = 10 * compute_grad2(d_real, real_rep).mean()
                    reg.backward()
                    """

                    # On fake data
                    # with torch.no_grad():
                    x_fake = self.generator(z)

                    # x_fake.requires_grad_()
                    d_fake = self.discriminator(x_fake.detach())
                    fake_labels = torch.zeros_like(d_fake)
                    d_loss_fake = bce_loss(d_fake, fake_labels)

                    # Backpropagation and optimization for Discriminator
                    disc_loss = (d_loss_real + d_loss_fake)/2
                    disc_loss.backward()

                    # dloss_fake = d_fake.mean()
                    # """
                    # y = d_fake.new_full(size=d_fake.size(), fill_value=0)
                    # dloss_fake = F.binary_cross_entropy_with_logits(d_fake, y)
                    # """
                    # # dloss_fake.backward()
                    # """
                    # dloss_fake.backward(retain_graph=True)
                    # reg = 10 * compute_grad2(d_fake, x_fake).mean()
                    # reg.backward()
                    # """
                    reg = 10 * self.wgan_gp_reg(real_rep, x_fake)
                    reg.backward()

                    self.discriminator_optm.step()
                    # d_loss = dloss_fake + dloss_real
                    avg_d_loss += disc_loss.item()
                    break

            avg_d_loss /= d_update

            toggle_grad(self.generator, True)
            toggle_grad(self.discriminator, False)
            self.generator_optm.zero_grad()
            z = torch.randn(batch_size, self.params['noise_dim']).to(
                self.device)  # batch_size, noise_dim
            x_fake = self.generator(z)  # batch_size, hidden_dim
            d_fake = self.discriminator(x_fake)  # batch_size, 1
            # g_loss = -torch.mean(self.discriminator(x_fake))  # [batch,1]->scalar
            # g_loss.backward()
            # self.generator_optm.step()
            # Generator's loss
            g_loss = bce_loss(d_fake, real_labels)
            g_loss2 = (bce_loss(d_fake, fake_labels) +
                       bce_loss(d_real, real_labels))/2
            # Backpropagation and optimization for Generator
            self.generator.zero_grad()
            g_loss.backward()
            self.generator_optm.step()

            wandb.log({"d_loss": avg_d_loss, "g_loss": g_loss.item(), "g_loss2": g_loss2.item()},
                      step=iteration+1)
            if iteration % 50 == 49:
                self.logger.info('[Iteration %d/%d] [%f] [D loss: %f] [G loss: %f] [%f]' % (
                    iteration, iterations, time.time()-t1, avg_d_loss, g_loss.item(), reg.item()
                ))

            if iteration % 10 == 0:
                # table = self.save_sample_wandb()
                # wandb.log(
                #     {"example_syn": wandb.plot.line(table, "t", "y",
                #                                     title="Example Synthesized Time Series")}, step=iteration+1)

                # plot one generated sample
                # plot = self.save_sample_wandb(
                #     seq_len=batch_x['seq_len'][0].item())
                fig = self.save_sample_wandb()
                wandb.log(
                    {"example_syn": wandb.Plotly(fig)}, step=iteration+1)

                # plot t-SNE of latent space
                plot = self.plot_tsne(real_rep.cpu().detach(
                ).numpy(), x_fake.cpu().detach().numpy())
                wandb.log(
                    {"tsne": plot}, step=iteration+1)

        torch.save(self.generator.state_dict(),
                   '{}/generator.dat'.format(self.params["root_dir"]))

    def plot_tsne(self, real_rep, x_fake):

        batch_size = real_rep.shape[0]
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, n_jobs=4)

        X = np.concatenate([real_rep, x_fake], axis=0)  # [2*bs, hidden_dim]
        X_tsne = tsne.fit_transform(X)  # [2*bs, 2]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_tsne[:batch_size, 0], y=X_tsne[:batch_size, 1], mode='markers', name='real'))
        fig.add_trace(go.Scatter(
            x=X_tsne[batch_size:, 0], y=X_tsne[batch_size:, 1], mode='markers', name='fake'))
        plot = wandb.Plotly(fig)

        return plot

    def save_sample_wandb(self, from_generator=True, test=None, dt=None):
        # sta, dyn = self.synthesize(1)
        # test = (real_rep, sta, dyn, lag, mask, priv, times, seq_len)
        # # # x_values = dyn[0].time.values
        # # # y_values = dyn[0].S1.values

        # # # fig = go.Figure()
        # # # fig.add_trace(go.Scatter(
        # # #     x=x_values, y=dyn[0].S1.values, mode='lines+markers', name='S1'))
        # # # fig.add_trace(go.Scatter(
        # # #     x=x_values, y=dyn[0].S2.values, mode='lines+markers', name='S2'))

        # if (test is not None) and False:
        #     sta, dyn, missing, gt = self.ae.decoder(*test, dt=dt)
        # else:
        #     sta, dyn = self.synthesize(
        #         9,  from_generator=from_generator, test=test, dt=dt)

        fig = make_subplots(rows=3, cols=3)
        # for i in range(3):
        #     for j in range(3):
        #         x_values = dyn[i*3+j].time.values
        #         y_values1 = dyn[i*3+j].S1.values
        #         y_values2 = dyn[i*3+j].S2.values
        #         fig.add_trace(go.Scatter(
        #             x=x_values, y=y_values1, mode='lines+markers', name='S1', line=dict(color='blue')), row=i+1, col=j+1)
        #         fig.add_trace(go.Scatter(
        #             x=x_values, y=y_values2, mode='lines+markers', name='S2', line=dict(color='red')), row=i+1, col=j+1)

        # plot = wandb.Plotly(fig)
        return fig

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=self.device).view(batch_size, -1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

    def synthesize(self, n, batch_size=500):
        pass

    def eval_ae(self, dataset):
        batch_size = self.params["gan_batch_size"]
        idxs = list(range(len(dataset)))
        batch = DataSetIter(
            dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())
        res = []
        h = []
        for batch_x, batch_y in batch:
            with torch.no_grad():
                sta = batch_x["sta"].to(self.device)
                dyn = batch_x["dyn"].to(self.device)
                mask = batch_x["mask"].to(self.device)
                lag = batch_x["lag"].to(self.device)
                priv = batch_x["priv"].to(self.device)
                nex = batch_x["nex"].to(self.device)
                times = batch_x["times"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)

                hidden = self.ae.encoder(
                    sta, dyn, priv, nex, mask, times, seq_len)
                h.append(hidden)
        h = torch.cat(h, dim=0).cpu().numpy()
        return h

    def gen_hidden(self, n):
        self.ae.decoder.eval()
        self.generator.eval()
        batch_size = self.params["gan_batch_size"]
        h = []

        def _gen(n):
            with torch.no_grad():
                z = torch.randn(n, self.params['noise_dim']).to(self.device)
                hidden = self.generator(z)
                h.append(hidden)

        tt = n // batch_size
        for i in range(tt):
            _gen(batch_size)
        res = n - tt * batch_size
        if res > 0:
            _gen(res)
        h = torch.cat(h, dim=0).cpu().numpy()  # n, hidden_dim
        return h

# Utility functions


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
