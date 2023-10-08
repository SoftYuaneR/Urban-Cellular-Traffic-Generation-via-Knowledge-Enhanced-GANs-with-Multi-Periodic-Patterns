import os, sys
import time
import copy
import matplotlib
import random

matplotlib.use('SVG')
import torch
import math
import warnings
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter  # import Parameter to create custom activations with learnable parameters
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import setproctitle

setproctitle.setproctitle("")
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

torch.manual_seed(5)
use_cuda = torch.cuda.is_available()
from RESGAN_Partly import TemporalConvNet
from RESGAN_Partly import DiscriminatorTCN
from RESGAN_Partly import Discriminator_HD
from RESGAN_Partly import Discriminator_DW
from RESGAN_Partly import MyDataset


class GeneratorP_HD_SUM(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns=0, pattern_length=24):
        super(GeneratorP_HD_SUM, self).__init__()
        self.linear_4hd = nn.Linear(NOISE_SIZE + KGE_SIZE, pattern_num) if USE_KGE else \
            nn.Linear(NOISE_SIZE, pattern_num)
        #        self.linear_4hdres = nn.Linear(32+KGE_SIZE, 24)
        self.norm_4hd = nn.LayerNorm(pattern_num)
        #        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        #        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(pattern_length)
        self.act = nn.Sigmoid() if activation == 'sigmoid' else nn.GELU()
        self.hd_patterns = torch.empty(pattern_num, pattern_length)
        self.hd_patterns.requires_grad = True
        self.init_weights()
        self.patterns = patterns
        if self.patterns:
            self.hd_patterns = patterns

    def init_weights(self):
        nn.init.orthogonal_(self.linear_4hd.weight)
        nn.init.orthogonal_(self.hd_patterns)

    def init_patterns(self, patterns):
        if not self.patterns:
            self.hd_patterns = patterns

    def forward(self, x):
        #        BZ = x.shape[0]
        #        print(x.shape)
        x_p = self.softmax(self.norm_4hd(self.linear_4hd(x))) if USE_KGE else \
            self.softmax(self.norm_4hd(self.linear_4hd(x[:, 0:32])))  # [:, 32:])))
        hours_in_day = self.act(self.norm(
            x_p @ self.hd_patterns))  # + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))
        #        hours_in_day = self.act(self.norm(x_p@self.hd_patterns))# + 0.01*self.tanh(self.linear_4hdres(torch.cat((x[:,0:32], x[:,64:]), 1))))
        return hours_in_day

class GeneratorP_DW_SUM(nn.Module):
    def __init__(self, pattern_num=32, dropout=0.1, activation="relu", patterns=0, pattern_length=168):
        super(GeneratorP_DW_SUM, self).__init__()
        self.linear_4dw = nn.Linear(NOISE_SIZE + KGE_SIZE, pattern_num) if USE_KGE else \
            nn.Linear(NOISE_SIZE, pattern_num)
        self.norm_4dw = nn.LayerNorm(pattern_num)
        self.norm = nn.LayerNorm(pattern_length)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        #        self.relu = nn.ReLU()
        self.act = nn.Sigmoid() if activation == 'sigmoid' else nn.GELU()
        self.dwr_patterns = torch.empty(pattern_num, pattern_length)
        self.dwr_patterns.requires_grad = True
        self.init_weights()
        self.patterns = patterns
        if self.patterns:
            self.dwr_patterns = patterns

    def init_weights(self):
        nn.init.orthogonal_(self.linear_4dw.weight)
        nn.init.orthogonal_(self.dwr_patterns)

    def init_patterns(self, patterns):
        if not self.patterns:
            self.dwr_patterns = patterns

    def forward(self, x, hours_in_day):
        #        BZ = x.shape[0]
        days_in_week_residual = self.softmax(self.norm_4dw(self.linear_4dw(x))) if USE_KGE else \
            self.softmax(self.norm_4dw(self.linear_4dw(x[:, 0:32])))
        days_in_week_residual = self.norm(days_in_week_residual @ self.dwr_patterns)
        days_in_week = days_in_week_residual + hours_in_day.repeat(1, int(
            days_in_week_residual.shape[1] / hours_in_day.shape[1]))
        return self.act(days_in_week)  # _weekend

class GeneratorP_WM_SUM(nn.Module):
    def __init__(self, input_size=20, num_channels=[1] * 6, kernel_size=[24, 2 * 24, 7 * 24], dropout=0.3, kge_size=32,
                 kge_squeeze_size=20, activation="relu"):
        super(GeneratorP_WM_SUM, self).__init__()
        self.kge_size = kge_size
        self.linear_kge = nn.Linear(self.kge_size, kge_squeeze_size)
        self.norm_kge = nn.LayerNorm(kge_squeeze_size)
        if USE_KGE:
            input_size = input_size + kge_squeeze_size
        self.tcn_d = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[0], dropout=dropout)
        self.tcn_w = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[1], dropout=dropout)
        self.tcn_m = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size[2], dropout=dropout)
        self.linear_4wm = nn.Linear(self.kge_size, kge_squeeze_size * 168)
        self.norm_4wm = nn.LayerNorm(kge_squeeze_size * 168)
        self.norm = nn.LayerNorm([1, 168])
        self.act = nn.Sigmoid() if activation == 'sigmoid' else nn.GELU()

        self.linear = nn.Linear(num_channels[-1] * len(kernel_size), num_channels[-1])
        self.init_weights()
        # self.relu = nn.ReLU()
        # self.soft_exponential = soft_exponential(num_channels[-1], alpha = 1.0)
        self.tanh = nn.Tanh()


    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, days_in_week):
        BZ = x.shape[0]
        kge = self.norm_kge(self.linear_kge(x[:, self.kge_size:]))
        x = self.norm_4wm(self.linear_4wm(x[:, 0:self.kge_size])).reshape(BZ, -1, 168)
        x = torch.cat((x, kge.view(kge.size(0), kge.size(1), 1).expand(-1, -1, x.size(2))), 1) if USE_KGE else x
        y_d = self.tcn_d(x)
        y_w = self.tcn_w(x)
        y_m = self.tcn_m(x)
        y = self.norm(self.linear(torch.cat((y_d, y_w, y_m), 1).transpose(1, 2)).transpose(1, 2))
        y = y + days_in_week.reshape(BZ, 1, -1)#.repeat(1, 4).reshape(BZ, 1, -1)
        return self.act(y.squeeze(1))  # , y_d, y_w, y_m, kge

class GeneratorP_ALL_LN_Matrioska(nn.Module):  #
    def __init__(self, nhead=[1, 1, 1], dim_feedforward=2048, dropout=0.3, activation="relu", num_layers=6):
        super(GeneratorP_ALL_LN_Matrioska, self).__init__()
        self.generator_hdd = GeneratorP_HD_SUM(activation=activation, pattern_length=24)
        self.generator_hde = GeneratorP_HD_SUM(activation=activation, pattern_length=24)
        self.generator_dwd = GeneratorP_DW_SUM(activation=activation, pattern_length=24*5)
        self.generator_dwe = GeneratorP_DW_SUM(activation=activation, pattern_length=24*2)
        self.generator_wm = GeneratorP_WM_SUM(activation=activation,kge_size=KGE_SIZE)
        # self.norm = nn.LayerNorm(
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=1)
        # self.relu = nn.ReLU()

    def init_patterns(self, hours_in_weekday_patterns, hours_in_weekend_patterns, days_in_weekday_residual_patterns, days_in_weekend_residual_patterns):
        self.generator_hdd.init_patterns(hours_in_weekday_patterns)
        self.generator_hde.init_patterns(hours_in_weekend_patterns)
        self.generator_dwd.init_patterns(days_in_weekday_residual_patterns)
        self.generator_dwe.init_patterns(days_in_weekend_residual_patterns)


    def forward(self, x):
        BZ = x.shape[0]
        hours_in_weekday = self.generator_hdd(x)
        hours_in_weekend = self.generator_hde(x)
        days_in_weekday = self.generator_dwd(x, hours_in_weekday)
        days_in_weekend = self.generator_dwe(x, hours_in_weekend)
        days_in_week = torch.cat((days_in_weekday, days_in_weekend), 1)
        tfc = self.generator_wm(x, days_in_week)
        #        tfc = hours_in_day.repeat(1, 4*7).reshape(BZ,1,-1) + days_in_week_residual.repeat(1, 4).reshape(BZ,1,-1) + weeks_in_month_residual
        return hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, tfc

def calc_gradient_penalty(netD, real_data, fake_data, kge):
    # use_cuda = 0
    LAMBDA = 10
    alpha = torch.rand(BATCH_SIZE, 1)

    if use_cuda:
        alpha = alpha.cuda(gpu)  # .to('cuda:6')# if use_cuda else alpha
    alpha = alpha.expand(real_data.size())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #    if use_cuda:
    #        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, kge)

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def generate_data(netG, kge, gene_size=128):
    noise = torch.randn(gene_size, NOISE_SIZE)  # , LENGTH)#torch.randn(BATCH_SIZE, NOISE_SIZE, dim_list_g[0])
    noise = noise.exponential_() if EXP_NOISE else noise
    if use_cuda:
        noise = noise.cuda(gpu)
        kge = kge.cuda(gpu)
    hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, output = netG(
        torch.cat((noise, kge), 1))  # , kge)
    return hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, output


work_dir = ''
gpu = 0

EXP_NOISE = True
USE_KGE = True
KGE_SIZE = 160
NOISE_SIZE = KGE_SIZE
LENGTH = 168
BATCH_SIZE = 32
BATCH_FIRST = True
node_degree_list = [10,12,14]
for node_degree in node_degree_list:
    save_dir_head = 'generate_data_KL_log_layerscat_d'+str(node_degree)
    node_file = os.path.join(work_dir, 'node_embedding/node_embedding_KL_log_layerscat_d' + str(node_degree) + '.npz')
    node_data = np.load(node_file)['node_embedding']
    dataset = MyDataset(data=os.path.join(work_dir, 'bs_record_w.npz'),
                        node_data=node_data)
    DATASET_SIZE = len(dataset)
    gene_size = DATASET_SIZE  # 1024

    real_dataset_list = [dataset.data['hours_in_weekday'], dataset.data['hours_in_weekend'], dataset.data['days_in_weekday'],
                         dataset.data['days_in_weekend'], dataset.data['bs_record']]
    hours_in_weekday_patterns = dataset.hours_in_weekday_patterns
    hours_in_weekend_patterns = dataset.hours_in_weekend_patterns
    days_in_weekday_patterns = dataset.days_in_weekday_patterns
    days_in_weekend_patterns = dataset.days_in_weekend_patterns
    days_in_weekday_residual_patterns = dataset.days_in_weekday_residual_patterns
    days_in_weekend_residual_patterns = dataset.days_in_weekend_residual_patterns
    if use_cuda:
        hours_in_weekday_patterns = hours_in_weekday_patterns.cuda(gpu)
        hours_in_weekend_patterns = hours_in_weekend_patterns.cuda(gpu)
        days_in_weekday_patterns = days_in_weekday_patterns.cuda(gpu)
        days_in_weekend_patterns = days_in_weekend_patterns.cuda(gpu)
        days_in_weekday_residual_patterns = days_in_weekday_residual_patterns.cuda(gpu)
        days_in_weekend_residual_patterns = days_in_weekend_residual_patterns.cuda(gpu)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    dropout = 0.3
    num_layers = 2
    netG = GeneratorP_ALL_LN_Matrioska()
    netG.init_patterns(hours_in_weekday_patterns, hours_in_weekend_patterns, days_in_weekday_residual_patterns,
                       days_in_weekend_residual_patterns)
    netD_0 = Discriminator_HD(patterns=hours_in_weekday_patterns)
    netD_1 = Discriminator_HD(patterns=hours_in_weekend_patterns)
    netD_2 = Discriminator_DW(
        patterns=days_in_weekday_patterns)  # days_in_weekday_residual_patterns+hours_in_weekday_patterns.repeat(1,5))
    netD_3 = Discriminator_DW(
        patterns=days_in_weekend_patterns)  # days_in_weekend_residual_patterns+hours_in_weekend_patterns.repeat(1,2))
    netD_4 = DiscriminatorTCN(kge_size=KGE_SIZE)
    start_it = 0

    LOAD_PRE_TRAIN = False
    LAMBDA = 10
    ITERS = 10#331
    CRITIC_ITERS = 5
    tb_writer = SummaryWriter()
    if use_cuda:
        netD_0 = netD_0.cuda(gpu)
        netD_1 = netD_1.cuda(gpu)
        netD_2 = netD_2.cuda(gpu)
        netD_3 = netD_3.cuda(gpu)
        netD_4 = netD_4.cuda(gpu)
        netG = netG.cuda(gpu)
    netD_list = [netD_0, netD_1, netD_2, netD_3, netD_4]

    sub_dir_list = ['HDD', 'HDE', 'DWD', 'DWE', 'ALL']
    ii_list = np.arange(len(netD_list))

    if node_degree == 10:
        pre_train_dir = ''
        pretrained_netG = torch.load(os.path.join(pre_train_dir,'netG'), map_location=torch.device(gpu))
        #print(pretrained_netG)
        #print(netG.generator_hdd.hd_patterns)
        netG.load_state_dict(pretrained_netG)
        netG.generator_dwd.dwr_patterns = torch.load(os.path.join(pre_train_dir,'netG.generator_dwd.dwr_patterns'))
        netG.generator_dwe.dwr_patterns = torch.load(os.path.join(pre_train_dir, 'netG.generator_dwe.dwr_patterns'))
        netG.generator_hdd.hd_patterns = torch.load(os.path.join(pre_train_dir, 'netG.generator_hdd.hd_patterns'))
        netG.generator_hde.hd_patterns = torch.load(os.path.join(pre_train_dir, 'netG.generator_hde.hd_patterns'))
        #print(netG.generator_hdd.hd_patterns)

        print('ALL loaded!')
        '''
        weights = [netG.generator_dwd.linear_4dw.weight, netG.generator_dwe.linear_4dw.weight,
                   netG.generator_hdd.linear_4hd.weight, netG.generator_hde.linear_4hd.weight,
                   netG.generator_wm.linear_kge.weight, netG.generator_wm.linear_4wm.weight,
                   netG.generator_wm.linear.weight]
        for w in weights:
            print(w.shape, w.T@w)#torch.linalg.matrix_rank(w))
        '''
        ii_list = [4]

    # if False:
    for ii in ii_list: #np.arange(len(netD_list)):
        TimeList = []
        D_costList = []
        G_costList = []
        sparsityList = []
        WDList = []
        dst_list = []
        netD = netD_list[ii]
        sub_dir = sub_dir_list[ii]
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        one = torch.tensor(1, dtype=torch.float32)
        mone = one * -1.0
        if use_cuda:
            one = one.cuda(gpu)
            mone = mone.cuda(gpu)
        print(time.localtime())
        for iteration in trange(ITERS):
            start_time = time.time()
            # print(time.localtime(), ' iteration: ', iteration)
            for idx, data in enumerate(data_loader):
                if True:  # (idx < 1500) | (idx%(CRITIC_ITERS*3) == 0):
                    ############################
                    # (1) Update D network
                    ######f#####################
                    for p in netD.parameters():  # reset requires_grad
                        p.requires_grad = True  # they are set to False below in netG update

                    # id_batch = data[0]
                    data_batch, kge_batch, hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend = data[
                                                                                                                  1:]
                    data_batch = data_batch.squeeze(1)
                    #                if not(BATCH_FIRST):
                    #                    data_batch = data_batch.permute(2,0,1)

                    netD.zero_grad()

                    real_data = [hours_in_weekday, hours_in_weekend, days_in_weekday, days_in_weekend, data_batch]
                    # kge = kge_batch
                    if use_cuda:
                        real_data[ii] = real_data[ii].cuda(gpu)
                        kge_batch = kge_batch.cuda(gpu)
                    D_real = netD(real_data[ii], kge_batch)
                    D_real = D_real.mean()
                    # print D_real
                    # TODO: Waiting for the bug fix from pytorch
                    D_real.backward(mone)

                    # generate noise
                    noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)  # , LENGTH)
                    noise_batch = noise_batch.exponential_() if EXP_NOISE else noise_batch
                    if use_cuda:
                        noise_batch = noise_batch.cuda(gpu)
                        # noise_kge = torch.cat((noise_batch, kge_batch.view(kge_batch.size(0),kge_batch.size(1),1).expand(-1, -1, LENGTH)), 1)
                    # noisev = autograd.Variable(noise, volatile=True)
                    # train with fake
                    fake_data = netG(torch.cat((noise_batch, kge_batch), 1))
                    #                fake_weeks_in_month_residual = netG(noise_batch, kge_batch)
                    D_fake = netD(fake_data[ii], kge_batch)
                    D_fake = D_fake.mean()
                    # TODO: Waiting for the bug fix from pytorch
                    D_fake.backward(one)

                    # train with gradient penalty
                    # print(fake_data.view(BATCH_SIZE,LENGTH).shape)
                    gradient_penalty = calc_gradient_penalty(netD, real_data[ii], fake_data[ii], kge_batch)
                    gradient_penalty.backward()

                    D_cost = D_fake - D_real + gradient_penalty
                    Wasserstein_D = D_real - D_fake
                    optimizerD.step()
                    tb_writer.add_scalar('loss\D_cost', D_cost, iteration)
                    tb_writer.add_scalar('loss\Wasserstein_D', Wasserstein_D, iteration)
                    # print('#######Wasserstein_D###########',Wasserstein_D)

                if idx % CRITIC_ITERS == 0:
                    ############################
                    # (2) Update G network
                    ###########################
                    for p in netD.parameters():
                        p.requires_grad = False  # to avoid computation
                    netG.zero_grad()

                    noise_batch = torch.randn(BATCH_SIZE, NOISE_SIZE)  # , LENGTH)
                    noise_batch = noise_batch.exponential_() if EXP_NOISE else noise_batch
                    if use_cuda:
                        noise_batch = noise_batch.cuda(gpu)
                    fake = netG(torch.cat((noise_batch, kge_batch), 1))
                    #                fake_weeks_in_month_residual = netG(noise_batch, kge_batch)
                    for fake_value in fake[0:ii]:
                        fake_value.detach_()  # to avoid computation
                    G = netD(fake[ii], kge_batch)
                    G = G.mean()
                    G.backward(mone)
                    G_cost = -G
                    optimizerG.step()
                    tb_writer.add_scalar('loss\G_cost', G_cost, iteration)
                    # print(G_cost, D_cost, Wasserstein_D)
                    G_costList.append(G_cost.cpu().data.numpy())
                    # WDList.append(Wasserstein_D.cpu().data.numpy())
                TimeList.append(time.time() - start_time)
                D_costList.append(D_cost.cpu().data.numpy())
                ##SD_costList.append(SD_cost.cpu().data.numpy())
                WDList.append(Wasserstein_D.cpu().data.numpy())
                # print(fake.max(), fake.min())
            if iteration % 10 == 0:  # True
                save_dir = save_dir_head + '/' + sub_dir + '/iteration-' + str(iteration + start_it)
                save_dir = os.path.join(work_dir, save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(netD.state_dict(), os.path.join(save_dir, 'netD'))
                torch.save(netG.state_dict(), os.path.join(save_dir, 'netG'))
                torch.save(netG.generator_hdd.hd_patterns, os.path.join(save_dir, 'netG.generator_hdd.hd_patterns'))
                torch.save(netG.generator_hde.hd_patterns, os.path.join(save_dir, 'netG.generator_hde.hd_patterns'))
                torch.save(netG.generator_dwd.dwr_patterns, os.path.join(save_dir, 'netG.generator_dwd.dwr_patterns'))
                torch.save(netG.generator_dwe.dwr_patterns, os.path.join(save_dir, 'netG.generator_dwe.dwr_patterns'))

                np.savez(os.path.join(save_dir, 'cost_generated.npz'),
                         time=np.array(TimeList),
                         D_cost=np.array(D_costList),
                         sparsity=np.array(sparsityList),
                         G_cost=np.array(G_costList),
                         WD=np.array(WDList))

                fake_data = generate_data(netG, dataset.kge, gene_size)
                #                     generate_data(netG, dataset.kge[random.sample(range(DATASET_SIZE), gene_size)], gene_size)
                generated_data = fake_data[4].reshape(gene_size, -1).cpu().detach().numpy()
                # kge_used = kge_gene.cpu().detach().numpy()
                hours_in_weekday = fake_data[0].view(gene_size, 24).cpu().detach().numpy()
                hours_in_weekend = fake_data[1].view(gene_size, 24).cpu().detach().numpy()
                days_in_weekday = fake_data[2].view(gene_size, 24 * 5).cpu().detach().numpy()
                days_in_weekend = fake_data[3].view(gene_size, 24 * 2).cpu().detach().numpy()
                fig, ax = plt.subplots(figsize=(24, 16))
                n_bins = 100
                line_w = 2
                use_cumulative = -1
                use_log = True
                n_real, _, _ = ax.hist(real_dataset_list[ii].flatten(), n_bins, density=True, histtype='step',
                                                cumulative=use_cumulative, label='real', log=use_log, facecolor='g',
                                                linewidth=line_w)
                fake_data_now = fake_data[ii].cpu().detach().numpy()
                n_gene, _, _ = ax.hist(fake_data_now.flatten(), n_bins, density=True, histtype='step',
                                                cumulative=use_cumulative, label='gene', log=use_log, facecolor='b',
                                                linewidth=line_w)
                ax.grid(True)
                ax.legend(loc='right')
                ax.set_title('Cumulative step histograms')
                ax.set_xlabel('Value')
                ax.set_ylabel('Likelihood of occurrence')
                plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
                plt.close()
                dst = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
                tb_writer.add_scalar('loss\dst', dst, iteration)
                dst_list.append(dst)
                np.savez(os.path.join(save_dir, 'generated.npz'),
                         generated_data=generated_data,
                         distance=dst,
                         distances=np.array(dst_list),
                         # kge_used = np.array(kge_used),
                         hours_in_weekday=hours_in_weekday,
                         hours_in_weekend=hours_in_weekend,
                         days_in_weekend=days_in_weekend,
                         days_in_weekday=days_in_weekday)
                # print(G_cost, D_cost, Wasserstein_D, dst)#, sparsity)
                fig_f_samples = plt.figure(figsize=(24, 16))
                plt.plot(generated_data[0::256])
                tb_writer.add_figure('fig\fig_f\fig_f_samples', fig_f_samples, iteration)
                plt.close()
        print(sub_dir + ' finished!')
    print('Train finished!')


