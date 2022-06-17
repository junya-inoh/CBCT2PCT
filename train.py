# %%
import os
import torch
import torchvision
from torch.autograd import Variable
import itertools
from torch.utils.tensorboard import SummaryWriter
import datetime
now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# print(now)

# my module
from dataset import ImageTransform
from dataset import DCMDataset
from net import Generator
from net import Discriminator

from utils import get
from utils import LambdaLR
from utils import ReplayBuffer
from utils import weights_init_normal
from param import Params
prm = Params()

# 確認用
# import matplotlib.pyplot as plt
# import numpy as np
# from utils import HUrescale

# %%
# root
root_trainA = prm.root_trainA
root_trainB = prm.root_trainB
root_testA = prm.root_testA
root_testB = prm.root_testB

# 各データのパスのリストを作成する
trainA_list = get(prm.root_trainA).__call__(extension='.dcm')
trainB_list = get(prm.root_trainB).__call__(extension='.dcm')
testA_list = get(prm.root_testA).__call__(extension='.dcm')
testB_list = get(prm.root_testB).__call__(extension='.dcm')

# 出力ディレクトリ
out_dir = f'./results/{now}'
os.makedirs(out_dir, exist_ok=True)

# Tensorboard writer の定義
writer_loss = SummaryWriter(log_dir='./' + str(out_dir) + '/logs/loss')
writer_img = SummaryWriter(log_dir='./' + str(out_dir) + '/logs/image')
writer_dis = SummaryWriter(log_dir='./' + str(out_dir) + '/logs/discriminator')
writer_gen = SummaryWriter(log_dir='./' + str(out_dir) + '/logs/generator')
# %%　データセットの作成
train_dataset = DCMDataset(trainA_list, trainB_list, transform=ImageTransform(), phase='train')
test_dataset = DCMDataset(testA_list, testB_list, transform=ImageTransform(), phase='test')

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=prm.batch_size, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=prm.batch_size, shuffle=False
)

# %%

# 生成器
netG_A2B = Generator(prm.input_nc, prm.output_nc)
netG_B2A = Generator(prm.output_nc, prm.input_nc)

# 識別器
netD_A = Discriminator(prm.input_nc)
netD_B = Discriminator(prm.output_nc)

# GPU
if torch.cuda.is_available():
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
else:
    print('GPU is not found')
    exit()


# %%
# 重みパラメータ初期化
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)


# TO DO: load learnt model ###################


# 損失関数
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
    lr=prm.lr, betas=(0.5, 0.999)
)
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=prm.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=prm.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(prm.n_epochs, prm.start_epoch, prm.decay_start_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(prm.n_epochs, prm.start_epoch, prm.decay_start_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(prm.n_epochs, prm.start_epoch, prm.decay_start_epoch).step)

# 入出力メモリ確保
Tensor = torch.cuda.FloatTensor
input_A = Tensor(prm.batch_size, prm.input_nc, prm.size, prm.size)
input_B = Tensor(prm.batch_size, prm.output_nc, prm.size, prm.size)
target_real = Variable(Tensor(prm.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(prm.batch_size).fill_(0.0), requires_grad=False)

# 過去データ分のメモリ確保
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
# %%
# for tensorboard
for i, batch in enumerate(train_dataloader):
    real_A_tf = Variable(input_A.copy_(batch['Atf']))
    real_B_tf = Variable(input_B.copy_(batch['Btf']))
    break

with writer_dis as wd:
    wd.add_graph(netD_A, real_A_tf[0].to())

with writer_gen as wg:
    wg.add_graph(netG_A2B, real_A_tf)




# %%
