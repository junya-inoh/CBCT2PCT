# %%
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# my module
from dataset import ImageTransform
from dataset import DCMDataset
from utils import get
from utils import HUrescale
from param import Params
prm = Params()
from torch.utils.tensorboard import SummaryWriter

import datetime
now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
print(now)
# %%
root_trainA = prm.root_trainA
root_trainB = prm.root_trainB
root_testA = prm.root_testA
root_testB = prm.root_testB

# %%
trainA_list = get(prm.root_trainA).__call__(extension='.dcm')
trainB_list = get(prm.root_trainB).__call__(extension='.dcm')
testA_list = get(prm.root_testA).__call__(extension='.dcm')
testB_list = get(prm.root_testB).__call__(extension='.dcm')
# print(testB_list)


# %%

train_dataset = DCMDataset(trainA_list, trainB_list, transform=ImageTransform(), phase='train')
test_dataset = DCMDataset(testA_list, testB_list, transform=ImageTransform(), phase='test')
# print(test_dataset[0])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=prm.batch_size, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=prm.batch_size, shuffle=False
)

# %%
img = test_dataset.__getitem__(index=50)['A']
print(np.min(img), np.max(img))
print(img.shape)
print(img.dtype)
print(type(img))
plt.imshow(img, cmap=plt.cm.gray)
plt.colorbar()
plt.show()

img_transformed = test_dataset.__getitem__(index=50)['Atf']
print(torch.min(img_transformed), torch.max(img_transformed))
print(img_transformed.shape)
print(img_transformed.dtype)
print(type(img_transformed))
plt.imshow(img_transformed.numpy()[0], cmap=plt.cm.gray)
plt.colorbar()
plt.show()
# %%
real_A_tf_list = []
real_B_tf_list = []
for i, batch in enumerate(train_dataloader):
    real_A = batch['A']
    real_B = batch['B']
    real_A_tf = batch['Atf']
    real_B_tf = batch['Btf']

    break

print(torch.min(real_A_tf), torch.max(real_A_tf))
print(real_A_tf.dtype)
print(type(real_A_tf))
print(real_A_tf.shape)
plt.imshow(real_A_tf.numpy()[0, 0], cmap=plt.cm.gray)
plt.colorbar()
plt.show()

print(torch.min(real_B_tf), torch.max(real_B_tf))
print(real_B_tf.dtype)
print(type(real_B_tf))
print(real_B_tf.shape)
plt.imshow(real_B_tf.numpy()[0, 0], cmap=plt.cm.gray)
plt.colorbar()
plt.show()


# %%
# len(real_A_tf_list)
# %%
import datetime
now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# 出力ディレクトリ
out_dir = f'./results_check/{now}'
os.makedirs(out_dir, exist_ok=True)

writer = SummaryWriter(log_dir='./' + str(out_dir) + '/logs/image')
img_grid_A = torchvision.utils.make_grid(real_A_tf)
print(torch.min(img_grid_A), torch.max(img_grid_A))
print(img_grid_A.dtype)
print(type(img_grid_A))
print(img_grid_A.shape)
plt.imshow(img_grid_A.numpy()[0], cmap=plt.cm.gray)
plt.colorbar()
plt.show()
writer.add_image('img_train_A', (img_grid_A + 1)/2)
writer.close()


# img_grid_B = torchvision.utils.make_grid(real_B_tf_list[:25][0], nrow=5)
# writer.add_image('img_train_B', img_grid_B)
# writer.close()
# %%

