import torch
from torchvision import transforms
from PIL import Image

from VGG_Model import vgg_mean, vgg_std
from Losses import BatchFeatureLoss_Model
import matplotlib.pyplot as plt
import numpy as np

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print("Balin-->", USE_CUDA)
device = torch.device("cuda:1" if USE_CUDA else "cpu")


lossFunc = BatchFeatureLoss_Model(device=device, c_alpha=1., s_beta=1.e4, s_layWei=[1., 1., 1., 1., 1.]).to(device)


def evaluResult (in_Dir, rst_Dir, style_Dir, mask_Dir, Frame0, Frame1):
    Img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(vgg_mean, vgg_std)])
    Mask_transform = transforms.Compose([transforms.ToTensor()])

    mask = Image.open(mask_Dir)
    mask = Mask_transform(mask)
    mask = mask[0, :, :].unsqueeze(0).unsqueeze(0).to(device)

    print(mask.size())

    rst_SLOSS = []
    in_SLOSS = []
    for f in range(Frame0, Frame1 + 1):
        imgN = in_Dir + str(f).zfill(7) + '.png'
        in_img = Image.open(imgN)
        in_img = Img_transform(in_img)
        in_img = in_img.unsqueeze(0).to(device)

        rstN = rst_Dir + str(f).zfill(7) + '.png'
        rst_img = Image.open(rstN)
        rst_img = Img_transform(rst_img)
        rst_img = rst_img.unsqueeze(0).to(device)

        styN = style_Dir + str(f).zfill(7) + '.png'
        sty_img = Image.open(styN)
        sty_img = Img_transform(sty_img)
        sty_img = sty_img.unsqueeze(0).to(device)

        _, rst_style_loss = lossFunc(X=rst_img, SG=sty_img, CX=in_img, MX=mask)
        _, in_style_loss = lossFunc(X=in_img, SG=sty_img, CX=in_img, MX=mask)

        rst_SLOSS.append(rst_style_loss.item())
        in_SLOSS.append(in_style_loss.item())

        print('Frame_', f, '--> rst_StyleL: ', rst_style_loss.item(), '--> in_StyleL: ', in_style_loss.item())

    return rst_SLOSS, in_SLOSS


def saveMeanStd(mean, std, fileName):
    with open(fileName, 'w') as f:
        f.write(str(mean) + '\n')
        f.write(str(std)+'\n')
        f.close()


Frame0 = 1
Frame1 = 200

#saveName = '../Data/case_1/score_DenimLight.svg'
F_Prefix = '../tableData/case_1/'
rst_Prefix = '../tableData/case_1/'

# DressA
caseName = 'UpDress/Chamuse/'
I_inDir = F_Prefix + caseName + 't_Ds10_L/'
I_StyleDir = F_Prefix + caseName + 't_10_L/'
I_rstDir = rst_Prefix + caseName + '/img_10_wool/'
I_maskDir = F_Prefix + caseName + 'Mask.png'

rst_SLOSS, in_SLOSS = evaluResult(in_Dir=I_inDir, rst_Dir=I_rstDir, style_Dir=I_StyleDir,
                                  mask_Dir=I_maskDir, Frame0=Frame0, Frame1=Frame1)

FX = [i for i in range(Frame0, Frame1+1)]

improvPerc_1 = [(1.-rst_SLOSS[i] / in_SLOSS[i]) * 100 for i in range(len(FX))]

meanImpro_1 = np.mean(improvPerc_1)
stdImpro_1 = np.std(improvPerc_1)
print(caseName, '-->', meanImpro_1, stdImpro_1)
#saveMeanStd(meanImpro_1, stdImpro_1, F_Prefix + caseName + '156_meanStd.txt')
minP = min(improvPerc_1)
maxP = max(improvPerc_1)

# DressB
#caseName = 'UpDress/DenimLight/'
I_inDir = F_Prefix + caseName + 't_Ds10_L/'
I_StyleDir = F_Prefix + caseName + 't_10_L/'
I_rstDir = rst_Prefix + caseName + '/img_10/'
I_maskDir = F_Prefix + caseName + 'Mask.png'

rst_SLOSS, in_SLOSS = evaluResult(in_Dir=I_inDir, rst_Dir=I_rstDir, style_Dir=I_StyleDir,
                                  mask_Dir=I_maskDir, Frame0=Frame0, Frame1=Frame1)

improvPerc_2 = [(1.-rst_SLOSS[i] / in_SLOSS[i]) * 100 for i in range(len(FX))]

meanImpro_2 = np.mean(improvPerc_2)
stdImpro_2 = np.std(improvPerc_2)
print(caseName, '-->', meanImpro_2, stdImpro_2)
#saveMeanStd(meanImpro_2, stdImpro_2, F_Prefix + caseName + 'ours_meanStd.txt')

minP = min(improvPerc_2 + [minP])
maxP = max(improvPerc_2 + [maxP])

# DressC
#caseName = 'ShortDress/DenimLight/'
# I_inDir = F_Prefix + caseName + 't_Ds10_L/'
# I_StyleDir = F_Prefix + caseName + 't_10_L/'
# I_rstDir = rst_Prefix + caseName + '/img_ps/'
# I_maskDir = F_Prefix + caseName + 'Mask.png'
#
# rst_SLOSS, in_SLOSS = evaluResult(in_Dir=I_inDir, rst_Dir=I_rstDir, style_Dir=I_StyleDir,
#                                   mask_Dir=I_maskDir, Frame0=Frame0, Frame1=Frame1)
#
# improvPerc_3 = [(1.-rst_SLOSS[i] / in_SLOSS[i]) * 100 for i in range(len(FX))]
#
# meanImpro_3 = np.mean(improvPerc_3)
# stdImpro_3 = np.std(improvPerc_3)
# print(caseName, '-->', meanImpro_3, stdImpro_3)
# #saveMeanStd(meanImpro_3, stdImpro_3, F_Prefix + caseName + 'meanStd.txt')
#
# minP = min(improvPerc_3 + [minP])
# maxP = max(improvPerc_3 + [maxP])

print('Worst improvement: ', minP)
minP = max([minP, 0])

#improvPerc_1 = [max(improvPerc_1[i], 0.) for i in range(len(improvPerc_1))]

#fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
fig, ax = plt.subplots()
#ax = axs.flat[0]
ax.set_title('Style Improvement varying with frames', fontsize=15, fontweight='demi')
ax.set_xlabel('Frame_t', fontsize=12)
ax.set_ylabel('%', fontsize=12)
ax.set_ylim([minP-5, 100])
ax.plot(FX, improvPerc_1, label='WrongMat', marker='.')
ax.plot(FX, improvPerc_2, label='RightMat', marker='.')
#ax.plot(FX, improvPerc_3, label='PS', marker='.')
ax.legend(loc='upper left')
plt.savefig(rst_Prefix + caseName + '/improve_score_WrongRight.svg')
plt.close(fig)

#ax = axs.flat[1]
fig, ax = plt.subplots()
ax.set_title('Style Improvement distribution', fontsize=15, fontweight='demi')
ax.set_ylabel('Frames', fontsize=12)
ax.set_xlabel('%', fontsize=12)
ax.set_xlim([minP-5, 100])
defbins = np.linspace(minP, maxP, 50, endpoint=True)
n, bins, patches = ax.hist(improvPerc_1, defbins, label='WrongMat', alpha=0.75)
n_, bins_, patches_ = ax.hist(improvPerc_2, defbins, label='RightMat', alpha=0.75)
#n_1, bins_1, patches_1 = ax.hist(improvPerc_3, defbins, label='PS', alpha=0.75)

ax.legend(loc='upper left')

plt.savefig(rst_Prefix + caseName + '/improve_distribution_WrongRight.svg')
plt.close(fig)
