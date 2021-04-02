from torchvision import transforms
from PIL import Image
import torch
from torchvision.utils import save_image

from models import classifyNet
from VGG_Model import VGG19, vgg_mean, vgg_std
import numpy as np
import matplotlib.pyplot as plt

PATCH_SIZE = 128

USE_CUDA = torch.cuda.is_available()
print("Balin-->", USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")

NUMClass = 5

# c_model = classifyNet(24576, NUMClass).to(device).eval()  # 24576 = 4 x 4 x 512 x 3
# print(c_model)
# ckpRoot = '../matCCTest/ckp_sigmoid/'
# ckp = torch.load(ckpRoot+'temp_500.ckp', map_location=lambda storage, loc: storage)
# c_model.load_state_dict(ckp['model'])

c_model = classifyNet(8192, NUMClass).to(device).eval()  # 24576 = 4 x 4 x 512 x 3
print(c_model)
ckpRoot = './ckp/'
ckp = torch.load(ckpRoot+'Classifier_model.ckp', map_location=lambda storage, loc: storage)
c_model.load_state_dict(ckp['model'])


def genPatchCropG(mask, imgH, imgW, patchSize):
    if imgH < patchSize or imgW < patchSize:
        return None, None
    img_X_beg_end = []
    img_Y_beg_end = []

    K_H = imgH // (patchSize // 2) + 1 if imgH % (patchSize // 2) > 0 else imgH // (patchSize // 2)
    K_W = imgW // (patchSize // 2) + 1 if imgW % (patchSize // 2) > 0 else imgW // (patchSize // 2)
    #print(K_H, K_W)

    for kky in range(K_H):
        for kkx in range(K_W):
            x = kkx * (patchSize // 2)
            y = kky * (patchSize // 2)
            ix_b = x if x >= 0 else 0
            iy_b = y if y >= 0 else 0
            ix_e = ix_b+patchSize
            if ix_e > imgW:
                ix_b = imgW - patchSize
                ix_e = imgW
            iy_e = iy_b+patchSize
            if iy_e > imgH:
                iy_b = imgH - patchSize
                iy_e = imgH

            m_area = mask[:, :, iy_b:iy_e, ix_b:ix_e]
            if torch.sum(m_area) <= float(patchSize) * float(patchSize) * 0.25:
                continue

            img_X_beg_end.append([ix_b, ix_e])
            img_Y_beg_end.append([iy_b, iy_e])

    return img_X_beg_end, img_Y_beg_end


fModel = VGG19(device).to(device).eval()  # in: [1, 3, 128, 128] --> out: [1, 512, 4, 4] --> vec: [1, 8192]


def calImgFeatures(FrameRoot, fID, CC, imgTransform, crop_X, crop_Y):
    ImgS = []
    for c in CC:
        imgN = FrameRoot + str(fID+c).zfill(7) + '.png'
        img = Image.open(imgN)
        img = imgTransform(img)
        img = img.unsqueeze(0)
        ImgS.append(img)

    ImgS = torch.cat(ImgS, dim=0)
    numP = len(crop_X)
    pFeatures = []
    for p in range(numP):
        x_b = crop_X[p][0]
        x_e = crop_X[p][1]
        y_b = crop_Y[p][0]
        y_e = crop_Y[p][1]
        pCC = ImgS[:, :, y_b:y_e, x_b:x_e].to(device)
        ifeat = fModel(pCC)
        ifeat = ifeat.view(1, -1)
        pFeatures.append(ifeat)
    pFeatures = torch.cat(pFeatures, dim=0)
    return pFeatures


def calcCCVoting(pFeatures):
    BatchSize = 16
    numPatches = pFeatures.size()[0]
    KK = numPatches // BatchSize + 1 if numPatches % BatchSize > 0 else numPatches // BatchSize
    PCC = []
    for k in range(KK):
        bk = k * BatchSize
        ek = (k + 1) * BatchSize if (k + 1) * BatchSize <= numPatches else numPatches
        inTensor = pFeatures[bk:ek, :]
        propC = c_model(inTensor)
        PCC.append(propC)
    PCC = torch.cat(PCC, dim=0)
    return PCC

'''
    0: Chamuse_Silk, 1: WoolMelton, 2: Knit_Terry,
    3: Chiffon_Silk, 4: DenimLight
'''

if __name__ == '__main__':
    F_Prefix = '../Data/case_1/'
    caseName = 'WoolMelton_l/'
    #frame0 = 150
    #frame1 = 150
    frame0 = 100
    frame1 = 200
    #CC = [-1, 0, 1]
    CC = [0]
    MaskName = F_Prefix + caseName + 'Mask.png'
    FrameRoot = F_Prefix + caseName + 't_Ds10_L/'

    Tensor_mask = Image.open(MaskName)
    imgH = Tensor_mask.height
    imgW = Tensor_mask.width
    Mask_transform = transforms.Compose([transforms.Resize((imgH, imgW)), transforms.ToTensor()])
    Img_transform = transforms.Compose([transforms.Resize((imgH, imgW)), transforms.ToTensor(),
                                        transforms.Normalize(vgg_mean, vgg_std)])

    Tensor_mask = Mask_transform(Tensor_mask)
    Tensor_mask = Tensor_mask[0, :, :].unsqueeze(0).unsqueeze(0).to(device)
    print(imgH, imgW)

    img_X_beg_end, img_Y_beg_end = genPatchCropG(Tensor_mask, imgH, imgW, PATCH_SIZE)
    numPatches = len(img_X_beg_end)
    print(numPatches)

    #fID = 200
    apcc = []
    tpqq = []
    for fID in range(frame0, frame1+1):
        pFeatures = calImgFeatures(FrameRoot=FrameRoot, fID=fID, CC=CC, imgTransform=Img_transform,
                                   crop_X=img_X_beg_end, crop_Y=img_Y_beg_end)

        PCC = calcCCVoting(pFeatures)
        pcc = torch.sum(PCC, dim=0) / torch.sum(PCC)
        print(fID, '-->', pcc)
        apcc.append(pcc.unsqueeze(0).detach().cpu().numpy())
        tpqq.append(PCC.detach().cpu().numpy())

        #tpqq = np.concatenate(tpqq, axis=0)
        # drawimg = torch.zeros(1, 4, imgH, imgW).type(torch.FloatTensor)
        # propimg = torch.zeros(1, 1, imgH, imgW).type(torch.FloatTensor)
        # ccimg = torch.zeros(1, 1, imgH, imgW).type(torch.FloatTensor)
        # bv = [tpqq[i][1] for i in range(tpqq.shape[0])]
        # bvmax = max(bv)
        # bvmin = min(bv)
        # print(bvmin, bvmax)
        # for i in range(tpqq.shape[0]):
        #     x_b = img_X_beg_end[i][0]
        #     x_e = img_X_beg_end[i][1]
        #     y_b = img_Y_beg_end[i][0]
        #     y_e = img_Y_beg_end[i][1]
        #     drawimg[0, 0, y_b:y_e, x_b:x_e] = 245./255.
        #     drawimg[0, 1, y_b:y_e, x_b:x_e] = 145./255.
        #     drawimg[0, 2, y_b:y_e, x_b:x_e] = 11./255.
        #     ccimg[0, 0, y_b:y_e, x_b:x_e] += 1.
        #     drawimg[0, 3, y_b:y_e, x_b:x_e] += (bv[i] - bvmin) / (bvmax - bvmin)
        # drawimg[0, 3, :, :] = drawimg[0, 3, :, :] / ccimg
        # drawimg= drawimg.to(device) * Tensor_mask
        # save_image(drawimg, '../xx_2.png')
        # exit(1)


    print(np.sum(apcc, axis=0) / float(frame1-frame0+1))
    print(apcc[0].shape)
    tacc = np.concatenate(apcc, axis=0)
    print(tacc.shape)

    defbins = np.linspace(np.min(tacc), np.max(tacc), 30, endpoint=True)
    px = [i for i in range(tacc.shape[0])]
    fig, ax = plt.subplots()
    for i in range(tacc.shape[1]):
         n, bins, patches = ax.hist(tacc[:, i], defbins, label=str(i), alpha=0.6)
    ax.legend(loc='best')
    plt.savefig('../matCCTest/dist7_100.svg')
    plt.close(fig)

    # tpqq = np.concatenate(tpqq, axis=0)
    # print(tpqq.shape)
    # defbins = np.linspace(np.min(tpqq), np.max(tpqq), 30, endpoint=True)
    # px = [i for i in range(tpqq.shape[0])]
    # fig, ax = plt.subplots()
    # for i in range(tpqq.shape[1]):
    #      n, bins, patches = ax.hist(tpqq[:, i], defbins, label=str(i), alpha=0.6)
    # ax.legend(loc='best')
    # plt.savefig('../matCCTest/dist6_50.svg')
    # plt.close(fig)




