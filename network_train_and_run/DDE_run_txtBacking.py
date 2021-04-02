import numpy as np
from torchvision import transforms
from PIL import Image
import torch
from torchvision.utils import save_image

from models import Generator_CNN, Generator_CNN_Cat, Generator_CNNCIN
from VGG_Model import vgg_mean, vgg_std
import math

PATCH_SIZE = 128

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print("Balin-->", USE_CUDA)
device = torch.device("cuda:1" if USE_CUDA else "cpu")

IFJJMask = False
numJJ = 8 if IFJJMask is True else 0
#model = Generator_CNN(3 + numJJ, 3).to(device) if IFJJMask is True else Generator_CNN(3 + numJJ, 3).to(device)
model = Generator_CNNCIN(3, 3, 5).to(device)
print(model)

ckpRoot = '' if IFJJMask is True else './ckp/'
#ckpRoot = './noMaskCase1_ckp/'
ckp = torch.load(ckpRoot+'DDE_model.ckp', map_location=lambda storage, loc: storage)
model.load_state_dict(ckp['model'])
betit = ckp['itter']
best = ckp['loss']
model.eval()

# for param_tensors in model.state_dict():
#     print(param_tensors, "\t", model.state_dict()[param_tensors].size())
# print(model.state_dict()['D4_CIN.inns.0.weight'])
# print(model.state_dict()['D4_CIN.inns.1.weight'])
# print(model.state_dict()['D4_CIN.inns.2.weight'])
#
# print(model.state_dict()['D4_CIN.inns.0.bias'])
# print(model.state_dict()['D4_CIN.inns.1.bias'])
# print(model.state_dict()['D4_CIN.inns.2.bias'])
# exit(1)

print("Test: begLoss: " + str(best) + " >> begIter: " + str(betit))


def genPatchCropG(mask, imgH, imgW, patchSize):
    if imgH < patchSize or imgW < patchSize:
        return None, None
    img_X_beg_end = []
    img_Y_beg_end = []

    K_H = imgH // (patchSize // 2) + 1 if imgH % (patchSize // 2) > 0 else imgH // (patchSize // 2)
    K_W = imgW // (patchSize // 2) + 1 if imgW % (patchSize // 2) > 0 else imgW // (patchSize // 2)
    print(K_H, K_W)

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
            if torch.sum(m_area) <= 0:
                continue

            img_X_beg_end.append([ix_b, ix_e])
            img_Y_beg_end.append([iy_b, iy_e])

    return img_X_beg_end, img_Y_beg_end


def cropImage(ImgTensor, i_X_beg_end, i_Y_beg_end):
    numP = len(i_X_beg_end)
    _, nC, nH, nW = ImgTensor.size()
    # print(ImgTensor.size())
    patchTArray = []
    for p in range(numP):
        x_b = i_X_beg_end[p][0]
        x_e = i_X_beg_end[p][1]
        y_b = i_Y_beg_end[p][0]
        y_e = i_Y_beg_end[p][1]
        patchTArray.append(ImgTensor[:, :, y_b:y_e, x_b:x_e])

    patchTArray = torch.cat(patchTArray, dim=0)

    return patchTArray


def blendPatches(DrawTensor, Tensor_mask, rstPatches, i_X_beg_end, i_Y_beg_end, Wei):
    weiPatch = rstPatches * Wei
    ccImg = torch.zeros_like(Tensor_mask)
    numP = rstPatches.shape[0]
    for p in range(numP):
        x_b = i_X_beg_end[p][0]
        x_e = i_X_beg_end[p][1]
        y_b = i_Y_beg_end[p][0]
        y_e = i_Y_beg_end[p][1]
        DrawTensor[:, :, y_b:y_e, x_b:x_e] = DrawTensor[:, :, y_b:y_e, x_b:x_e] + weiPatch[p, :, :, :]
        ccImg[0, :, y_b:y_e, x_b:x_e] = ccImg[0, :, y_b:y_e, x_b:x_e] + Wei[p]
    validC = torch.where(ccImg > 0.)
    vvImg = torch.ones_like(Tensor_mask)
    vvImg[validC] = ccImg[validC]
    DrawTensor = DrawTensor / vvImg
    DrawTensor = DrawTensor*torch.tensor(vgg_std).view(-1, 1, 1).to(device) \
                 + torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
    DrawTensor = DrawTensor * Tensor_mask

    return DrawTensor


if __name__ == '__main__':
    '''
    Generate sequence patch crop index
    '''
    F_Prefix = '../Data/case_3/'
    caseName = 'Chamuse_tango/skirt/'
    MaskName = F_Prefix + caseName + 'Mask.png'
    saveRoot = '../PatchTest/' + caseName + '/img_2/'
    FrameRoot = F_Prefix + caseName + 't_Ds10_L/'
    '''
        0: Chamuse_Silk, 1: WoolMelton, 2: Knit_Terry,
        3: Chiffon_Silk, 4: DenimLight
    '''
    matID = 0
    Frame0 = 2
    Frame1 = 2
    BatchSize = 64

    JJInfo_InfoDir = \
        [F_Prefix + caseName + 'uv/JJWei_Ds10/' + str(i) + '_' for i in range(numJJ)] if IFJJMask is True else []

    Tensor_mask = Image.open(MaskName)
    imgH = Tensor_mask.height
    imgW = Tensor_mask.width
    Mask_transform = transforms.Compose([transforms.Resize((imgH, imgW)), transforms.ToTensor()])
    Tensor_mask = Mask_transform(Tensor_mask)
    Tensor_mask = Tensor_mask[0, :, :].unsqueeze(0).unsqueeze(0).to(device)
    print(imgH, imgW)

    img_X_beg_end, img_Y_beg_end = genPatchCropG(Tensor_mask, imgH, imgW, PATCH_SIZE)
    numPatches = len(img_X_beg_end)
    print(numPatches)

    # maskPatches = cropImage(Tensor_mask, img_X_beg_end, img_Y_beg_end)
    # print(maskPatches.size())
    patchWei = torch.zeros([1, 1, PATCH_SIZE, PATCH_SIZE], dtype=torch.float, device=device)
    for py in range(PATCH_SIZE):
        for px in range(PATCH_SIZE):
            value = (py - float(PATCH_SIZE//2))*(py - float(PATCH_SIZE//2)) +\
                    (px - float(PATCH_SIZE//2))*(px - float(PATCH_SIZE//2))
            value = math.exp(-value / (float(PATCH_SIZE//2) * float(PATCH_SIZE//2)))
            patchWei[0, 0, py, px] = value
    #save_image(patchWei, filename='../PatchTest/wei.png')
    patchWei = torch.cat([patchWei for _ in range(numPatches)], dim=0)

    Img_transform = transforms.Compose([transforms.Resize((imgH, imgW)), transforms.ToTensor(),
                                        transforms.Normalize(vgg_mean, vgg_std)])

    KK = numPatches // BatchSize + 1 if numPatches % BatchSize > 0 else numPatches // BatchSize
    import time
    tCount = 0
    for f in range(Frame0, Frame1+1):
        imgName = FrameRoot + str(f).zfill(7) + '.png'
        in_img = Image.open(imgName)
        in_img = Img_transform(in_img)
        in_img = in_img.unsqueeze(0).to(device)

        if IFJJMask is True:
            Maps = []
            for j in range(numJJ):
                mapN = JJInfo_InfoDir[j] + str(f).zfill(7) + '.png'
                imap = Image.open(mapN)
                imap = Mask_transform(imap)
                imap = imap.unsqueeze(0).to(device)
                Maps.append(imap)
            if len(Maps) > 0:
                Maps = torch.cat(Maps, dim=1)
                in_img = torch.cat([in_img, Maps], dim=1)

        ##
        # show_in = in_img * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        # save_image(show_in, filename='../patchTest/in_img.png')
        ##

        imgPatches = cropImage(in_img, img_X_beg_end, img_Y_beg_end)

        ##
        # showPatch = imgPatches[1, :, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        # save_image(showPatch, filename='../patchTest/pp.png')
        ##

        rstPatches = torch.zeros_like(imgPatches[:, 0:3, :, :])
        t = time.time()
        for k in range(KK):
            bk = k * BatchSize
            ek = (k+1) * BatchSize if (k+1) * BatchSize <= numPatches else numPatches
            inTensor = imgPatches[bk:ek, :, :, :]
            matIDs = [matID for _ in range(ek-bk)]
            rstPatches[bk:ek, :, :, :] = model(inTensor, matIDs)
            #rstPatches[bk:ek, :, :, :] = model(inTensor)

        tCount = tCount + (time.time() - t)
        rstImg = torch.zeros_like(in_img[:, 0:3, :, :])
        rstImg = blendPatches(DrawTensor=rstImg, Tensor_mask=Tensor_mask, rstPatches=rstPatches,
                              i_X_beg_end=img_X_beg_end, i_Y_beg_end=img_Y_beg_end, Wei=patchWei)

        save_image(rstImg, filename=saveRoot + str(f).zfill(7) + '.png')
        print('Frame_', f, '--> DONE')

    print('Time--->', tCount)
