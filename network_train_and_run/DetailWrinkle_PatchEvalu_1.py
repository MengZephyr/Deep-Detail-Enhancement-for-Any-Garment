from torchvision import transforms
from PIL import Image
import torch
from random import shuffle
from torchvision.utils import save_image
from VGG_Model import vgg_mean, vgg_std, PatchStyleFeatures
import math
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt

PATCH_SIZE = 128

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print("Balin-->", USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")


Img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(vgg_mean, vgg_std)])
Mask_transform = transforms.Compose([transforms.ToTensor()])


def BBox_PatchCenter(center, patchsize, imgH, imgW):
    minX = center[0]-patchsize//2
    minY = center[1]-patchsize//2
    maxX = center[0]+patchsize//2
    maxY = center[1]+patchsize//2
    if minX < 0 or minY < 0:
        return -1, -1, -1, -1
    if maxX > imgW-1 or maxY > imgH-1:
        return -1, -1, -1, -1
    return minX, minY, maxX, maxY


def calc_ValidArea(mask, minX, minY, maxX, maxY):
    p_mask = mask[minY:maxY, minX:maxX]
    A = torch.sum(p_mask)
    return A


def check_pickcenter(pC, PSize, imgH, imgW, mask):
    minX, minY, maxX, maxY = BBox_PatchCenter(pC, PSize, imgH, imgW)
    if minX >= 0 and minY >= 0:
        area = calc_ValidArea(mask, minX, minY, maxX, maxY)
        if area > PSize * PSize * 0.6:
            return True
    return False


def genValidCenterArray(PSize, imgH, imgW, mask):
    validCenter = []
    for y in range(imgH):
        for x in range(imgW):
            pCenter = [x, y]
            if check_pickcenter(pCenter, PSize, imgH, imgW, mask):
                validCenter.append(pCenter)
    return validCenter


def getRandomPatch(img_Dir, mask, CenterArray, Frame0, Frame1, numPPF, imgH, imgW):
    patches = []
    maskPatches = []
    KK = len(CenterArray) // numPPF
    cc = 0
    for f in range(Frame0, Frame1+1):
        imgN = img_Dir + str(f).zfill(7) + '.png'
        in_img = Image.open(imgN)
        in_img = Img_transform(in_img)
        in_img = in_img.unsqueeze(0).to(device)

        if cc % KK == 0:
            shuffle(CenterArray)
            cc = 0
        for p in range(numPPF):
            if numPPF == 1:
                patches.append(in_img)
                maskPatches.append(mask)
            else:
                pcenter = CenterArray[cc*numPPF + p]
                minX, minY, maxX, maxY = BBox_PatchCenter(center=pcenter, patchsize=PATCH_SIZE, imgH=imgH, imgW=imgW)
            # print(minX, minY, maxX, maxY)
            # save_image(in_img*torch.tensor(vgg_std).view(-1, 1, 1).to(device)
            #            + torch.tensor(vgg_mean).view(-1, 1, 1).to(device), '../PatchTest/t_i.png')
                patches.append(in_img[:, :, minY:maxY, minX:maxX])
                maskPatches.append(mask[:, :, minY:maxY, minX: maxX])
            # save_image(in_img[:, :, minY:maxY, minX:maxX]*torch.tensor(vgg_std).view(-1, 1, 1).to(device)
            #            + torch.tensor(vgg_mean).view(-1, 1, 1).to(device), '../PatchTest/t_p_'+str(p) + '.png')
            # save_image(mask[:, :, minY:maxY, minX: maxX], '../PatchTest/t_m.png')
        cc = cc+1
    return torch.cat(patches, dim=0), torch.cat(maskPatches, dim=0)


def calGramFeatures(fModel, PImgs, PMasks, layID):
    numP = PImgs.size()[0]
    BatSize = 16
    KK = numP // BatSize + 1 if numP % BatSize > 0 else numP // BatSize
    GFF = []
    for k in range(KK):
        bk = k*BatSize
        ek = (k+1)*BatSize if (k+1)*BatSize < numP else numP
        inPatch = PImgs[bk:ek, :, :, :]
        inMasks = PMasks[bk:ek, :, :, :]
        GramFeatures = fModel(inPatch, inMasks, layID)
        GFF.append(GramFeatures)
    GFF = torch.cat(GFF, dim=0)
    return GFF


embedding = SpectralEmbedding(n_components=2)


def saveEmbPnts(features, fdir):
    numF = features.shape[0]
    print(numF)
    with open(fdir, 'w') as f:
        f.write(str(numF) + '\n')
        for i in range(numF):
            f.write(str(features[i, 0]) + ' ' + str(features[i, 1]) + '\n')
        f.close()


def gramFeatureEmbedding(cFeat, Gtlabel, numD, l, saveName):
    cmap = 'viridis'
    dot_size = 20

    F_transformed = embedding.fit_transform(cFeat)
    print(F_transformed.shape)

    fig, ax = plt.subplots()
    i = -1
    # labels = ['coarse', 'style', 'result']
    # for color in ['maroon', 'green', 'goldenrod']:
    labels = ['PD10ds', 'PD30', 'GroundTruth']
    for color in ['blue', 'maroon', 'green']:
        id0 = 0 if i < 0 else numD[i]
        i = i+1
        id1 = 0 if i < 0 else numD[i]
        ax.scatter(F_transformed[id0:id1, 0], F_transformed[id0:id1, 1], c=color, label=labels[i],
                   s=dot_size, alpha=0.5, edgecolors='none')
        #saveEmbPnts(F_transformed[id0:id1, :], saveName + labels[i] + '_' + str(l+1) + '.txt')
    ax.legend(loc='upper right')

    plt.savefig(saveName + str(l) + "_PD10ds_PD30_Gt.svg")
    plt.close(fig)

    fig, ax = plt.subplots()
    i = 1
    labels = ['GroundTruth', 'Result[PD30]']
    l = -1
    for color in ['green', 'orange']:
        l = l+1
        id0 = 0 if i < 0 else numD[i]
        i = i + 1
        id1 = 0 if i < 0 else numD[i]
        print(id0, id1)
        ax.scatter(F_transformed[id0:id1, 0], F_transformed[id0:id1, 1], c=color, label=labels[l],
                   s=dot_size, alpha=0.5, edgecolors='none')
        # saveEmbPnts(F_transformed[id0:id1, :], saveName + labels[i] + '_' + str(l+1) + '.txt')
    ax.legend(loc='upper right')

    plt.savefig(saveName + str(l) + "_RstPD30_gt.svg")
    plt.close(fig)

    print('layer ', l, ': embedding DONE.')


if __name__ == '__main__':
    Frame0 = 1
    Frame1 = 200
    numRP = 5

    dimRange = [4096, 16384, 65536, 262144, 262144]
    numLayer = 3
    layerID = [i for i in range(numLayer)]

    prefRoot = '../tableData/'
    caseNameList = ['case_1/PleatedNoHem/Chamuse/']

    for caseName in caseNameList:
        print('----------- CaseName_', caseName, '------------------')
        I_inDir = prefRoot + caseName + '/t_30_L/'
        I_StyleDir = prefRoot + caseName + '/t_10_L/'
        I_rstDir = prefRoot + caseName + '/img_30/'
        I_tinDir = prefRoot + caseName + '/t_Ds10_L/'
        I_maskDir = prefRoot + caseName + '/Mask.png'
        saveName = prefRoot + caseName + '/em'

        mask = Image.open(I_maskDir)
        mask = Mask_transform(mask)[0, :, :].to(device)
        imgH = mask.size()[1]
        imgW = mask.size()[0]

        validCenter = genValidCenterArray(PSize=PATCH_SIZE, imgH=imgH, imgW=imgW, mask=mask)
        print(len(validCenter))

        mask = mask.unsqueeze(0).unsqueeze(0)
        shuffle(validCenter)

        shuffle(validCenter)
        tpd10ds_Patches, tpd10ds_Masks = getRandomPatch(I_tinDir, mask, validCenter, Frame0, Frame1, numRP, imgH, imgW)
        print('trainInPatch: ', tpd10ds_Patches.size(), tpd10ds_Masks.size())

        shuffle(validCenter)
        Coarse_Patches, Coarse_Masks = getRandomPatch(I_inDir, mask, validCenter, Frame0, Frame1, numRP, imgH, imgW)
        print('CoarsePatch: ', Coarse_Patches.size(), Coarse_Masks.size())

        shuffle(validCenter)
        Style_Patches, Style_Masks = getRandomPatch(I_StyleDir, mask, validCenter, Frame0, Frame1, numRP, imgH, imgW)
        print('StylePatch: ', Style_Patches.size(), Style_Masks.size())

        shuffle(validCenter)
        Result_Patches, Result_Masks = getRandomPatch(I_rstDir, mask, validCenter, Frame0, Frame1, numRP, imgH, imgW)
        print('ResultPatch: ', Result_Patches.size(), Result_Masks.size())

        FeatureModel = PatchStyleFeatures(device=device).to(device)

        for layer in [1]:

            print('----------- layer_', layer, '------------------')
            Pd10dsFeatures = calGramFeatures(fModel=FeatureModel, PImgs=tpd10ds_Patches, PMasks=tpd10ds_Masks,
                                             layID=layer)
            print('Pd10dsFeatures: ', Pd10dsFeatures.size())
            label_Pd10ds = [0 for _ in range(Pd10dsFeatures.size()[0])]

            CoarseFeatures = calGramFeatures(fModel=FeatureModel, PImgs=Coarse_Patches, PMasks=Coarse_Masks, layID=layer)
            print('CoarseFeatures: ', CoarseFeatures.size())
            label_Coarse = [0 for _ in range(CoarseFeatures.size()[0])]

            StyleFeatures = calGramFeatures(fModel=FeatureModel, PImgs=Style_Patches, PMasks=Style_Masks, layID=layer)
            print('StyleFeatures: ', StyleFeatures.size())
            label_Style = [1 for _ in range(StyleFeatures.size()[0])]

            ResultFeatures = calGramFeatures(fModel=FeatureModel, PImgs=Result_Patches, PMasks=Result_Masks, layID=layer)
            print('ResultFeatures: ', ResultFeatures.size())
            label_Rst = [2 for _ in range(ResultFeatures.size()[0])]

            labels = label_Pd10ds + label_Coarse + label_Style + label_Rst
            Features = torch.cat([Pd10dsFeatures, CoarseFeatures, StyleFeatures, ResultFeatures], dim=0)
            print('Features: ', Features.size())
            numD = [len(label_Pd10ds)] + [len(label_Pd10ds) + len(label_Coarse)] \
                   + [len(label_Pd10ds) + len(label_Coarse) + len(label_Style)] \
                   + [len(label_Pd10ds) + len(label_Coarse) + len(label_Style) + len(label_Rst)]
            print('labels: ', len(labels))

            gramFeatureEmbedding(cFeat=Features.cpu().numpy(), Gtlabel=labels, numD=numD, l=layer, saveName=saveName)



















