import cv2
import numpy as np
from random import shuffle
import os

maskTest = np.zeros((512, 512), dtype=int)


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
    A = np.sum(p_mask) / 255.
    return A


def check_pickcenter(pC, PSize, imgH, imgW, mask):
    minX, minY, maxX, maxY = BBox_PatchCenter(pC, PSize, imgH, imgW)
    if minX >= 0 and minY >= 0:
        area = calc_ValidArea(mask, minX, minY, maxX, maxY)
        if area > PSize * PSize * 0.09:
            return True
    return False


def genValidCenterArray(PSize, imgH, imgW, mask):
    validCenter = []
    for y in range(imgH):
        for x in range(imgW):
            pCenter = [x, y]
            if check_pickcenter(pCenter, PSize, imgH, imgW, mask):
                validCenter.append(pCenter)
    print(len(validCenter))
    return validCenter


RANDOMPATTERN = [0, 1, 2, 3, 4, 5]


def ImgRandom(img, pID):
    if pID == 0:
        return img
    if pID == 1:
        return np.flip(img, 0)
    if pID == 2:
        return np.flip(img, 1)
    if pID == 3:
        return np.flip(np.flip(img, 0), 1)
    if pID == 4:
        return np.rot90(img)
    if pID == 5:
        return np.rot90(img, 3)


def randomImgPatchCrop(imgPrefix, dsimgPrefix, savePrefix, mask, imgH, imgW, PSize, numCrop, vCenters, fID, pCont,
                       caseID, extraInfoN, minfID, maxfID, wSize=3):
    if wSize < 1 or fID > maxfID - wSize // 2:
        return 0
    img = cv2.imread(imgPrefix + str(fID).zfill(7) + '.png', cv2.IMREAD_COLOR)
    dsimg = cv2.imread(dsimgPrefix + str(fID).zfill(7) + '.png', cv2.IMREAD_COLOR)
    infoMap = []
    for info in extraInfoN:
        iMap = cv2.imread(info, cv2.IMREAD_GRAYSCALE)
        infoMap.append(iMap)

    halphW = wSize // 2
    beforeImg = []
    afterImg = []
    ## b [1, 2] f a [1, 2]
    for h in range(halphW):
        b_fId = max(fID-halphW+h, minfID)
        a_fId = min(fID+h+1, maxfID)
        print(b_fId, a_fId)
        beforeImg.append(cv2.imread(dsimgPrefix + str(b_fId).zfill(7) + '.png', cv2.IMREAD_COLOR))
        afterImg.append(cv2.imread(dsimgPrefix + str(a_fId).zfill(7) + '.png', cv2.IMREAD_COLOR))

    vCId = 0
    numV = len(vCenters)
    shuffle(vCenters)
    cp = pCont

    def getValidCenter(CId, pmask):
        pC = vCenters[CId]
        unh = 1000
        uu = 0
        while pmask[pC[1], pC[0]] > 0:
            uu = uu + 1
            if uu > unh:
                return [-1, -1], -1
            CId = CId + 1
            if CId >= numV:
                shuffle(vCenters)
                CId = 0
            pC = vCenters[CId]
        return pC, CId

    pickmask = np.zeros((imgH, imgW), dtype=int)
    drawImg = img.copy()
    pcount = 0

    for p in range(numCrop):
        if vCId >= numV:
            shuffle(vCenters)
            vCId = 0
        pCenter, vCId = getValidCenter(vCId, pickmask)
        if vCId < 0:
            break
        vCId = vCId + 1
        minX, minY, maxX, maxY = BBox_PatchCenter(pCenter, PSize, imgH, imgW)

        shuffle(RANDOMPATTERN)
        patternID = RANDOMPATTERN[0]
        #patternID = 0

        pImg = img[minY:maxY, minX:maxX, :]
        pDs = dsimg[minY:maxY, minX:maxX, :]
        pMask = mask[minY:maxY, minX:maxX]

        if not os.path.exists(savePref + '_gt/'):
            os.mkdir(savePref+'_gt/')
        cv2.imwrite(savePref+'_gt/' + str(cp) + '_' + str(fID) + '_' + str(caseID) + '.png',
                    ImgRandom(pImg, patternID))

        if not os.path.exists(savePref + '_in/'):
            os.mkdir(savePref + '_in/')
        cv2.imwrite(savePref + '_in/' + str(cp) + '_' + str(fID) + '_' + str(caseID) + '.png',
                    ImgRandom(pDs, patternID))

        if not os.path.exists(savePref + '_Mask/'):
            os.mkdir(savePref + '_Mask/')
        cv2.imwrite(savePref + '_Mask/' + str(cp) + '_' + str(fID) + '_' + str(caseID) + '.png',
                    ImgRandom(pMask, patternID))

        if halphW > 0:
            if not os.path.exists(savePref + '_in_a/'):
                os.mkdir(savePref+'_in_a/')
            if not os.path.exists(savePref + '_in_b/'):
                os.mkdir(savePref+'_in_b/')
        for h in range(halphW):
            b_pimg = beforeImg[h][minY:maxY, minX:maxX, :]
            a_pimg = afterImg[h][minY:maxY, minX:maxX, :]

            saveN_b = savePref+'_in_b/'+str(h+1) + '/'
            if not os.path.exists(saveN_b):
                os.mkdir(saveN_b)
            cv2.imwrite(saveN_b + str(cp) + '_' + str(fID) + '_' + str(caseID) + '.png', ImgRandom(b_pimg, patternID))

            saveN_a = savePref+'_in_a/' + str(h+1) + '/'
            if not os.path.exists(saveN_a):
                os.mkdir(saveN_a)
            cv2.imwrite(saveN_a + str(cp) + '_' + str(fID) + '_' + str(caseID) + '.png', ImgRandom(a_pimg, patternID))

        for i in range(len(infoMap)):
            pIMap = infoMap[i][minY:maxY, minX:maxX]
            saveN = '../Data/case_1/Chamuse/InfoPatch/T_JJ/'+str(i) + '/'
            if not os.path.exists(saveN):
                os.mkdir(saveN)
            cv2.imwrite(saveN + str(cp) + '_' + str(fID) + '_' + str(caseID) + '.png', ImgRandom(pIMap, patternID))

        pickmask[minY:maxY, minX:maxX] = 1
        #maskTest[minY:maxY, minX:maxX] = 1
        #drawImg[pCenter[1], pCenter[0], :] = [0, 0, 255]
        cv2.rectangle(drawImg, (minX, minY), (maxX, maxY), (255, 255, 255), 1)
        pcount = pcount + 1
        cp = cp + 1
    cv2.imwrite('../TrainData/MatCC/p'+str(fID)+'.png', drawImg)
    exit(1)
    #cv2.imwrite('../patchTest/ps_'+str(fID)+'.png', pickmask*255)
    print(fID, '-->', pcount)
    return pcount


if __name__ == '__main__':
    PatchSize = 128
    numPPImg = 30
    NumJJ = 8

    FPref = '../TrainData/'
    savePref = FPref + 'MatCC/e_'

    # caseList = ['s_1/', 's_2/', 's_3/', 's_4/', 's_5/', 's_6/', 's_7/', 's_8/', 's_9/', 's_10/',
    #             's_11/', 's_12/', 's_13/', 's_14/', 's_15/']
    # BegF = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # EndF = [901, 850, 988, 988, 988, 901, 901, 850, 850, 850, 850, 988, 988, 901, 901]

    caseList = ['e_0/', 'e_1/', 'e_2/']
    BegF = [100, 161, 221]
    EndF = [160, 220, 280]

    pCc = 0

    for c in range(0, len(caseList)):
        caseName = caseList[c]
        print(caseName, c)
        imgPref = FPref + caseName + 't_10_L/'
        imgDsPref = FPref + caseName + 't_Ds10_L/'
        uvPref = FPref + caseName

        Frame0 = BegF[c]
        Frame1 = EndF[c]

        maskF = uvPref + 'Mask.png'
        mask = cv2.imread(maskF, cv2.IMREAD_GRAYSCALE)
        img_H = mask.shape[0]
        img_W = mask.shape[1]
        print(img_H, img_W)
        validCenters = genValidCenterArray(PSize=PatchSize, imgH=img_H, imgW=img_W, mask=mask)
        print(len(validCenters))

        for f in range(Frame0, Frame1):
            #imgF = imgPref + str(f).zfill(7) + '.png'
            #dsF = imgDsPref + str(f).zfill(7) + '.png'
            #extraInfoN = [uvPref + 'JJWei/' + str(i) + '_' + str(f).zfill(7) + '.png' for i in range(NumJJ)]

            shuffle(validCenters)
            numPP = randomImgPatchCrop(imgPrefix=imgPref, dsimgPrefix=imgDsPref, savePrefix=savePref,
                                       mask=mask, imgH=img_H, imgW=img_W, PSize=PatchSize,
                                       numCrop=numPPImg, vCenters=validCenters, fID=f, pCont=pCc, caseID=c,
                                       extraInfoN=[], minfID=Frame0, maxfID=Frame1, wSize=3)
            pCc = pCc + numPP

    #cv2.imwrite('../patchTest/maskTT.png', maskTest*255)



