import os
import cv2
import numpy as np
if __name__ == '__main__':
    FlOO = False

    caseName = 'UpDress/Chamuse/'
    prefRoot = 'D:/models/MD/DataModel/DressOri/case_1/'
    #prefRoot = 'D:/models/MD/mixamo_Skinning/Claire/breakDance/'
    #caseName = '/'

    maskName = prefRoot + caseName + 'Mask.png'
    mask = cv2.imread(maskName, cv2.IMREAD_COLOR)
    mask = np.ones_like(mask) * 255 - mask

    frame0 = 156
    frame1 = 156
    fID = [f for f in range(frame0, frame1+1)]
    saveRPref = 'D:/models/MD/DetailTask/paperResult/'
    #saveRPref = prefRoot
    #saveRoot = saveRPref
    saveRoot = saveRPref + caseName + 'img_show/input_result/w'
    if not os.path.exists(saveRoot):
        os.makedirs(saveRoot)

    cv2.imwrite(saveRoot + 't.png', mask)

    ll = '_LSkirt'
    imgFileR = prefRoot + caseName + 't_Ds10_L/' if FlOO else saveRPref + caseName + 'img_w/'
    for f in fID:
        imgName = imgFileR + str(f).zfill(7) + '.png'
        img = cv2.imread(imgName, cv2.IMREAD_COLOR)
        img = img + mask
        if FlOO:
            cv2.imwrite(saveRoot+str(f).zfill(7)+'_o.png', img)
        else:
            cv2.imwrite(saveRoot + str(f).zfill(7) + '.png', img)
