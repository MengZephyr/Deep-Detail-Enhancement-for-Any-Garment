import cv2
import numpy as np

rootPref = 'D:/models/NR/Data/ver_4/Data/'
caseName = 'bNorm/'
#rootPref = 'D:/models/MD/mixamo_Skinning/Claire/breakDance/render/'
#caseName = '/'
characRoot1 = rootPref + caseName + '/'
#characRoot2 = rootPref + caseName + 'render_rst30/'
videoName = rootPref + 'bNorm.mp4'
vcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(videoName, vcc, 30.0, (1024, 1024))
FrameID = [i for i in range(400, 601)]
for fID in FrameID:
     img1 = cv2.imread(characRoot1 + str(fID).zfill(7) + '.png')
     # img = np.array(img, dtype='float32')
     # imgc = np.ones_like(img[:, :, 3]) * 255. - img[:, :, 3]
     # img[:, :, 0] = img[:, :, 0] + imgc
     # img[:, :, 1] = img[:, :, 1] + imgc
     # img[:, :, 2] = img[:, :, 2] + imgc
     # cv2.imwrite(rootPref + caseName + 't.png', img[:, :, 0:3])
     # imgv = cv2.imread(rootPref + caseName + 't.png')
     #exit(1)
     #img2 = cv2.imread(characRoot2 + str(fID)+'.png')
     #img = np.concatenate([img1, img2], axis=1)
     videoWriter.write(img1)
#
videoWriter.release()
exit(1)


imageRoot1 = 'D:/models/MD/DetailTask/test/Geo/render/in_30/Chamuse/'
imageRoot2 = 'D:/models/MD/DetailTask/test/Geo/render/rst_30_10/Chamuse/'
#imageRoot3 = 'D:/models/MD/DetailTask/test/Geo/render/gt_10/Chamuse/'
#imageRoot4 = 'D:/models/MD/Test_24/particleDistTest/Render/30/'

# imageRoot4 = 'D:/models/MD/DataModel/DressOri/case_1/WoolMelton_l/t_Ds10_L/'
# # imageRoot5 = 'D:/models/MD/DataModel/DressOri/case_1/WoolMelton_l/t_10_L/'
# # imageRoot6 = 'D:/models/MD/DetailTask/test/rst/case_1/WoolMelton_l/'
# # imageRoot7 = 'D:/models/MD/DataModel/DressOri/case_1/WoolMelton_l/t_30_L/'
# # imageRoot8 = 'D:/models/MD/DetailTask/test/rst_1/case_1/WoolMelton_l/'

imageRoot3 = 'D:/models/MD/FuseCharac/MikeBreakDance/textures/constN.png'
imageRoot33 = 'D:/models/MD/DataModel/DressOri/case_3/tango_1/upCoat_SilkChiffon/t_30_L/'
imageRoot4 = 'D:/models/MD/FuseCharac/MikeBreakDance/t_30_L/'
imageRoot44 = 'D:/models/MD/DetailTask/patchTest/tango_1/upCoat_SilkChiffon/img30_5cin/'

imageRoot5 = 'D:/models/MD/FuseCharac/MikeBreakDance/render/ori/'
imageRoot55 = 'D:/models/MD/DataModel/DressOri/case_3/tango_1/upCoat_SilkChiffon/t_10_L/'
imageRoot6 = 'D:/models/MD/FuseCharac/MikeBreakDance/DE/'
imageRoot66 = 'D:/models/MD/DataModel/DressOri/case_3/tango_2/upCoat_DenimLightwei/t_30_L/'

imageRoot7 = 'D:/models/MD/FuseCharac/MikeBreakDance/img_Denim/'
imageRoot77 = 'D:/models/MD/DetailTask/patchTest/tango_2/upCoat_DenimLightwei/img30_5cin/'
imageRoot8 = 'D:/models/MD/FuseCharac/MikeBreakDance/render/de/'
imageRoot88 = 'D:/models/MD/DataModel/DressOri/case_3/tango_2/upCoat_DenimLightwei/t_10_L/'

imageRoot9 = 'D:/models/MD/DataModel/DressOri/case_1/Chamuse_Complex/t_10_L/'
imageRoot10 = 'D:/models/MD/DataModel/DressOri/case_1/Chamuse_Complex/render/10/'

imageRoot11 = 'D:/models/MD/DataModel/DressOri/case_2/Wool_Melton/t_Ds10_L/'
imageRoot12 = 'D:/models/MD/DataModel/DressOri/case_2/Wool_Melton/t_10_L/'

#imageRoot9 = 'D:/models/MD/DataModel/DressOri/case_1/Chamuse/t_10_L/'
#imageRoot10 = 'D:/models/MD/DetailTask/test/Geo/render/rst_Ds_10/NylonCarvas_l/'



# imageRoot3 = 'D:/models/MD/DataModel/DressOri/case_1/WoolMelton_l/t_30_L/'
# imageRoot4 = 'D:/models/MD/DetailTask/test/rst_1/case_1/WoolMelton_l/'
# imageRoot5 = 'D:/models/MD/DataModel/DressOri/case_1/Chiffon/t_30_L/'
# imageRoot6 = 'D:/models/MD/DetailTask/test/rst_1/case_1/Chiffon/'
# imageRoot7 = 'D:/models/MD/DataModel/DressOri/case_1/NylonCarvas_l/t_30_L/'
# imageRoot8 = 'D:/models/MD/DetailTask/test/rst_1/case_1/NylonCarvas_l/'

Frame0 = 1
Frame1 = 196
FrameID = [i for i in range(Frame0, Frame1+1)]
#FrameID = [601]

videoName_geo = 'D:/models/MD/DetailTask/test/geo.mp4'
vcc = cv2.VideoWriter_fourcc(*'mp4v')
video_geo = cv2.VideoWriter(videoName_geo, vcc, 30, (3000, 800))
#video_geo = cv2.VideoWriter(videoName_geo, 0, 30, (2400, 700))
videoName_texture = 'D:/models/MD/FuseCharac/MikeBreakDance/rst.mp4'
video_texture = cv2.VideoWriter(videoName_texture, vcc, 30, (1263, 842))


def combineImg (BImg, img1, img2):
    h, w = BImg.shape[0], BImg.shape[1]
    h1, w1 = img1.shape[0], img1.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    BImg[h - h2:h, h - w2:w] = img2
    BImg[0:h1, 0:w1, :] = img1
    return BImg


ifText = True
ifGeo = False

img4 = cv2.imread(imageRoot4 + str(1).zfill(7) + '.png')
imgB = np.zeros_like(img4)
imgB = cv2.resize(imgB, (850, 850), cv2.INTER_AREA)
for fID in FrameID:
    if ifGeo:
        img1 = cv2.imread(imageRoot1+str(fID)+'.png')
        img2 = cv2.imread(imageRoot2+str(fID)+'.png')
        img3 = cv2.imread(imageRoot3+str(fID)+'.png')
        img4 = cv2.imread(imageRoot4+str(fID)+'.png')

        img_geo = np.concatenate([img1, img2, img3], axis=1)
        # img_geo_do = np.concatenate([img3, img4], axis=1)
        # img_geo = np.concatenate([img_geo_up, img_geo_do], axis=0)
        #img_geo = np.concatenate([img_geo, img5], axis=1)
        #img_geo = np.concatenate([img1[100:800, 200:1000], img3[100:800, 200:1000]], axis=1)
        #img_geo = np.concatenate([img_geo, img5[100:800, 200:1000]], axis=1)
        # cv2.imwrite('../work_fff/20200107/Chama_10/img.png', img_geo)

        # img2 = cv2.imread(imageRoot2 + str(fID) + '.png')
        # img4 = cv2.imread(imageRoot4 + str(fID) + '.png')
        # img6 = cv2.imread(imageRoot6 + str(fID) + '.png')
        # img_txt = np.concatenate([img2[100:800, 200:1000], img4[100:800, 200:1000]], axis=1)
        # img_txt = np.concatenate([img_txt, img6[100:800, 200:1000]], axis=1)
        #
        # img_geo = np.concatenate([img_geo, img_txt], axis=0)

        # cv2.imwrite('D:/models/MD/DetailTask/test/g.png', img_geo)
        video_geo.write(img_geo)

    if ifText:
        # img3 = cv2.imread(imageRoot3 + str(fID).zfill(7) + '.png')
        #
        # img4 = np.zeros_like(img3)
        # imgO4 = cv2.imread(imageRoot4 + str(fID).zfill(7) + '.png')
        # img4[0:imgO4.shape[0], 0:imgO4.shape[1], :] = imgO4
        #
        # img5 = np.zeros_like(img3)
        # imgO5 = cv2.imread(imageRoot5 + str(fID).zfill(7) + '.png')
        # img5[0:imgO5.shape[0], 0:imgO5.shape[1], :] = imgO5
        #
        # img6 = np.zeros_like(img3)
        # imgO6 = cv2.imread(imageRoot6 + str(fID).zfill(7) + '.png')
        # img6[0:imgO6.shape[0], 0:imgO6.shape[1], :] = imgO6

        # img7 = cv2.imread(imageRoot7 + str(fID).zfill(7) + '_t.png')
        # img8 = cv2.imread(imageRoot8 + str(fID).zfill(7) + '_t.png')
        # img9 = cv2.imread(imageRoot9 + str(fID).zfill(7) + '.png')
        # img3 = cv2.resize(img        3, (img4.shape[0], img4.shape[1]), cv2.INTER_AREA)

        img3 = cv2.imread(imageRoot3)
        #img33 = cv2.imread(imageRoot33 + str(fID).zfill(7) + '.png')
        #img3 = combineImg(np.copy(imgB), img33[0:350, 0:500, :], img3)
        img4 = cv2.imread(imageRoot4 + str(fID).zfill(7) + '.png')
        #img44 = cv2.imread(imageRoot44 + str(fID).zfill(7) + '.png')
        #img4 = combineImg(np.copy(imgB), img44[0:350, 0:500, :], img4)

        img5 = cv2.imread(imageRoot5 + str(fID).zfill(7) + '.png')
        img5 = cv2.resize(img5, (img4.shape[1], img4.shape[0]), cv2.INTER_AREA)
        #img55 = cv2.imread(imageRoot55 + str(fID).zfill(7) + '.png')
        #img5 = combineImg(np.copy(imgB), img55[0:350, 0:500, :], img5)
        img6 = cv2.imread(imageRoot6 + str(fID).zfill(7) + '.png')
        #img66 = cv2.imread(imageRoot66 + str(fID).zfill(7) + '.png')
        #img6 = combineImg(np.copy(imgB), img66[0:350, 0:500, :], img6)

        img7 = cv2.imread(imageRoot7 + str(fID).zfill(7) + '.png')
        #img77 = cv2.imread(imageRoot77 + str(fID).zfill(7) + '.png')
        #img7 = combineImg(np.copy(imgB), img77[0:350, 0:500, :], img7)
        img8 = cv2.imread(imageRoot8 + str(fID).zfill(7) + '.png')
        img8 = cv2.resize(img8, (img7.shape[1], img7.shape[0]), cv2.INTER_AREA)
        #img88 = cv2.imread(imageRoot88 + str(fID).zfill(7) + '.png')
        #img8 = combineImg(np.copy(imgB), img88[0:350, 0:500, :], img8)

        #img9 = cv2.imread(imageRoot9 + str(fID).zfill(7) + '.png')
        #img10 = cv2.imread(imageRoot10 + str(fID) + '.png')

        #img12 = cv2.imread(imageRoot12 + str(fID).zfill(7) + '.png')
        # img9 = cv2.imread(imageRoot9 + str(fID) + '_t.png')
        #img10 = cv2.imread(imageRoot10 + str(fID) + '.png')
        #img11 = cv2.imread(imageRoot11 + str(fID).zfill(7) + '.png')
        #img12 = cv2.imread(imageRoot12 + str(fID).zfill(7) + '.png')

        #imgB = np.zeros_like(img3)

        #img_txt_up = np.concatenate([img3[100:650, 150:850], img4[100:650, 150:850], img5[100:650, 150:850]], axis=1)
        #img_txt_do = np.concatenate([img6[100:650, 150:850], img7[100:650, 150:850], img8[100:650, 150:850]], axis=1)
        img_txt_up = np.concatenate([img5, img3, img4], axis=1)
        img_txt_do = np.concatenate([img8, img6, img7], axis=1)
        #img_txt_do = cv2.resize(img_txt_do, (img_txt_up.shape[1], img_txt_up.shape[1]//4), cv2.INTER_AREA)
        #img_txt_up = np.concatenate([img3, img4], axis=1)
        #img_txt_do = np.concatenate([img5, img6], axis=1)
        # img_txt_up = np.concatenate([img3[100:800, 150:1000], img5[100:800, 150:1000], img7[100:800, 150:1000]], axis=1)
        # img_txt_do = np.concatenate([img4[100:800, 150:1000], img6[100:800, 150:1000], imgB[100:800, 150:1000]], axis=1)
        img_txt = np.concatenate([img_txt_up, img_txt_do], axis=0)

        #cv2.imwrite('D:/models/MD/FuseCharac/MikeBreakDance/t.png', img_txt)
        #exit(1)
        video_texture.write(img_txt)


if ifGeo:
    video_geo.release()
if ifText:
    video_texture.release()
