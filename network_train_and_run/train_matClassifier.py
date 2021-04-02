import torch
from VGG_Model import VGG19, vgg_mean, vgg_std
from torchvision import transforms
from PIL import Image
import glob
import os
from random import shuffle
from models import classifyNet
import torch.optim as optim
import time

USE_CUDA = torch.cuda.is_available()
print("balin-->", USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")

Patch_H = 128
Patch_W = 128

Img_transform = transforms.Compose([transforms.Resize((Patch_H, Patch_W)), transforms.ToTensor(),
                                    transforms.Normalize(vgg_mean, vgg_std)])

NUMClass = 5
BSize = 32
numEpoch = 10


def get_fileSet_list(dir, withF=[]):
    if len(withF) == 0:
        return glob.glob(os.path.join(dir, "*"))
    else:
        fL = []
        for wf in withF:
            fL = fL + glob.glob(os.path.join(dir, wf))
        return fL

fModel = VGG19(device).to(device).eval()


def getBatchData(DirList, batchIDs, RootList, device, refMat):
    iX = []
    oY = []
    for id in batchIDs:
        imgN = DirList[id]
        p = imgN.split('/')
        imgs = []
        for r in RootList:
            name = r + p[-1]
            img = Image.open(name)
            img = Img_transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0).to(device)
        ifeat = fModel(imgs)
        ifeat = ifeat.view(1, -1)
        #print(ifeat.size())
        #exit(1)
        iX.append(ifeat)

        pm = imgN.split('_')
        pmm = pm[-1].split('.')
        mid = refMat[int(pmm[0])]
        #print(mid)
        y = torch.zeros(1, NUMClass)
        y[0, mid] = 1
        oY.append(y)

    iX = torch.cat(iX, dim=0).to(device)
    oY = torch.cat(oY, dim=0).type(torch.FloatTensor).to(device)

    return iX, oY


TrainDir = ['../TrainData/TT_InfoPatch/APT_in/']

'''
    0: Chamuse_Silk, 1: WoolMelton, 2: Knit_Terry,
    3: Chiffon_Silk, 4: DenimLight
'''
Train_RefMat = [0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
wF = ["*_1.png", "*_2.png", "*_3.png", "*_4.png",
      "*_7.png", "*_8.png", "*_9.png", "*_10.png", "*_11.png", "*_12.png"]
Train_imgList = get_fileSet_list(TrainDir[0], withF=wF)
TrainID = [i for i in range(len(Train_imgList))]
shuffle(TrainID)
TrainKK = len(TrainID) // BSize
print(TrainKK)

Test_RefMat = [0, 1, 2]
# TestDir = ['../TrainData/TT_InfoPatch/PE_in_b/1/',
#            '../TrainData/TT_InfoPatch/PE_in/',
#            '../TrainData/TT_InfoPatch/PE_in_a/1/']
TestDir = ['../TrainData/TT_InfoPatch/PE_in/']
Test_imgList = get_fileSet_list(TestDir[0], withF=[])
TestID = [i for i in range(len(Test_imgList))]
shuffle(TestID)
TestKK = len(TestID) // BSize

#test_x, test_y = getBatchData(Test_imgList, TestID[0:8], TestDir, device, test_RefMat)
#c_model = classifyNet(24576, NUMClass).to(device)  # 24576 = 4 x 4 x 512 x 3
c_model = classifyNet(8192, NUMClass).to(device)  # 8192 = 4 x 4 x 512
print(c_model)

lossFunction = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(params=c_model.parameters(), lr=0.01, betas=(0.9, 0.999), amsgrad=True)

IFSumWriter = True
if IFSumWriter:
    from torch.utils.tensorboard import SummaryWriter
    #from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir='./runs_CC/CC_sigF/')

for epo in range(numEpoch):
    shuffle(TrainID)
    for kk in range(TrainKK):
        t = time.time()
        c_model.train()
        optimizer.zero_grad()
        iterN = epo*TrainKK + kk
        train_x, train_y = getBatchData(Train_imgList, TrainID[kk*BSize: (kk+1)*BSize], TrainDir, device, Train_RefMat)
        out_y = c_model(train_x)
        loss = lossFunction(out_y, train_y)
        loss.backward()
        optimizer.step()
        print("Train Iter: {}, loss: {:.4f} time:{:.4f}s".
              format(iterN, loss.item(), time.time() - t))

        if iterN % 100 == 0 or iterN == TrainKK * numEpoch - 1:
            writer.add_scalar('trainLoss', loss, iterN)
            t = time.time()
            c_model.eval()
            with torch.no_grad():
                ttkk = iterN % TestKK
                if ttkk == 0:
                    shuffle(TestID)
                test_x, test_y = getBatchData(Test_imgList, TestID[ttkk*BSize: (ttkk+1)*BSize],
                                              TestDir, device, Test_RefMat)
                yy = c_model(test_x)
                loss = lossFunction(yy, test_y)
                print("Test Iter: {}, loss: {:.4f} time:{:.4f}s".
                      format(iterN, loss.item(), time.time() - t))
                writer.add_scalar('testLoss', loss, iterN)

        if iterN % 500 == 0:
            saveName = '../matCCTest/ckp_sigF/temp_500.ckp'
            torch.save({'itter': epo * TrainKK - 1, 'model': c_model.state_dict(), 'optimizer': optimizer.state_dict()},
                       saveName)

    saveName = '../matCCTest/ckp_sigF/e_' + str(epo+1) + 't_' + str((epo + 1) * TrainKK - 1) + '.ckp'
    torch.save({'model': c_model.state_dict()}, saveName)

