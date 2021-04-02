import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

lossF = nn.L1Loss()
transform = transforms.Compose([transforms.Resize((577, 577)), transforms.ToTensor()])
prefR = 'D:/models/MD/DetailTask/paperResult/UpDress/dist/'
GtImg = Image.open(prefR+'Gt_0000156.png')
GtImg = transform(GtImg)
DWImg = Image.open(prefR+'DW_0000156.png')
DWImg = transform(DWImg)
OurImg = Image.open(prefR+'Our_0000156.png')
OurImg = transform(OurImg)
InImg = Image.open(prefR+'In_0000156.png')
InImg = transform(InImg)

Dist_DW = lossF(DWImg, GtImg)
Dist_Our = lossF(OurImg, GtImg)
Dist_In = lossF(InImg, GtImg)
impr_DW = (1.-Dist_DW.item()/Dist_In.item()) * 100.
impr_Our = (1.-Dist_Our.item()/Dist_In.item()) * 100.
print("DW: {:.6f} Improve {:.6f}, Our: {:.6f} Improve {:.6f}".
      format(Dist_DW.item(), impr_DW, Dist_Our.item(), impr_Our))


prefR2 = 'D:/models/MD/DetailTask/paperResult/UpDress/Chamuse/'
GTRoot = 'D:/models/MD/DataModel/DressOri/case_1/UpDress/Chamuse/t_10_L/'
DD_DW = 0.
DD_Our = 0.
for i in range(1, 201):
    DWImg = Image.open(prefR2+'img_gan/'+str(i).zfill(7) + '.png')
    DWImg = transform(DWImg)
    OurImg = Image.open(prefR2+'img_LSkirt/'+str(i).zfill(7) + '.png')
    OurImg = transform(OurImg)
    GtImg = Image.open(GTRoot + str(i).zfill(7) + '.png')
    GtImg = transform(GtImg)
    Dist_DW = lossF(DWImg, GtImg)
    Dist_Our = lossF(OurImg, GtImg)
    DD_DW += Dist_DW
    DD_Our += Dist_Our

print("DW: {:.6f} mean {:.6f}, Our: {:.6f} mean {:.6f}".
      format(DD_DW.item(), DD_DW.item() / 200., DD_Our.item(), DD_Our.item() / 200.))
