import torch.nn as nn
import torch


class classifyNet(nn.Module):
    def __init__(self, inDim, outDim, dropout=0.5):
        super(classifyNet, self).__init__()
        self.DO0 = nn.Dropout(dropout)
        self.FC0 = nn.Linear(inDim, 8192)
        self.BN0 = nn.BatchNorm1d(8192)
        self.Relu0 = nn.ReLU(True)
        self.DO1 = nn.Dropout(dropout)
        self.FC1 = nn.Linear(8192, 4096)
        self.BN1 = nn.BatchNorm1d(4096)
        self.Relu1 = nn.ReLU(True)
        self.DO2 = nn.Dropout(dropout)
        self.FC2 = nn.Linear(4096, 1000)
        self.BN2 = nn.BatchNorm1d(1000)
        self.Relu2 = nn.ReLU(True)
        self.FC3 = nn.Linear(1000, outDim)
        self.Sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        x = self.DO0(X)
        x = self.Relu0(self.BN0(self.FC0(x)))
        x = self.DO1(x)
        x = self.Relu1(self.BN1(self.FC1(x)))
        x = self.DO2(x)
        x = self.Relu2(self.BN2(self.FC2(x)))
        x = self.FC3(x)
        x = self.Sigmoid(x)
        #x = self.softmax(x)
        return x


class batch_InstanceNorm2d(nn.Module):
    def __init__(self, style_num, in_channels):
        super(batch_InstanceNorm2d, self).__init__()
        self.inns = torch.nn.ModuleList([torch.nn.InstanceNorm2d(in_channels, affine=True) for i in range(style_num)])

    def forward(self, x, style_id):
        out = torch.stack([self.inns[style_id[i]](x[i].unsqueeze(0)).squeeze(0) for i in range(len(style_id))])
        return out


class Generator_CNNCIN(nn.Module):
    def __init__(self, inDim, outDim, styleNum, ifADD=True):
        super(Generator_CNNCIN, self).__init__()
        self.ifADD = ifADD
        self.E1_Conv2d = nn.Conv2d(inDim, 64, kernel_size=3, stride=1, padding=1)
        self.E1_ReLu = nn.ReLU(True)
        self.E1_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 64, H//2, W//2], Default H = 512, W = 512

        self.E2_Conv2d = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.E2_ReLu = nn.ReLU(True)
        self.E2_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 128, H//4, W//4]

        self.E3_Conv2d = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.E3_ReLu = nn.ReLU(True)
        self.E3_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 256, H//8, W//8]

        self.E4_Conv2d = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.E4_ReLu = nn.ReLU(True)
        self.E4_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 512, H//16, W//16]

        self.D1_Deconv2d = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # [B, 512, H//8, W//8]
        self.D1_CIN = batch_InstanceNorm2d(styleNum, 512)
        self.D1_ReLu = nn.ReLU(True)

        self.D2_Deconv2d = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) if ifADD is True else \
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=4, stride=2, padding=1)  # [B, 256, H//4, W//4]
        self.D2_CIN = batch_InstanceNorm2d(styleNum, 256)
        self.D2_ReLu = nn.ReLU(True)

        self.D3_Deconv2d = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) if ifADD is True else \
            nn.ConvTranspose2d(256+256, 128, kernel_size=4, stride=2, padding=1)  # [B, 128, H//2, W//2]
        self.D3_CIN = batch_InstanceNorm2d(styleNum, 128)
        self.D3_ReLu = nn.ReLU(True)

        self.D4_Deconv2d = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) if ifADD is True else \
            nn.ConvTranspose2d(128+128, 64, kernel_size=4, stride=2, padding=1)  # [B, 64, H, W]
        self.D4_CIN = batch_InstanceNorm2d(styleNum, 64)
        self.D4_ReLu = nn.ReLU(True)

        self.out_Conv2d = nn.Conv2d(64, outDim, kernel_size=3, stride=1, padding=1) if ifADD is True else \
            nn.ConvTranspose2d(64+64, outDim, kernel_size=3, stride=1, padding=1)

    def forward(self, X, styleId):
        e1 = self.E1_Conv2d(X)
        e1 = self.E1_ReLu(e1)  # [B, 64, H, W]
        e2 = self.E1_Pool(e1)  # [B, 64, H//2, W//2]

        e2 = self.E2_Conv2d(e2)
        e2 = self.E2_ReLu(e2)  # [B, 128, H//2, W//2]
        e3 = self.E2_Pool(e2)  # [B, 128, H//4, W//4]

        e3 = self.E3_Conv2d(e3)
        e3 = self.E3_ReLu(e3)  # [B, 256 H//4, W//4]
        e4 = self.E3_Pool(e3)  # [B, 256, H//8, W//8]

        e4 = self.E4_Conv2d(e4)
        e4 = self.E4_ReLu(e4)  # [B, 512, H//8, W//8]
        e5 = self.E4_Pool(e4)  # [B, 512, H//16, W//16]

        d1 = self.D1_Deconv2d(e5)  # [B, 512, H//8, W//8]
        if self.ifADD:
            d1 = self.D1_CIN(d1 + e4, styleId)
            d1 = self.D1_ReLu(d1)
        else:
            d1 = self.D1_ReLu(self.D1_CIN(d1, styleId))
            d1 = torch.cat([d1, e4], dim=1)

        d2 = self.D2_Deconv2d(d1)  # [B, 256, H//4, W//4]
        if self.ifADD:
            d2 = self.D2_CIN(d2 + e3, styleId)
            d2 = self.D2_ReLu(d2)
        else:
            d2 = self.D2_ReLu(self.D2_CIN(d2, styleId))
            d2 = torch.cat([d2, e3], dim=1)

        d3 = self.D3_Deconv2d(d2)  # [B, 128, H//2, W//2]
        if self.ifADD:
            d3 = self.D3_CIN(d3 + e2, styleId)
            d3 = self.D3_ReLu(d3)
        else:
            d3 = self.D3_ReLu(self.D3_CIN(d3, styleId))
            d3 = torch.cat([d3, e2], dim=1)

        d4 = self.D4_Deconv2d(d3)  # [B, 64, H, W]
        if self.ifADD:
            d4 = self.D4_CIN(d4 + e1, styleId)
            d4 = self.D4_ReLu(d4)
        else:
            d4 = self.D4_ReLu(self.D4_CIN(d4, styleId))
            d4 = torch.cat([d4, e1], dim=1)

        y = self.out_Conv2d(d4)
        return y


class Generator_CNN(nn.Module):
    def __init__(self, inDim, outDim):
        super(Generator_CNN, self).__init__()
        self.E1_Conv2d = nn.Conv2d(inDim, 64, kernel_size=3, stride=1, padding=1)
        self.E1_ReLu = nn.ReLU(True)
        self.E1_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 64, H//2, W//2], Default H = 512, W = 512

        self.E2_Conv2d = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.E2_ReLu = nn.ReLU(True)
        self.E2_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 128, H//4, W//4]

        self.E3_Conv2d = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.E3_ReLu = nn.ReLU(True)
        self.E3_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 256, H//8, W//8]

        self.E4_Conv2d = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.E4_ReLu = nn.ReLU(True)
        self.E4_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 512, H//16, W//16]

        self.D1_Deconv2d = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # [B, 512, H//8, W//8]
        self.D1_ReLu = nn.ReLU(True)

        self.D2_Deconv2d = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # [B, 256, H//4, W//4]
        self.D2_ReLu = nn.ReLU(True)

        self.D3_Deconv2d = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # [B, 128, H//2, W//2]
        self.D3_ReLu = nn.ReLU(True)

        self.D4_Deconv2d = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # [B, 64, H, W]
        self.D4_ReLu = nn.ReLU(True)

        self.out_Conv2d = nn.Conv2d(64, outDim, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        e1 = self.E1_Conv2d(X)
        e1 = self.E1_ReLu(e1)  # [B, 64, H, W]
        e2 = self.E1_Pool(e1)  # [B, 64, H//2, W//2]

        e2 = self.E2_Conv2d(e2)
        e2 = self.E2_ReLu(e2)  # [B, 128, H//2, W//2]
        e3 = self.E2_Pool(e2)  # [B, 128, H//4, W//4]

        e3 = self.E3_Conv2d(e3)
        e3 = self.E3_ReLu(e3)  # [B, 256 H//4, W//4]
        e4 = self.E3_Pool(e3)  # [B, 256, H//8, W//8]

        e4 = self.E4_Conv2d(e4)
        e4 = self.E4_ReLu(e4)  # [B, 512, H//8, W//8]
        e5 = self.E4_Pool(e4)  # [B, 512, H//16, W//16]

        d1 = self.D1_Deconv2d(e5)  # [B, 256, H//8, W//8]
        d1 = self.D1_ReLu(d1 + e4)

        d2 = self.D2_Deconv2d(d1)  # [B, 128, H//4, W//4]
        d2 = self.D2_ReLu(d2 + e3)

        d3 = self.D3_Deconv2d(d2)  # [B, 64, H//2, W//2]
        d3 = self.D3_ReLu(d3 + e2)

        d4 = self.D4_Deconv2d(d3)  # [B, 64, H, W]
        d4 = self.D4_ReLu(d4 + e1)

        y = self.out_Conv2d(d4)
        return y


class Generator_CNN_Cat(nn.Module):
    def __init__(self, inDim, outDim):
        super(Generator_CNN_Cat, self).__init__()
        self.E1_Conv2d = nn.Conv2d(inDim, 64, kernel_size=3, stride=1, padding=1)
        self.E1_ReLu = nn.ReLU(True)
        self.E1_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 64, H//2, W//2], Default H = 512, W = 512

        self.E2_Conv2d = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.E2_ReLu = nn.ReLU(True)
        self.E2_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 128, H//4, W//4]

        self.E3_Conv2d = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.E3_ReLu = nn.ReLU(True)
        self.E3_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 256, H//8, W//8]

        self.E4_Conv2d = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.E4_ReLu = nn.ReLU(True)
        self.E4_Pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                                    ceil_mode=False)  # [B, 512, H//16, W//16]

        self.D1_Deconv2d = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # [B, 512, H//8, W//8]
        self.D1_ReLu = nn.ReLU(True)

        self.D2_Deconv2d = nn.ConvTranspose2d(512+512, 256, kernel_size=4, stride=2, padding=1)  # [B, 256, H//4, W//4]
        self.D2_ReLu = nn.ReLU(True)

        self.D3_Deconv2d = nn.ConvTranspose2d(256+256, 128, kernel_size=4, stride=2, padding=1)  # [B, 128, H//2, W//2]
        self.D3_ReLu = nn.ReLU(True)

        self.D4_Deconv2d = nn.ConvTranspose2d(128+128, 64, kernel_size=4, stride=2, padding=1)  # [B, 64, H, W]
        self.D4_ReLu = nn.ReLU(True)

        self.out_Conv2d = nn.Conv2d(64+64, outDim, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        e1 = self.E1_Conv2d(X)
        e1 = self.E1_ReLu(e1)  # [B, 64, H, W]
        e2 = self.E1_Pool(e1)  # [B, 64, H//2, W//2]

        e2 = self.E2_Conv2d(e2)
        e2 = self.E2_ReLu(e2)  # [B, 128, H//2, W//2]
        e3 = self.E2_Pool(e2)  # [B, 128, H//4, W//4]

        e3 = self.E3_Conv2d(e3)
        e3 = self.E3_ReLu(e3)  # [B, 256 H//4, W//4]
        e4 = self.E3_Pool(e3)  # [B, 256, H//8, W//8]

        e4 = self.E4_Conv2d(e4)
        e4 = self.E4_ReLu(e4)  # [B, 512, H//8, W//8]
        e5 = self.E4_Pool(e4)  # [B, 512, H//16, W//16]

        d1 = self.D1_Deconv2d(e5)  # [B, 256, H//8, W//8]
        d1 = self.D1_ReLu(d1)
        d1 = torch.cat([d1, e4], dim=1)

        d2 = self.D2_Deconv2d(d1)  # [B, 128, H//4, W//4]
        d2 = self.D2_ReLu(d2)
        d2 = torch.cat([d2, e3], dim=1)

        d3 = self.D3_Deconv2d(d2)  # [B, 64, H//2, W//2]
        d3 = self.D3_ReLu(d3)
        d3 = torch.cat([d3, e2], dim=1)

        d4 = self.D4_Deconv2d(d3)  # [B, 64, H, W]
        d4 = self.D4_ReLu(d4)
        d4 = torch.cat([d4, e1], dim=1)

        y = self.out_Conv2d(d4)
        return y


if __name__ == '__main__':
    model = batch_InstanceNorm2d(2, 5)
    params = list(model.parameters())
    print(len(params))
    for i in range(len(params)):
        print(params[i])
    x = torch.ones(2, 5, 10, 10)
    ids = [0, 1]
    y = model(x, ids)
    print(y.size())
