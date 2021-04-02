import torch
import torch.nn as nn
import numpy as np
from VGG_Model import VGG19, get_Style_Mask_Model, get_Content_Mask_Model, vgg_mean, vgg_std


def content_loss(noise: torch.Tensor, image: torch.Tensor):
    """
        Simple SSE loss over the generated image and the content image
            arXiv:1508.06576v2 - equation (1)
    """
    return 1/2. * torch.sum(torch.pow(noise - image, 2))


def LapSmooth_loss(Lap: torch.Tensor, vert: torch.Tensor):
    lv = torch.matmul(Lap, vert)
    loss = torch.bmm(lv.unsqueeze(1), lv.unsqueeze(-1))
    return torch.sum(loss) / float(vert.size()[0])


class BatchFeatureLoss_Model(nn.Module):
    def __init__(self, device, c_alpha, s_beta, s_layWei=[1., 1., 1., 1., 1.]):
        super(BatchFeatureLoss_Model, self).__init__()
        self.device = device
        # init the model
        self.vgg = VGG19(device).to(device).eval()
        # replace the MaxPool with the AvgPool layers
        for name, child in self.vgg.vgg.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.vgg.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

        # get mask operation layers
        self.contentMask_net = get_Content_Mask_Model(self.vgg.get_content_layer())
        self.styleMask_net = get_Style_Mask_Model(self.vgg.get_style_layers())
        # lock the gradients
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.c_alpha = c_alpha
        self.s_betat = s_beta
        self.s_layWei = s_layWei

    def forward(self, X, SG, CX, MX):
        # prepare mask
        style_masks = [self.styleMask_net[i](MX) for i in range(len(self.styleMask_net))]
        cont_masks = self.contentMask_net(MX)
        style_NR = [torch.sum(style_masks[i], dim=(2, 3)) for i in range(len(style_masks))]
        cont_NR = torch.sum(cont_masks, dim=(2, 3))

        # compute style loss
        x = X * MX
        sg = SG * MX
        cx = CX * MX
        X_style_act = self.vgg.get_style_activations(x)
        G_style_act = self.vgg.get_style_activations(sg)
        for i in range(len(X_style_act)):
            X_style_act[i] = X_style_act[i] * style_masks[i]
            xb, xc, xh, xw = X_style_act[i].size()
            X_style_act[i] = X_style_act[i].view(xb, xc, xh * xw)

            G_style_act[i] = G_style_act[i] * style_masks[i]
            gb, gc, gh, gw = G_style_act[i].size()
            G_style_act[i] = G_style_act[i].view(gb, gc, gh * gw)

        X_grams = [torch.bmm(X_style_act[i], X_style_act[i].transpose(1, 2)) for i in range(len(X_style_act))]
        G_grams = [torch.bmm(G_style_act[i], G_style_act[i].transpose(1, 2)) for i in range(len(G_style_act))]
        style_loss = 0.
        for i in range(len(X_style_act)):
            a = torch.sum(torch.pow(X_grams[i] - G_grams[i], 2), dim=(1, 2))
            b = torch.pow(style_NR[i] * 2, 2).squeeze(-1)
            style_loss += self.s_layWei[i] * (a / (5. * b))

        # compute content loss
        X_Content_act = self.vgg.get_content_activations(x) * cont_masks
        CX_Content_act = self.vgg.get_content_activations(cx) * cont_masks
        cc_loss = content_loss(X_Content_act, CX_Content_act)

        return self.c_alpha * cc_loss, self.s_betat * torch.sum(style_loss.to(self.device))


class DFeatureLoss_Model(nn.Module):
    def __init__(self, device, c_alpha, s_beta, mask_Img, s_layWei = [1., 1., 1., 1., 1.]):
        super(DFeatureLoss_Model, self).__init__()
        self.device = device
        # init the model
        self.vgg = VGG19(device).to(device).eval()
        # replace the MaxPool with the AvgPool layers
        for name, child in self.vgg.vgg.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.vgg.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

        # get mask operation layers
        self.contentMask_net = get_Content_Mask_Model(self.vgg.get_content_layer())
        self.styleMask_net = get_Style_Mask_Model(self.vgg.get_style_layers())
        # lock the gradients
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.mask_Img = mask_Img
        self.style_masks = [self.styleMask_net[i](mask_Img) for i in range(len(self.styleMask_net))]
        self.cont_masks = self.contentMask_net(mask_Img)

        self.style_NR = [torch.sum(self.style_masks[i], dim=(2, 3)) for i in range(len(self.style_masks))]
        self.cont_NR = torch.sum(self.cont_masks, dim=(2, 3))
        self.c_alpha = c_alpha
        self.s_betat = s_beta
        self.s_layWei = s_layWei

    def forward(self, X, SG, CX):
        x = X * self.mask_Img
        sg = SG * self.mask_Img
        cx = CX * self.mask_Img
        X_style_act = self.vgg.get_style_activations(x)
        G_style_act = self.vgg.get_style_activations(sg)
        for i in range(len(X_style_act)):
            X_style_act[i] = X_style_act[i] * self.style_masks[i]
            xb, xc, xh, xw = X_style_act[i].size()
            X_style_act[i] = X_style_act[i].view(xb, xc, xh * xw)

            G_style_act[i] = G_style_act[i] * self.style_masks[i]
            gb, gc, gh, gw = G_style_act[i].size()
            G_style_act[i] = G_style_act[i].view(gb, gc, gh * gw)

        X_grams = [torch.bmm(X_style_act[i], X_style_act[i].transpose(1, 2)) for i in range(len(X_style_act))]
        G_grams = [torch.bmm(G_style_act[i], G_style_act[i].transpose(1, 2)) for i in range(len(G_style_act))]

        style_loss = 0.
        for i in range(len(X_style_act)):
            a = torch.sum(torch.pow(X_grams[i] - G_grams[i], 2), dim=(1, 2))
            b = torch.pow(self.style_NR[i] * 2, 2).squeeze(-1)
            style_loss += self.s_layWei[i] * (a / (5.*b))

        X_Content_act = self.vgg.get_content_activations(x) * self.cont_masks
        CX_Content_act = self.vgg.get_content_activations(cx) * self.cont_masks
        cc_loss = content_loss(X_Content_act, CX_Content_act)

        return self.c_alpha*cc_loss, self.s_betat * torch.sum(style_loss.to(self.device))


class Loss_NormalCrossVert(nn.Module):
    def __init__(self, vertEdges_0, vertEdges_1, EdgeCounts, numV, device):
        super(Loss_NormalCrossVert, self).__init__()
        self.edge_0 = vertEdges_0
        self.edge_1 = vertEdges_1
        self.edge_c = EdgeCounts
        self.r = np.sum(EdgeCounts)
        assert (self.r == self.edge_1.shape[0])
        assert (self.r == self.edge_0.shape[0])
        self.numV = numV
        self.device = device

    def forward(self, normalArray: torch.Tensor, vertArray: torch.Tensor):
        assert (normalArray.size()[0] == self.numV)
        assert (vertArray.size()[0] == self.numV)
        v_0 = vertArray[self.edge_0, :]
        v_1 = vertArray[self.edge_1, :]
        ve = v_0 - v_1
        led = torch.sqrt(torch.bmm(ve.unsqueeze(1), ve.unsqueeze(-1))).squeeze(-1)
        ve = ve / led
        vn = torch.cat([normalArray[i].repeat(self.edge_c[i], 1) for i in range(self.numV)], dim=0)
        dotV = torch.bmm(vn.unsqueeze(1), ve.unsqueeze(-1))
        loss = torch.sum(torch.pow(dotV, 2)) / float(self.r)
        return loss