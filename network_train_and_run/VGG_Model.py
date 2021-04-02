import torch.nn as nn
import torchvision.models as models
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]


class Mask_layers(nn.Module):
    def __init__(self, vgg, method='simple'):
        super(Mask_layers, self).__init__()
        self.method = method
        self.model = torch.nn.Sequential()
        if self.method is 'simple':
            for name, child in vgg.named_children():
                if isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d):
                    self.model.add_module(name, nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_Content_Mask_Model(Content_layer):
    return Mask_layers(Content_layer).eval()


def get_Style_Mask_Model(Style_layers):
    return [Mask_layers(Style_layers[i]).eval() for i in range(len(Style_layers))]


class vgg_Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(vgg_Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).to(device).view(-1, 1, 1)
        self.std = torch.tensor(std).to(device).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# define the VGG
class VGG19(nn.Module):
    def __init__(self, device, ifNorm=False):
        super(VGG19, self).__init__()

        # load the vgg model's features
        if ifNorm:
            self.normalization = vgg_Normalization(vgg_mean, vgg_std, device).to(device)
        self.vgg = models.vgg19(pretrained=True).features
        self.device = device
        self.ifNorm = ifNorm

    def get_content_layer(self):
        return self.vgg[:7]

    def get_style_layers(self):
        return [self.vgg[:4]] + [self.vgg[:7]] + [self.vgg[:12]] + [self.vgg[:21]] + [self.vgg[:30]]

    def get_content_activations(self, x: torch.Tensor) \
            -> torch.Tensor:
        """
            Extracts the features for the content loss from the block4_conv2 of VGG19
            Args:
                x: torch.Tensor - input image we want to extract the features of
            Returns:
                features: torch.Tensor - the activation maps of the block2_conv1 layer
        """
        y = x
        if self.ifNorm:
            y = self.normalization(x)
        features = self.vgg[:7](y)
        return features

    def get_style_activations(self, x):
        """
            Extracts the features for the style loss from the block1_conv1,
                block2_conv1, block3_conv1, block4_conv1, block5_conv1 of VGG19
            Args:
                x: torch.Tensor - input image we want to extract the features of
            Returns:
                features: list - the list of activation maps of the block1_conv1,
                    block2_conv1, block3_conv1, block4_conv1, block5_conv1 layers
        """
        y = x
        if self.ifNorm:
            y = self.normalization(x)
        features = [self.vgg[:4](y)] + [self.vgg[:7](y)] + [self.vgg[:12](y)] + [self.vgg[:21](y)] + [self.vgg[:30](y)]
        return features

    def forward(self, x):
        y = x
        if self.ifNorm:
            y = self.normalization(x)
        return self.vgg(y)


class PatchStyleFeatures(nn.Module):
    def __init__(self, device):
        super(PatchStyleFeatures, self).__init__()
        self.device = device
        # init the model
        self.vgg = VGG19(device).to(device).eval()
        # replace the MaxPool with the AvgPool layers
        for name, child in self.vgg.vgg.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.vgg.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

        # get mask operation layers
        self.styleMask_net = get_Style_Mask_Model(self.vgg.get_style_layers())
        # lock the gradients
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, X, MX, layID):
        # prepare mask
        style_masks = self.styleMask_net[layID](MX)
        style_NR = torch.sum(style_masks, dim=(2, 3))

        x = X * MX
        X_style_act = self.vgg.get_style_activations(x)[layID]
        X_style_act = X_style_act * style_masks
        xb, xc, xh, xw = X_style_act.size()
        X_style_act = X_style_act.view(xb, xc, xh * xw)

        b = torch.pow(style_NR * 2, 2)
        GramFeature = torch.bmm(X_style_act, X_style_act.transpose(1, 2))
        GramFeature = GramFeature.view(xb, xc * xc)
        GramFeature = GramFeature / b
        return GramFeature


class GramStyleFeatures(nn.Module):
    def __init__(self, device, mask_Img):
        super(GramStyleFeatures, self).__init__()
        self.device = device
        # init the model
        self.vgg = VGG19(device).to(device).eval()
        # replace the MaxPool with the AvgPool layers
        for name, child in self.vgg.vgg.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.vgg.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

        # get mask operation layers
        self.styleMask_net = get_Style_Mask_Model(self.vgg.get_style_layers())
        # lock the gradients
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.mask_Img = mask_Img
        self.style_masks = [self.styleMask_net[i](mask_Img) for i in range(len(self.styleMask_net))]
        self.style_NR = [torch.sum(self.style_masks[i], dim=(2, 3)) for i in range(len(self.style_masks))]

    def forward(self, X):
        x = X * self.mask_Img
        xb = x.size()[0]
        X_style_act = self.vgg.get_style_activations(x)
        layDim = []
        for i in range(len(X_style_act)):
            X_style_act[i] = X_style_act[i] * self.style_masks[i]
            xb, xc, xh, xw = X_style_act[i].size()
            X_style_act[i] = X_style_act[i].view(xb, xc, xh * xw)
            layDim.append(xc*xc)

        X_grams = [torch.bmm(X_style_act[i], X_style_act[i].transpose(1, 2)) for i in range(len(X_style_act))]
        X_grams = torch.cat([x.view(xb, -1) for x in X_grams], dim=1)
        return X_grams, layDim


if __name__ == '__main__':
    vgg = VGG19(vggmean=vgg_mean, vggstd=vgg_std, device="cpu").eval()
    for name, child in vgg.get_content_layer().named_children():
        print(name, child)

    maskImg = Image.open('./styles/mask.png')
    transform = transforms.Compose([transforms.Resize((800, 800)),
                                    transforms.ToTensor()])
    maskImg = transform(maskImg)[0, :, :].unsqueeze(0).unsqueeze(0)
    print(maskImg.size())
    content_MaskNet = get_Content_Mask_Model(vgg.get_content_layer())
    style_MaskNet = get_Style_Mask_Model(vgg.get_style_layers())
    print("ContentMask:", content_MaskNet)
    print("StyleMask: ")
    for i in range(len(style_MaskNet)):
        print(style_MaskNet[i])

    x = style_MaskNet[0](maskImg)
    tIn = torch.ones_like(x)
    tIn = tIn * x
    print(torch.sum(x))
    # print(x.size())
    save_image(tIn, filename='./cont_m.png')

