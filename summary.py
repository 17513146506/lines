
from torchsummary import summary

from nets.unet import Unet

if __name__ == "__main__":
    model = Unet(num_classes = 2, backbone = 'resnet50').train().cuda()
    summary(model, (3, 512, 512))
    