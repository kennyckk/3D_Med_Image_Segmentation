"""combine the model parts together to form the 3D attnUNet"""

from model_parts import *

class UNet3D(nn.Module):
    def __init__ (self,n_channels, n_classes, bilinear=False,Attn=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc=(DoubleConv3d(n_channels,64))
        self.down1=(Down3d(64,128))
        self.down2 = (Down3d(128, 256))
        self.down3 = (Down3d(256, 512))
        factor=2 if bilinear else 1
        self.down4=(Down3d(512, 1024//factor))

        self.up1 = (Up3d(1024, 512 // factor, bilinear, Attn))
        self.up2 = (Up3d(512, 256 // factor, bilinear,Attn))
        self.up3 = (Up3d(256, 128 // factor, bilinear,Attn))
        self.up4 = (Up3d(128, 64, bilinear,Attn))

        self.outc=(OutConv3d(64,n_classes))

    def forward(self,x): #input will be 3D input: B,C,H,W,D
        x1=self.inc(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x=self.up1(x5,x4)
        x=self.up2(x,x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits=self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class AttnUNet3D(UNet3D):
    def __init__(self,n_channels, n_classes, bilinear=False):
        super().__init__(n_channels, n_classes, bilinear=bilinear,Attn=True)

if __name__ =="__main__":
    device=torch.device("cuda")
    model=UNet3D(1,3).to(device)

    sample=torch.rand(1,1,32,32,32).to(device)

    out=model(sample)
    print(out.size())