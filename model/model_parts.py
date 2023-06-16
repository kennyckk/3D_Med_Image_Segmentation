import torch
import torch.nn as nn
import torch.nn.functional as F

"""modification to have attn Unet with 3D volumetric data"""

class DoubleConv3d(nn.Module):
    def __init__(self,in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv3d = nn.Sequential(
            #input dim: (B,C_in,H,W,D)
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # input dim: (B,C_out,H,W,D)
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # input dim: (B,C_out,H,W,D)
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv3d(x)

class Down3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv3d = nn.Sequential(
            # input dim: (B,C_in,H,W,D)
            nn.MaxPool3d(2),
            # input dim: (B,C_in,H//2,W//2,D//2)
            DoubleConv3d(in_channels, out_channels)
            # input dim: (B,C_out,H//2,W//2,D//2)
        )
    def forward(self,x):
        return self.maxpool_conv3d(x)

class AttentionBlock(nn.Module):
    def __init__(self,f_g,f_l,f_int):
        super().__init__()
        # to make B,C,H,W,D--> B,f_l,H_g,W_g,D_g
        self.W_gate=nn.Sequential(
            nn.Conv3d(f_g,f_int,kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(f_int)
        )
        # to also make B,C,H,W,D--> B,f_l,H_g,W_g,D_g
        self.W_x=nn.Sequential(
            nn.Conv3d(f_l,f_int,kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1), #only one channel left
            nn.Sigmoid()
        )

        self.relu=nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        #gate and skip should already in the same dims
        assert gate.size() == skip.size(), "W_gate {} & W_x {} should match in dims".format(
            gate.size(), skip.size())

        g=self.W_gate(gate) #gate becomes B,f_int,H,W,D
        s=self.W_x(skip)    # skip becomes B,f_int,H,W,D
        psi=self.relu(s+g)
        psi=self.psi(psi) #psi is now B,1,H,W,D
        out=psi*skip # (B,1,H,W,D) *(B,f_l,H,W,D)-->(B,f_l,H,W,D)
        return out

class Up3d(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=False,Attn=False):
        super().__init__()
        self.attn=Attn
        if bilinear:
            #to be implemented
            pass
        else:
            self.up3d=nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.attn_blk=AttentionBlock(in_channels//2,in_channels//2,out_channels//2)
            self.conv3d=DoubleConv3d(in_channels, out_channels)

    def padding3d(self, gate,skip):
        """to ensure the gating signal same dim as skip-connected map"""
        diffX=skip.size()[-3]-gate.size()[-3]
        diffY=skip.size()[-2]-gate.size()[-2]
        diffZ=skip.size()[-1]-gate.size()[-1]

        gate=F.pad(gate,[diffX//2,diffX-diffX//2,
                         diffY//2,diffY-diffY//2,
                         diffZ//2,diffZ-diffZ//2])
        return gate


    def forward(self, x_1,skip): #x_1 is from lower scale x_2 is from upper
        # input dim: (B,C_in,H,W,D)
        g=self.up3d(x_1)
        # g_input dim: (B,C_in//2,H*2,W*2,D*2)
        g=self.padding3d(g,skip) #pad g to match skip dims in H,W,D --> B,f_g,H,W,D

        # the gate and skip going to attn blk are in same dim
        s=self.attn_blk(g,skip) if self.attn else skip #if no attn --> 3D UNet
        #concat the filter skip-connected FM & padded g from lower level
        out=torch.cat([s,g],dim=1)

        out=self.conv3d(out)

        return out

class OutConv3d(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv=nn.Conv3d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        return self.conv(x)


if __name__ =="__main__":
    g=torch.rand(1,32,21,21,21)
    x=torch.rand(1,16,42,42,42)
    unit_test=Up3d(32,16,bilinear=False)

    out=unit_test(g,x)
    print(out.size())
