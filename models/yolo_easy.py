from common2 import *
from yolo import Detect

class QuantizedNet(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedNet, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # FP32 model
        self.model_fp32 = model_fp32
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        # x[0] = self.dequant(x[0])
        # x[1] = self.dequant(x[1])
        # x[2] = self.dequant(x[2])
        # x[3] = self.dequant(x[3])
        x = self.dequant(x)
        return x

class output_maker(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.grid1, self.anchor_grid1 = _make_grid([19, 27, 44, 40, 38, 94], 160, 160, 8, torch.device("cuda:0"), torch.float32)
        self.grid2, self.anchor_grid2 = _make_grid([96, 68, 86, 152, 180, 137], 80, 80, 16, torch.device("cuda:0"), torch.float32)
        self.grid3, self.anchor_grid3 = _make_grid([140, 301, 303, 264, 238, 542], 40, 40, 32, torch.device("cuda:0"), torch.float32)
        self.grid4, self.anchor_grid4 = _make_grid([436, 615, 739, 380, 925, 792], 20, 20, 64, torch.device("cuda:0"), torch.float32)

        self.grid1 = nn.Parameter(self.grid1, requires_grad=False)
        self.grid2 = nn.Parameter(self.grid2, requires_grad=False)
        self.grid3 = nn.Parameter(self.grid3, requires_grad=False)
        self.grid4 = nn.Parameter(self.grid4, requires_grad=False)

        self.anchor_grid1 = nn.Parameter(self.anchor_grid1, requires_grad=False)
        self.anchor_grid2 = nn.Parameter(self.anchor_grid2, requires_grad=False)
        self.anchor_grid3 = nn.Parameter(self.anchor_grid3, requires_grad=False)
        self.anchor_grid4 = nn.Parameter(self.anchor_grid4, requires_grad=False)

    def forward(self, x):
        bs = x.shape[0]

        x23 = x[:,:,:25600]
        x26 = x[:,:,25600:32000]
        x29 = x[:,:,32000:33600]
        x32 = x[:,:,33600:34000]

        x23 = x23.view(bs, 3, 6, 160, 160).permute(0, 1, 3, 4, 2).contiguous()
        x26 = x26.view(bs, 3, 6, 80, 80).permute(0, 1, 3, 4, 2).contiguous()
        x29 = x29.view(bs, 3, 6, 40, 40).permute(0, 1, 3, 4, 2).contiguous()
        x32 = x32.view(bs, 3, 6, 20, 20).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:
            y23 = x23.sigmoid()
            y23[..., 0:2] = (y23[..., 0:2] * 2 + self.grid1) * 8
            y23[..., 2:4] = (y23[..., 2:4] * 2) ** 2 * self.anchor_grid1

            y26 = x26.sigmoid()
            y26[..., 0:2] = (y26[..., 0:2] * 2 + self.grid2) * 16
            y26[..., 2:4] = (y26[..., 2:4] * 2) ** 2 * self.anchor_grid2

            y29 = x29.sigmoid()
            y29[..., 0:2] = (y29[..., 0:2] * 2 + self.grid3) * 32
            y29[..., 2:4] = (y29[..., 2:4] * 2) ** 2 * self.anchor_grid3

            y32 = x32.sigmoid()
            y32[..., 0:2] = (y32[..., 0:2] * 2 + self.grid4) * 64
            y32[..., 2:4] = (y32[..., 2:4] * 2) ** 2 * self.anchor_grid4

            z = [y23.view(bs, -1, 6), y26.view(bs, -1, 6), y29.view(bs, -1, 6), y32.view(bs, -1, 6)]
            
            return (torch.cat(z, 1), [x23, x26, x29, x32])

        return x23, x26, x29, x32

def _make_grid(anchors, nx, ny, stride, device, dtype):
    anchors = torch.tensor(anchors).float().view(-1, 2).to(device)
    anchors /= stride
    # print(anchors)
    d = device
    t = dtype
    # print(t)
    shape = 1, 3, ny, nx, 2  # grid shape
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    yv, xv = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = (anchors * stride).view((1, 3, 1, 1, 2)).expand(shape)
    # print(grid, anchor_grid)
    return grid.to(device), anchor_grid.to(device)

class yolov5s6(nn.Module):
    def __init__(self):
        super(yolov5s6, self).__init__()

        self.layer_0to4 = nn.Sequential(
            Conv(3,32,6,2,2),
            Conv(32,64,3,2),
            C3(64,64,1),
            Conv(64,128,3,2),
            C3(128,128,2),
            C3(128,128,2)
        )

        self.layer_5to6 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C3(256, 256, 3),
            C3(256, 256, 3),
            C3(256, 256, 3)
        )

        self.layer_7to8 = nn.Sequential(
            Conv(256, 384, 3, 2),
            C3(384, 384, 1)
        )

        self.layer_9to12 = nn.Sequential(
            Conv(384, 512, 3, 2),
            C3(512, 512, 1),
            SPPF(512, 512, 5),
            Conv(512, 384, 1, 1)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.layer_15to16 = nn.Sequential(
            C3(768, 384, 1, False),
            Conv(384, 256, 1, 1)
        )

        self.layer_19to20 = nn.Sequential(
            C3(512, 256, 1, False),
            Conv(256, 128, 1, 1)
        )

        self.layer_23 = C3(256, 128, 1, False)
        self.layer_24 = Conv(128, 128, 3, 2)

        self.layer_26 = C3(256, 256, 1, False)
        self.layer_27 = Conv(256, 256, 3, 2)

        self.layer_29 = C3(512, 384, 1, False)
        self.layer_30 = Conv(384, 384, 3, 2)

        self.layer_32 = C3(768, 512, 1, False)

        self.scale1 = nn.Conv2d(128, 18, 1)
        self.scale2 = nn.Conv2d(256, 18, 1)
        self.scale3 = nn.Conv2d(384, 18, 1)
        self.scale4 = nn.Conv2d(512, 18, 1)

    def forward(self, x):
        x4 = self.layer_0to4(x)
        x6 = self.layer_5to6(x4)
        x8 = self.layer_7to8(x6)
        x12 = self.layer_9to12(x8)

        x16 = self.layer_15to16(torch.cat((self.upsample(x12), x8), dim=1))
        x20 = self.layer_19to20(torch.cat((self.upsample(x16), x6), dim=1))
        x23 = self.layer_23(torch.cat((self.upsample(x20), x4), dim=1))

        x26 = self.layer_26(torch.cat((self.layer_24(x23), x20), dim=1))
        x29 = self.layer_29(torch.cat((self.layer_27(x26), x16), dim=1))
        x32 = self.layer_32(torch.cat((self.layer_30(x29), x12), dim=1))

        x23 = self.scale1(x23)
        x26 = self.scale2(x26)
        x29 = self.scale3(x29)
        x32 = self.scale4(x32)
        
        bs = x.shape[0]
        x23 = x23.reshape(bs,18,-1)
        x26 = x26.reshape(bs,18,-1)
        x29 = x29.reshape(bs,18,-1)
        x32 = x32.reshape(bs,18,-1)

        x = torch.cat((x23,x26,x29,x32), dim=2)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0")
    a = yolov5s6().to(device)
    b = torch.rand(1,3,1280,1280).to(device)
    b = (b-0.5)/0.5

    a.train()
    c = a(b)
    print(c.shape)
    # print(len(c))

    d = output_maker().to(device)
    e = d(c)
    print(e[0].shape, e[1].shape, e[2].shape, e[3].shape)
    
    d.eval()
    e,f = d(c)

    print(e.shape)
    print(f[0].shape, f[1].shape, f[2].shape, f[3].shape)