"""
 Copyright 2020 Mahmoud Afifi.
 Released under the MIT License.
 If you use this code, please cite the following paper:
 Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith
 Punnappurath, and Michael S Brown.
 CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks.
 arXiv preprint, 2020.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]


from .local_net import *
from .global_net import *

class CIEXYZNet(nn.Module):
    def __init__(self, device='cuda', localdepth=16, local_convdepth=32,
                 globaldepth=5, global_convdepth=64,
                 global_in=128, scale=0.25, state='orig'):
        super(CIEXYZNet, self).__init__()
        self.localdepth = localdepth
        self.local_convdepth = local_convdepth
        self.globaldepth = globaldepth
        self.global_convdepth = global_convdepth
        self.global_in = global_in
        self.scale = scale
        self.device = device
        self.state = state
        self.srgb2xyz_local_net = localSubNet(
            blockDepth=self.localdepth, convDepth=self.local_convdepth,
            scale=self.scale)
        self.xyz2srgb_local_net = localSubNet(
            blockDepth=self.localdepth, convDepth=self.local_convdepth,
            scale=self.scale)
        self.srgb2xyz_globa_net = globalSubNet(
            blockDepth=self.globaldepth,
            convDepth=self.global_convdepth, in_img_sz=self.global_in,
            device=self.device)
        self.xyz2srgb_globa_net = globalSubNet(
            blockDepth=self.globaldepth, convDepth=self.global_convdepth,
            in_img_sz=self.global_in, device=self.device)

    def forward_local(self, x, target):
        if target == "xyz":
            localLayer = self.srgb2xyz_local_net(x)
        elif target == 'srgb':
            localLayer = self.xyz2srgb_local_net(x)
        else:
            raise Exception("Wrong target. It is expected to be srgb or xyz, "
                            "but the input target is %s\n" % target)
        return localLayer

    def forward_global(self, x, target):
        if target == "xyz":
            m_v = self.srgb2xyz_globa_net(x)
        elif target == "srgb":
            m_v = self.xyz2srgb_globa_net(x)
        else:
            raise Exception("Wrong target. It is expected to be srgb or xyz, "
                            "but the input target is %s\n" % target)
        m = torch.reshape(m_v, (x.size(0), 3, 6))
        # multiply
        y = x.clone()
        for i in range(m.size(0)):
            temp = torch.mm(torch.squeeze(m[i, :, :]), self.kernel(
                torch.reshape(torch.squeeze(x[i, :, :, :]), (3, -1))))
            y[i, :, :, :] = torch.reshape(temp, (x.size(1), x.size(2),
                                                 x.size(3)))
        if self.state == 'orig':
            return y
        elif self.state == 'self-sup':
            return y, m

    def forward_srgb2xyz(self, srgb):
        l_xyz = srgb - self.forward_local(srgb, target='xyz')
        if self.state == 'orig':
            xyz = self.forward_global(l_xyz, target='xyz')
            return xyz
        elif self.state == 'self-sup':
            xyz, m = self.forward_global(l_xyz, target='xyz')
            return xyz, m

    def forward_xyz2srgb(self, xyz):
        if self.state == 'orig':
            g_srgb = self.forward_global(xyz, target='srgb')
            srgb = g_srgb + self.forward_local(g_srgb, target='srgb')
            return srgb
        elif self.state == 'self-sup':
            g_srgb, m = self.forward_global(xyz, target='srgb')
            srgb = g_srgb + self.forward_local(g_srgb, target='srgb')
            return srgb, m

    def forward(self, x):
        if self.state == 'orig':
            xyz = self.forward_srgb2xyz(x)
            srgb = self.forward_xyz2srgb(xyz)
            return xyz, srgb
        elif self.state == 'self-sup':
            x_1 = torch.squeeze(x[:, 0, :, :, :])
            x_2 = torch.squeeze(x[:, 1, :, :, :])
            xyz_1, m_inv_1 = self.forward_srgb2xyz(x_1)
            srgb_1, m_fwd_1 = self.forward_xyz2srgb(xyz_1)
            xyz_2, m_inv_2 = self.forward_srgb2xyz(x_2)
            srgb_2, m_fwd_2 = self.forward_xyz2srgb(xyz_2)
            return xyz_1, srgb_1, m_inv_1, m_fwd_1, xyz_2, srgb_2, m_inv_2, m_fwd_2

    @staticmethod
    def kernel(x):
        return torch.cat((x, x * x), dim=0)
