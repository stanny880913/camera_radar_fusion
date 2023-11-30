from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np


class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformConv3d, self).__init__()
        self.kernel_size = kernel_size
        N = kernel_size ** 3
        self.stride = stride
        self.padding = padding
        self.zero_padding = nn.ConstantPad3d(padding, 0)
        self.conv_kernel = nn.Conv3d(in_channels * N, out_channels, kernel_size=1, bias=bias)
        self.offset_conv_kernel = nn.Conv3d(in_channels, N * 3, kernel_size=kernel_size, padding=padding, bias=bias)
        
        self.mode = "deformable"
        
    def deformable_mode(self, on=True): # 
        if on:
            self.mode = "deformable"
        else:
            self.mode = "regular"
        
    def forward(self, x):
        if self.mode == "deformable":
            offset = self.offset_conv_kernel(x)
        else:
            b, c, h, w, d = x.size()
            offset = torch.zeros(b, 3 * self.kernel_size ** 3, h, w, d).to(x)
        
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 3

        if self.padding:
            x = self.zero_padding(x)

        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)
        p = p[:, :, ::self.stride, ::self.stride, ::self.stride]

        # (b, h, w, d, 3N), N == ks ** 3, 3N - 3 coords for each point on the activation map
        p = p.contiguous().permute(0, 2, 3, 4, 1) # 5D array
        
        q_sss = Variable(p.data, requires_grad=False).floor() # point with all smaller coords
#         q_sss = p.data.floor() - same? / torch.Tensor(p.data).floor()
        q_lll = q_sss + 1 # all larger coords

        # 8 neighbor points with integer coords
        q_sss = torch.cat([
            torch.clamp(q_sss[..., :N], 0, x.size(2) - 1), # h_coord
            torch.clamp(q_sss[..., N:2 * N], 0, x.size(3) - 1), # w_coord
            torch.clamp(q_sss[..., 2 * N:], 0, x.size(4) - 1) # d_coord
        ], dim=-1).long()
        q_lll = torch.cat([
            torch.clamp(q_lll[..., :N], 0, x.size(2) - 1), # h_coord
            torch.clamp(q_lll[..., N:2 * N], 0, x.size(3) - 1), # w_coord
            torch.clamp(q_lll[..., 2 * N:], 0, x.size(4) - 1) # d_coord
        ], dim=-1).long()
        q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)

        # (b, h, w, d, N)
        mask = torch.cat([
            p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
            p[..., N:2 * N].lt(self.padding) + p[..., N:2 * N].gt(x.size(3) - 1 - self.padding),
            p[..., 2 * N:].lt(self.padding) + p[..., 2 * N:].gt(x.size(4) - 1 - self.padding),
        ], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p)) # все еще непонятно, что тут происходит за wtf
        p = p * (1 - mask) + floor_p * mask
        
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),
            torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1),
        ], dim=-1)
        
        # trilinear kernel (b, h, w, d, N)  
        g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        
        # get values in all 8 neighbor points
        # (b, c, h, w, d, N) - 6D-array
        x_q_sss = self._get_x_q(x, q_sss, N)
        x_q_lll = self._get_x_q(x, q_lll, N)
        x_q_ssl = self._get_x_q(x, q_ssl, N)
        x_q_sls = self._get_x_q(x, q_sls, N)
        x_q_sll = self._get_x_q(x, q_sll, N)
        x_q_lss = self._get_x_q(x, q_lss, N)
        x_q_lsl = self._get_x_q(x, q_lsl, N)
        x_q_lls = self._get_x_q(x, q_lls, N)
        
        # (b, c, h, w, d, N)
        x_offset = g_sss.unsqueeze(dim=1) * x_q_sss + \
                   g_lll.unsqueeze(dim=1) * x_q_lll + \
                   g_ssl.unsqueeze(dim=1) * x_q_ssl + \
                   g_sls.unsqueeze(dim=1) * x_q_sls + \
                   g_sll.unsqueeze(dim=1) * x_q_sll + \
                   g_lss.unsqueeze(dim=1) * x_q_lss + \
                   g_lsl.unsqueeze(dim=1) * x_q_lsl + \
                   g_lls.unsqueeze(dim=1) * x_q_lls
        
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)
        
        return out
    
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), 
            indexing='ij')
        
        # (3N, 1) - 3 coords for each of N offsets
        # (x1, ... xN, y1, ... yN, z1, ... zN)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)
        
        return p_n
    
    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)

        return p_0
    
    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype).to(offset)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype).to(offset)
        p = p_0 + p_n + offset
        
        return p
    
    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()
        
        #           (0, 1, 2, 3, 4)
        # x.size == (b, c, h, w, d)
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
        # (b, c, h*w*d)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        # offset_x * w * d + offset_y * d + offset_z
        index = q[..., :N] * padded_w * padded_d + q[..., N:2 * N] * padded_d + q[..., 2 * N:]
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)
        
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)
        
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, d, N = x_offset.size()
        x_offset = x_offset.permute(0, 1, 5, 2, 3, 4)
        x_offset = x_offset.contiguous().view(b, c * N, h, w, d)

        return x_offset
        
def deform_conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return DeformConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class DeformBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DeformBasicBlock, self).__init__()
        self.conv1 = deform_conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = deform_conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
    
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape
    def forward(self, input):
        return input.view((-1,) + self.shape)


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class DeformVoxResNet(nn.Module):
    def __init__(self, input_shape=(128, 128, 128), num_classes=2, n_filters=32, stride=2, n_blocks=3, 
                 n_flatten_units=None, dropout=0, n_fc_units=128, deform_idx=[], deform_block_idx=[], first_kernel_size=7):
        super(self.__class__, self).__init__()
        """
        Args:
        --- deform_idx - (list) - list of indices of convolutional layers to make deformable (starting from 1).
        """
        self.deform_idx = deform_idx
        self.deform_block_idx = deform_block_idx
        
        self.model = nn.Sequential()

        if 1 in deform_idx:
            self.model.add_module("conv3d_1", DeformConv3d(1, n_filters, kernel_size=first_kernel_size, padding=1, stride=stride)) # n * (x/s) * (y/s) * (z/s)
        else:
            self.model.add_module("conv3d_1", nn.Conv3d(1, n_filters, kernel_size=first_kernel_size, padding=1, stride=stride)) # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("batch_norm_1", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_1", nn.ReLU(inplace=True))
        
        if 2 in deform_idx:
            self.model.add_module("conv3d_2", DeformConv3d(n_filters, n_filters, kernel_size=3, padding=1)) # n * (x/s) * (y/s) * (z/s)
        else:
            self.model.add_module("conv3d_2", nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1)) # n * (x/s) * (y/s) * (z/s)        
        self.model.add_module("batch_norm_2", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_2", nn.ReLU(inplace=True))

#         1
        if 3 in deform_idx:
            self.model.add_module("conv3d_3", DeformConv3d(n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2)) # 2n * (x/2s) * (y/2s) * (z/2s)
        else:
            self.model.add_module("conv3d_3", nn.Conv3d(n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2)) # 2n * (x/2s) * (y/2s) * (z/2s)
        #
        if 1 in deform_block_idx:
            self.model.add_module("block_1", DeformBasicBlock(2 * n_filters, 2 * n_filters))
        else:
            self.model.add_module("block_1", BasicBlock(2 * n_filters, 2 * n_filters))
        #
        if 2 in deform_block_idx:
            self.model.add_module("block_2", DeformBasicBlock(2 * n_filters, 2 * n_filters))
        else:
            self.model.add_module("block_2", BasicBlock(2 * n_filters, 2 * n_filters))
        self.model.add_module("batch_norm_3", nn.BatchNorm3d(2 * n_filters))
        self.model.add_module("activation_3", nn.ReLU(inplace=True))

#         2
        if n_blocks >= 2:
            if 4 in deform_idx:
                self.model.add_module("conv3d_4", DeformConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2)) # 2n * (x/4s) * (y/4s) * (z/4s)
            else:
                self.model.add_module("conv3d_4", nn.Conv3d(2 * n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2)) # 2n * (x/4s) * (y/4s) * (z/4s)
            #
            if 3 in deform_block_idx:
                self.model.add_module("block_3", DeformBasicBlock(2 * n_filters, 2 * n_filters))
            else:
                self.model.add_module("block_3", BasicBlock(2 * n_filters, 2 * n_filters))
            #
            if 4 in deform_block_idx:
                self.model.add_module("block_4", DeformBasicBlock(2 * n_filters, 2 * n_filters))
            else:
                self.model.add_module("block_4", BasicBlock(2 * n_filters, 2 * n_filters))
            self.model.add_module("batch_norm_4", nn.BatchNorm3d(2 * n_filters))
            self.model.add_module("activation_4", nn.ReLU(inplace=True))

#         3
        if n_blocks >= 3:
            if 5 in deform_idx:
                self.model.add_module("conv3d_5", DeformConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2)) # 4n * (x/8s) * (y/8s) * (z/8s)
            else:
                self.model.add_module("conv3d_5", nn.Conv3d(2 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2)) # 4n * (x/8s) * (y/8s) * (z/8s)
            #
            if 5 in deform_block_idx:
                self.model.add_module("block_5", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_5", BasicBlock(4 * n_filters, 4 * n_filters))
            #
            if 6 in deform_block_idx:
                self.model.add_module("block_6", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_6", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_5", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_5", nn.ReLU(inplace=True))

#         4
        if n_blocks >= 4:
            if 6 in deform_idx:
                self.model.add_module("conv3d_6", DeformConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2)) # 4n * (x/16s) * (y/16s) * (z/16s)
            else:
                self.model.add_module("conv3d_6", nn.Conv3d(4 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2)) # 4n * (x/16s) * (y/16s) * (z/16s)
            #
            if 7 in deform_block_idx:
                self.model.add_module("block_7", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_7", BasicBlock(4 * n_filters, 4 * n_filters))
            #
            if 8 in deform_block_idx:
                self.model.add_module("block_8", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_8", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_6", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_6", nn.ReLU(inplace=True))

#         self.model.add_module("max_pool3d_1", nn.MaxPool3d(kernel_size=3)) # (b/2)n * (x/(2^b)sk) * (y/(2^b)sk) * (z/(2^b)sk) ?
        
        if n_flatten_units is None:
            n_flatten_units = (n_blocks) // 2 * 2 * n_filters * np.prod(np.array(input_shape) // (2 ** n_blocks * stride))
        print(n_flatten_units)
        
        self.model.add_module("flatten_1", Flatten())
        self.model.add_module("fully_conn_1", nn.Linear(n_flatten_units*2, n_fc_units))
        self.model.add_module("activation_6", nn.ReLU(inplace=True))
        self.model.add_module("dropout_1", nn.Dropout(dropout))

        self.model.add_module("fully_conn_2", nn.Linear(n_fc_units, num_classes))

            
    def forward(self, x):
        print("Input shape:", x.shape)
        return self.model(x)