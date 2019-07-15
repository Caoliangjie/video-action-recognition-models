import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    
    def __init__(self, dim_in, dim_out, down_sampling=False, is_decomposed=False, freeze_bn=False):
        
        super(BasicBlock, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.down_sampling = down_sampling
        self.is_decomposed = is_decomposed
        self.block = nn.ModuleDict()
        
        kernels = (3, 3, 3)
        if down_sampling:
            strides = (2, 2, 2)
        else:
            strides = (1, 1, 1)
        pads = (1, 1, 1)
        
        self.add_conv(dim_in, dim_out, kernels, strides, pads, 1, is_decomposed, freeze_bn)
        self.block["bn1"] = nn.BatchNorm3d(dim_out, track_running_stats=(not freeze_bn))
        self.block["relu"] = nn.ReLU(inplace=True)
        
        self.add_conv(dim_out, dim_out, kernels, (1, 1, 1), pads, 2, is_decomposed, freeze_bn)
        self.block["bn2"] = nn.BatchNorm3d(dim_out, track_running_stats=(not freeze_bn))
        
        if (dim_in != dim_out) or down_sampling:
            self.block["shortcut"] = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=strides,
                padding=(0, 0, 0),
                bias=False)
            self.block["shortcut_bn"] = nn.BatchNorm3d(dim_out, track_running_stats=(not freeze_bn))
        
    def add_conv(self, dim_in, dim_out, kernels, strides=(1, 1, 1), pads=(0, 0, 0), conv_idx=1, is_decomposed=False, freeze_bn=False):
        if is_decomposed:
            i = 3 * dim_in * dim_out * kernels[1] * kernels[2]
            i /= dim_in * kernels[1] * kernels[2] + 3 * dim_out
            dim_inner = int(i)
            
            # 3x1x1 layer
            conv_middle = "conv{}_middle".format(str(conv_idx))
            self.block[conv_middle] = nn.Conv3d(
                dim_in,
                dim_inner,
                kernel_size=(1, kernels[1], kernels[2]),
                stride=(1, strides[1], strides[2]),
                padding=(0, pads[1], pads[2]),
                bias=False)
            
            bn_middle = "bn{}_middle".format(str(conv_idx))
            self.block[bn_middle] = nn.BatchNorm3d(dim_inner, track_running_stats=(not freeze_bn))
            
            relu_middle = "relu{}_middle".format(str(conv_idx))
            self.block[relu_middle] = nn.ReLU(inplace=True)
            
            # 1x3x3 layer
            conv = "conv{}".format(str(conv_idx))
            self.block[conv] = nn.Conv3d(
                dim_inner,
                dim_out,
                kernel_size=(kernels[0], 1, 1),
                stride=(strides[0], 1, 1),
                padding=(pads[0], 0, 0),
                bias=False)
        else:
            # 3x3x3 layer
            conv = "conv{}".format(str(conv_idx))
            self.block[conv] = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=kernels,
                stride=strides,
                padding=pads,
                bias=False)
        
    def forward(self, x):
        residual = x
        out = x
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)

        if (self.dim_in != self.dim_out) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
            
        out += residual
        out = self.block["relu"](out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 model_depth,
                 sample_duration,
                 freeze_bn=False,
                 num_classes=400):
        
        layers, block, final_temporal_kernel, final_spatial_kernel = obtain_arc(sample_duration, model_depth)
        
        super(ResNet, self).__init__()
        
        self.conv1_middle = nn.Conv3d(
            3,
            45,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False)
        self.bn1_middle = nn.BatchNorm3d(45, track_running_stats=(not freeze_bn))
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv3d(
            45,
            64,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=(not freeze_bn))
        
        self.layer1 = self._make_layer(block_func=block, dim_in=64, 
                                       dim_out=64, layer_id=1, 
                                       num_blocks=layers[0], 
                                       is_decomposed=True,
                                       freeze_bn=freeze_bn)
        
        self.layer2 = self._make_layer(block_func=block, dim_in=64, 
                                       dim_out=128, layer_id=2, 
                                       num_blocks=layers[1], 
                                       is_decomposed=True,
                                       freeze_bn=freeze_bn)
        
        self.layer3 = self._make_layer(block_func=block, dim_in=128, 
                                       dim_out=256, layer_id=3, 
                                       num_blocks=layers[2], 
                                       is_decomposed=True,
                                       freeze_bn=freeze_bn)
        
        self.layer4 = self._make_layer(block_func=block, dim_in=256, 
                                       dim_out=512, layer_id=4, 
                                       num_blocks=layers[3], 
                                       is_decomposed=True,
                                       freeze_bn=freeze_bn)
        
        self.avgpool = nn.AvgPool3d((final_temporal_kernel, final_spatial_kernel, final_spatial_kernel), stride=(1, 1, 1), padding=(0, 0, 0))
        self.fc = nn.Linear(512, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def _make_layer(self, block_func, dim_in, dim_out, layer_id, num_blocks, is_decomposed, freeze_bn):

        layers = []
        for idx in range(num_blocks):
            down_sampling = False
            if layer_id > 1 and idx == 0:
                down_sampling = True
            layers.append(block_func(dim_in=dim_in, dim_out=dim_out, down_sampling=down_sampling, is_decomposed=is_decomposed, freeze_bn=freeze_bn))
            dim_in = dim_out
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1_middle(x)
        x = self.bn1_middle(x)
        x = self.relu(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)      
        x = self.fc(x)

        return x
    
def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters
    
def obtain_arc(sample_duration, model_depth):
    if model_depth == 18:
        layers = (2, 2, 2, 2)
    elif model_depth == 34:
        layers = (3, 4, 6, 3)
        
    block = BasicBlock
    final_temporal_kernel = int(sample_duration / 8)    
    final_spatial_kernel = 7
    
    return layers, block, final_temporal_kernel, final_spatial_kernel

def create_model(**kwargs):
    model = ResNet(**kwargs)
    return model
