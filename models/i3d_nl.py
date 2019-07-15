import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    
    def __init__(self, dim_in, dim_out, stride, dim_inner, 
                 use_temp_conv=0, temp_stride=1, freeze_bn=False):
        
        super(Bottleneck, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride = stride
        self.dim_inner = dim_inner
        self.use_temp_conv = use_temp_conv
        self.temp_stride = temp_stride
        self.freeze_bn = freeze_bn
        
        # 1x1 layer
        self.conv1 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1 + use_temp_conv * 2, 1, 1),
            stride=(temp_stride, 1, 1),
            padding=(use_temp_conv, 0, 0),
            bias=False)
        self.bn1 = nn.BatchNorm3d(dim_inner, track_running_stats=(not freeze_bn))
        
        # 3x3 layer
        self.conv2 = nn.Conv3d(
            dim_inner,
            dim_inner,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm3d(dim_inner, track_running_stats=(not freeze_bn))
        
        # 1x1 layer
        self.conv3 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False)
        self.bn3 = nn.BatchNorm3d(dim_out, track_running_stats=(not freeze_bn))
        
        self.relu = nn.ReLU(inplace=True)
        
        if not (self.dim_in == self.dim_out and self.temp_stride == 1 and self.stride == 1): 
            self.conv_sc = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=(temp_stride, stride, stride),
                padding=(0, 0, 0),
                bias=False)
            self.bn_sc = nn.BatchNorm3d(dim_out, track_running_stats=(not freeze_bn))
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if not (self.dim_in == self.dim_out and self.temp_stride == 1 and self.stride == 1):
            residual = self.conv_sc(residual)
            residual = self.bn_sc(residual)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 model_depth,
                 sample_duration,
                 freeze_bn=False,
                 num_classes=400):
        
        dim_inner=64
        layers, block, use_temp_convs_set, temp_strides_set, pool_stride = obtain_arc(sample_duration, model_depth)
        
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(1 + use_temp_convs_set[0][0] * 2, 7, 7),
            stride=(temp_strides_set[0][0], 2, 2),
            padding=(use_temp_convs_set[0][0], 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=(not freeze_bn))
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.layer1 = self._make_layer(block_func=block, dim_in=64, 
                                       dim_out=256, dim_inner=dim_inner, 
                                       num_blocks=layers[0], stride=1,
                                       use_temp_convs=use_temp_convs_set[1],
                                       temp_strides=temp_strides_set[1],
                                       freeze_bn=freeze_bn)
        
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        
        self.layer2 = self._make_layer(block_func=block, dim_in=256, 
                                       dim_out=512, dim_inner=dim_inner * 2, 
                                       num_blocks=layers[1], stride=2,
                                       use_temp_convs=use_temp_convs_set[2],
                                       temp_strides=temp_strides_set[2],
                                       freeze_bn=freeze_bn)
        
        self.layer3 = self._make_layer(block_func=block, dim_in=512, 
                                       dim_out=1024, dim_inner=dim_inner * 4, 
                                       num_blocks=layers[2], stride=2,
                                       use_temp_convs=use_temp_convs_set[3],
                                       temp_strides=temp_strides_set[3],
                                       freeze_bn=freeze_bn)
        
        self.layer4 = self._make_layer(block_func=block, dim_in=1024, 
                                       dim_out=2048, dim_inner=dim_inner * 8, 
                                       num_blocks=layers[3], stride=2,
                                       use_temp_convs=use_temp_convs_set[4],
                                       temp_strides=temp_strides_set[4],
                                       freeze_bn=freeze_bn)
        
        self.avgpool = nn.AvgPool3d((pool_stride, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0))
        self.dropout = nn.Dropout3d(p=0.5, inplace=True)
        self.fc = nn.Linear(2048, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def _make_layer(self, block_func, dim_in, dim_out, dim_inner, num_blocks, stride, use_temp_convs, temp_strides, freeze_bn):
        
        if use_temp_convs is None:
            use_temp_convs = np.zeros(num_blocks).astype(int)
        if temp_strides is None:
            temp_strides = np.ones(num_blocks).astype(int)

        if len(use_temp_convs) < num_blocks:
            for _ in range(num_blocks - len(use_temp_convs)):
                use_temp_convs.append(0)
                temp_strides.append(1)
        
        layers = []
        for idx in range(num_blocks):
            block_stride = 2 if (idx == 0 and stride == 2) else 1
            
            layers.append(block_func(dim_in=dim_in, dim_out=dim_out, stride=block_stride,
                                     dim_inner=dim_inner, use_temp_conv=use_temp_convs[idx],
                                     temp_stride=temp_strides[idx], freeze_bn=freeze_bn))
            dim_in = dim_out
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if self.training:
            x = self.dropout(x)
            
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
    if model_depth == 50:
        layers = (3, 4, 6, 3)
        use_temp_convs_1 = [2]
        temp_strides_1   = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2   = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3   = [1, 1, 1, 1]
        use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
        temp_strides_4   = [1, 1, 1, 1, 1, 1]
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5   = [1, 1, 1]
    
    block = Bottleneck
    pool_stride = int(sample_duration / 2)

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set   = [temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4, temp_strides_5]

    return layers, block, use_temp_convs_set, temp_strides_set, pool_stride

def create_model(**kwargs):
    model = ResNet(**kwargs)
    return model
