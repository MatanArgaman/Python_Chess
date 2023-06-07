import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BatchNorm2dRon(nn.BatchNorm2d):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.isnan(torch.sum(input)):
            #This is very very weird, we are passing the input.... if the output of the whole netwrk will be nan you should skip the backward pass
            print(f"Got NaN in input.")
            return input

        return super().forward(input)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes,eps=1e-05)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes,eps=1e-05)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,skip_last=False,normalize=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.skip_last = skip_last
        self.normalize = normalize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)

        x = x.view(x.size(0), -1)
        if self.normalize:
            x = nn.functional.normalize(x, p = 2, dim = 1, eps = 1e-6)
        if not self.skip_last:
            x = self.fc(x)
            x = nn.functional.normalize(x, p=2, dim=1, eps=1e-6)

        #return x
        return x, layer3 # to allow head running on top too!!




def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class MyResNet34(nn.Module):
    def __init__(self):
        super(MyResNet34, self).__init__()

        self.model = resnet34(pretrained=True, skip_last=True, normalize=True)
        self.fc1 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.normalize(x, p=2, dim=1, eps=1e-6)


        return x


class MyResNet34_for_Quality(nn.Module):
    def __init__(self):
        super(MyResNet34_for_Quality, self).__init__()

        self.model = resnet34(pretrained=True, skip_last=True, normalize=True)
        self.fc1 = nn.Linear(512, 1)
        # self.fc2 = nn.Linear(512, 1)
        self.Dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))

        # x = x.view(x.size(0), -1)
        # x = nn.functional.dropout(nn.functional.relu(self.fc1(x)), p=0.5)
        # x = torch.sigmoid(self.fc2(x))


        return x


class MyResNet50_for_Wrong(nn.Module):
    def __init__(self):
        super(MyResNet50_for_Wrong, self).__init__()

        self.model = resnet50(pretrained=True, skip_last=True, normalize=True)
        self.fc1 = nn.Linear(2048, 32) # Need to fix this
        self.fc2 = nn.Linear(32, 1)
        # self.fc3 = nn.Linear(1, 1)
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        # x = self.dropout(F.relu(self.fc1(x)))
        x = nn.functional.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc2(x))
        # x = torch.sigmoid(self.fc2(x))

        return x



class MyResNet34_with_GRU(nn.Module):
    def __init__(self):
        super(MyResNet34_with_GRU, self).__init__()

        self.model = resnet34(pretrained=True, skip_last=True, normalize=False)
        self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 1)
        # self.Dropout = nn.Dropout(p=0.5)

    def forward(self, input, labels, sequence_length):

        size = input.size()
        # x = x.reshape(size[0] * size[1],size[2], size[3], size[4])
        #
        input = input.reshape(size[0]*size[1], size[2], size[3], size[4])
        input_part = input[:,:,:20,:20]
        ix = input_part.sum(1).sum(1).sum(1) != 0
        input_wo_padding = input[ix, :, :, :]      #Size: [B x N - zero_padding, 3, 224, 224]


        #[B, S, 3, 244, 244]

        # input_wo_padding = torch.zeros(sum(sequence_length), size[2], size[3], size[4]).to(device="cuda")      #Size: [B x N - zero_padding, 3, 224, 224]
        # for i in range(size[0]):
        #     input_wo_padding[sum(sequence_length[:i]):sum(sequence_length[:i+1]),:,:,:] = input[i,0:sequence_length[i],:,:,:]

        x = self.model.conv1(input_wo_padding)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)    #Size: [B x N - zero_padding, 512, 1, 1]

        x = torch.squeeze(x)

        #Sanity check
        if len(x.size()) != 2:
            raise Exception("Output of CNN is not 2-dimensional, something went wrong here.")

        embedding = torch.zeros((size[0] * size[1], 512)).to(device="cuda", non_blocking=True)  #Size: [B, N, 512]
        embedding[ix,:] = x
        embedding = embedding.reshape(size[0], size[1], 512)
        # for i in range(size[0]):
        #     embedding[i,0:sequence_length[i],:] = x[sum(sequence_length[:i]):sum(sequence_length[:i+1]),:]
        # x = x.reshape(size[0],size[1],512,1,1)
        # embedding = torch.squeeze(x)   #Size: [B , N, 512]

        # print('----------------------------------------------------------')
        # print(embedding.size(), '           expected: [B , N, 512]')
        self.gru.flatten_parameters()
        output_gru, hidden = self.gru(embedding) #Size: [B , N, 1024]
        output_gru = output_gru.reshape(size[0] * size[1], 1024) #Size: [B x N, 1024]
        # print(output_gru.size(), '           expected: [B x N, 1024]' )

        out_fc = self.fc1(output_gru)
        out_fc[ix==False] = 0
        out_fc = out_fc.reshape(size[0], size[1], 512) #Size: [B , N, 512]

        #set weights to zero for inputs which are pure zero padding
        # for i in range(size[0]):
        #     out_fc[i, sequence_length[i]:,:] = 0

        # print(out_fc.size(), '           expected: [B , N, 512]')

        # weights = nn.functional.softmax(out_fc, dim=2) #[B , N, 512]
        weights = nn.functional.normalize(out_fc, p=2, dim=1, eps=1e-8)

        # print(weights.size(), '           expected: [B , N, 512]')

        # attention_vec = torch.zeros((size[0], size[1])).to(device="cuda") #Size[B,N]
        #
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         attention_vec[i,j] = torch.dot(embedding[i,j,:], weights[i,j,:])

        attention_vec = torch.sum(embedding * weights, dim=2) #Size[B,N]

        attention_vec = nn.functional.softmax(attention_vec, dim=1)

        attention_vec = torch.unsqueeze(attention_vec, dim=2) #Size[B,N,1]

        # print(attention_vec.size(), '           expected: [B , N, 1]')

        output_final = torch.sum(attention_vec * embedding, dim = 1) #Size[B,512]
        output_final = nn.functional.normalize(output_final, p=2, dim=1, eps=1e-8)

        # print(output_final.size(), '           expected: [B , 512]')












        # output_final = nn.functional.normalize(output[:,size[1]-1,:], p=2, dim=1, eps=1e-6)

        return output_final



class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()

        self.model = resnet18(pretrained=True, skip_last=True, normalize=True)
        self.fc1 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.normalize(x, p=2, dim=1, eps=1e-6)


        return x


class MyResNet18_for_Quality(nn.Module):
    def __init__(self):
        super(MyResNet18_for_Quality, self).__init__()

        self.model = resnet18(pretrained=True, skip_last=True, normalize=True)
        # self.fc = nn.Linear(512,256)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        # x = nn.functional.dropout(nn.functional.relu(self.fc(x)), p=0.5)
        x = torch.sigmoid(self.fc1(x))

        # x = torch.sigmoid(self.fc1(x))


        return x


class MyResNet18ChildSeat(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True, skip_last=True, normalize=True)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))

        return x


class resnet50_with_arcface_head(nn.Module):

    def __init__(self, feature_size = 512, nr_classes = 1000):
        super(resnet50_with_arcface_head, self).__init__()

        self.model = resnet50(pretrained=True, skip_last=False, normalize=False)
        self.model.fc = nn.Linear(4*512, feature_size)
        self.arcface_matrix = nn.Linear(feature_size, nr_classes, bias=False)

    def set_weight(self, weight):
        self.arcface_matrix.weight = nn.Parameter(nn.functional.normalize(weight))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ this forward also returns internal layer results, useful for running a head of top of it"""
        x, layer3= self.model.forward(inputs)
        cosine = self.arcface_matrix(x)
        return cosine

class resnet_with_arcface_head(nn.Module):

    def __init__(self, feature_size = 512, nr_classes = 1000, model_type='resnet50'):
        super(resnet_with_arcface_head, self).__init__()

        if model_type=='resnet101':
            self.model = resnet101(pretrained=True, skip_last=False, normalize=False)
        else:
            self.model = resnet50(pretrained=True, skip_last=False, normalize=False)
        self.model.fc = nn.Linear(4*512, feature_size)
        self.arcface_matrix = nn.Linear(feature_size, nr_classes, bias=False)

    def set_weight(self, weight):
        self.arcface_matrix.weight = nn.Parameter(nn.functional.normalize(weight))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, layer3 = self.model.forward(inputs)
        cosine = self.arcface_matrix(x)
        return cosine

# for multi_obj  649,571 parameters, compared to 25K-30K of whole resnet


class MyFlatten(nn.Module):
    """ use it instead of Flatten as TensorRT dont' know how to convert it
    """
    def __init__(self,):
        super().__init__()

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = input.view(input.size(0),-1)
        return out



def create_conv_head_as_list(in_features, out_classes, spatial_size, mid_conv_channels_list=[32, 64, 128],
                             mid_linear_out=32, dropout=0.05, verbose=False):
    head = nn.ModuleList([])  # [PrintLayer(verbose)]

    for out_channels in mid_conv_channels_list:
        head.extend([nn.Conv2d(in_channels=in_features, kernel_size=3, out_channels=out_channels),
                     nn.Dropout2d(dropout / 2),
                     nn.ReLU(),
                     # PrintLayer(verbose)
                     ])
        in_features = out_channels

    head.extend([  # PrintLayer(verbose,'[before flatten]'),
        MyFlatten(), #nn.Flatten(), #unsupported
        # PrintLayer(verbose,'[after flatten]'),
        nn.Linear(in_features * (spatial_size - 2 * len(mid_conv_channels_list)) ** 2, mid_linear_out),
        # PrintLayer(verbose),
        nn.ReLU(),
        # PrintLayer(),
        nn.Dropout(dropout),
        nn.Linear(mid_linear_out, out_classes)
        # PrintLayer(verbose)])
    ])

    head = nn.Sequential(*head)
    head.train()
    return head

def create_alert_head(dropout=0.01844):
    # am I at least one , am I at least 2
    return create_conv_head_as_list(1024, out_classes=2,spatial_size=14, mid_conv_channels_list=[8*4,16*4,32*4],
                                                mid_linear_out=32, dropout=dropout, verbose=False)


class resnet50_hydra(nn.Module):
    """ DEPRECATED. use resnet_hydra instead
    arcface head AND alert head from layer3"""

    def __init__(self, feature_size=512, nr_classes=1000):
        super(resnet50_hydra, self).__init__()

        self.model = resnet50(pretrained=True, skip_last=False, normalize=False)
        self.model.fc = nn.Linear(4 * 512, feature_size)
        self.arcface_matrix = nn.Linear(feature_size, nr_classes, bias=False)
        self.alert_head = create_alert_head()

    def set_weight(self, weight):
        self.arcface_matrix.weight = nn.Parameter(nn.functional.normalize(weight))

    # def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    #     x = self.model.forward(inputs)
    #     cosine = self.arcface_matrix(x)
    #     return cosine

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ this forward also returns internal layer results, useful for running a head of top of it"""
        x, layer3 = self.model.forward(inputs)
        cosine = self.arcface_matrix(x)
        counts = torch.sigmoid(self.alert_head(layer3))
        return cosine, counts


class resnet_hydra(nn.Module):
    """ arcface head AND alert head from layer3"""

    def __init__(self, feature_size = 512, nr_classes = 1000, model_type='resnet50'):
        super(resnet_hydra, self).__init__()

        if model_type == 'resnet101':
            self.model = resnet101(pretrained=True, skip_last=False, normalize=False)
        else:
            self.model = resnet50(pretrained=True, skip_last=False, normalize=False)
        self.model.fc = nn.Linear(4*512, feature_size)
        self.arcface_matrix = nn.Linear(feature_size, nr_classes, bias=False)
        self.alert_head = create_alert_head()

    def set_weight(self, weight):
        self.arcface_matrix.weight = nn.Parameter(nn.functional.normalize(weight))

    # def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    #     x = self.model.forward(inputs)
    #     cosine = self.arcface_matrix(x)
    #     return cosine

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ this forward also returns internal layer results, useful for running a head of top of it"""
        x, layer3 = self.model.forward(inputs)
        cosine = self.arcface_matrix(x)
        counts = torch.sigmoid(self.alert_head(layer3))
        return cosine, counts