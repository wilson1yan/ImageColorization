import torch
import torch.nn as nn

class FusionColorizer(nn.Module):
    def __init__(self):
        super(FusionColorizer, self).__init__()
        self.low_level_net = LowLevelNet()
        self.mid_level_net = MidLevelNet()
        self.global_feat_net = GlobalFeaturesNet()
        self.class_net = ClassificationNet()
        self.fusion_net = FusionNet()
        self.color_net = ColorNet()
 
    def forward(self, input, input_scaled):
        out = self.low_level_net(input)
        out = self.mid_level_net(out)
        out_scaled = self.low_level_net(input_scaled)
        out_class, out_scaled = self.global_feat_net(out_scaled)
        class_scores = self.class_net(out_class)
        
        out_fusion = self.fusion_net(out, out_scaled)
        out_color = self.color_net(out_fusion)
        return out_color, class_scores
        
class LowLevelNet(nn.Module):
    def __init__(self):
        super(LowLevelNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
    
    def forward(self, input):
        out = F.relu(self.bn1(self.conv1(input)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        return out
    
class MidLevelNet(nn.Module):
    def __init__(self):
        super(MidLevelNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, input):
        out = F.relu(self.bn1(self.conv1(input)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class GlobalFeaturesNet(nn.Module):
    def __init__(self):
        super(GlobalFeaturesNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(7*7*512, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
    
    def forward(self, input):
        out = F.relu(self.bn1(self.conv1(input)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn5(self.fc1(out)))
        out_class = F.relu(self.bn6(self.fc2(out)))
        out_fusion = F.relu(self.bn7(self.fc3(out_class)))
        return out_class, out_fusion

class ClassificationNet(nn.Module):
    def __init__(self, n_classes):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, n_classes)
        self.bn2 = nn.BatchNorm1d(n_classes)

    def forward(self, input):
        out = F.relu(self.bn1(self.fc1(input)))
        out = F.log_softmax(self.bn2(self.fc2(out)))
        return out
    
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.fc = nn.Linear(512, 256)
        self.bn = nn.BatchNorm1d(256)
    
    def forward(self, out_mid, out_global):
        b, _, w, h = out_mid.size()
        out_global = out_global.unsqueeze(0).unsqueeze(0)
        out_global = out_global.repeat(1, 1, w, h)
        fusion = torch.cat((out_mid, out_global), 1)
        fusion = fusion.permute(2, 3, 0, 1).contiguous()
        fusion = fusion.view(-1, 512)
        fusion = self.bn(self.fc(fusion))
        fusion = fusion.view(w, h, b, 512)
        fusion = fusion.permute(2, 3, 0, 1).contiguous()
        return fusion
        
    
class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(2)
    
    def forward(self, input):
        out = self.upsample(input)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.upsample(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.upsample(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.sigmoid(self.bn5(self.conv5(out)))
        out = self.upsample(out)
        return out
        
