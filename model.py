import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

class FusionModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(512 * 4, 512 * 4),
            nn.Sigmoid()
        )
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.fc = nn.Sequential(
            nn.Linear(4 * 512, 256),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x1, x2, x3, x4):
        def get_features(x):
            return self.feature_extractor(x).flatten(1)
            
        f1 = get_features(x1)
        f2 = get_features(x2)
        f3 = get_features(x3)
        f4 = get_features(x4)
        
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        weights = self.attention(fused)  
        weighted_fused = fused * weights
        return self.fc(weighted_fused)

if __name__ == '__main__':
    model = FusionModel()
    summary(model, input_size=[(1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224)])
    # summary(model, input_size=[(1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224)], device='cpu') 