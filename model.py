# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class OcularAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        

        self._init_backbones()
        
 
        self.eye_feat_dim = 2048  # ResNet50
        self.pupil_feat_dim = 1280  # EfficientNet-b0
        self.proj_dim = 512
        
 
        self.eye_proj = nn.Sequential(
            nn.Linear(self.eye_feat_dim, self.proj_dim),
            nn.ReLU(inplace=True)
        )
        self.pupil_proj = nn.Sequential(
            nn.Linear(self.pupil_feat_dim, self.proj_dim),
            nn.ReLU(inplace=True)
        )
        

        self.fusion = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=2
        )
        

        self.regressor = nn.Sequential(
            nn.Linear(self.proj_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def _init_backbones(self):

        resnet = torchvision.models.resnet50(pretrained=False)
        self.eye_extractor = nn.Sequential(*list(resnet.children())[:-1])
        

        effnet = torchvision.models.efficientnet_b0(pretrained=False)
        self.pupil_extractor = nn.Sequential(
            effnet.features,
            effnet.avgpool,
            nn.Flatten()
        )

    def _extract_features(self, imgs, extractor):

        batch_size, num_imgs = imgs.shape[:2]
        

        combined = imgs.view(-1, *imgs.shape[2:])  # [batch*num_imgs, C, H, W]
        features = extractor(combined)  # [batch*num_imgs, feat_dim]
        

        return features.view(batch_size, num_imgs, -1)  # [batch, num_imgs, feat_dim]

    def forward(self, eye_imgs, pupil_imgs):

        eye_feats = self._extract_features(eye_imgs, self.eye_extractor)  # [B,2,2048]
        pupil_feats = self._extract_features(pupil_imgs, self.pupil_extractor)  # [B,2,1280]
        

        eye_proj = self.eye_proj(eye_feats)  # [B,2,512]
        pupil_proj = self.pupil_proj(pupil_feats)  # [B,2,512]
        

        all_feats = torch.cat([eye_proj, pupil_proj], dim=1)  # [B,4,512]
        

        fused = self.fusion(all_feats.permute(1,0,2))  # [4, B, 512]
        

        output = self.regressor(fused.mean(dim=0))  # [B, 1]
        return output.squeeze(-1)  # [B]


if __name__ == "__main__":

    model = OcularAgeModel()
    

    dummy_eyes = torch.randn(2, 2, 3, 224, 224)  # batch_size=2
    dummy_pupils = torch.randn(2, 2, 3, 224, 224)
    

    pred_ages = model(dummy_eyes, dummy_pupils)
    print(f"D:{pred_ages.shape}")  
