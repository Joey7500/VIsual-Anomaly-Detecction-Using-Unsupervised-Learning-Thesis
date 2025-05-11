import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights

class efficientnet_feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        def get_features(module, input, output):
            self.features.append(output)
        self.model.features[1].register_forward_hook(get_features)
        self.model.features[2][-1].register_forward_hook(get_features)
        self.model.features[3][-1].register_forward_hook(get_features)
        self.model.features[4][-1].register_forward_hook(get_features)
        self.model.features[5][-1].register_forward_hook(get_features)
        self.model.features[6][-1].register_forward_hook(get_features)
        
        self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

    def forward(self, input):
        self.features = []  # Reset feature container.
        with torch.no_grad():
            _ = self.model(input)
        self.feature_maps = self.features
        agg_features = []
        for fmap in self.features:
            fmap_up = F.interpolate(fmap, size=(self.input_size, self.input_size),
                                    mode='bilinear', align_corners=False)
            fmap_agg = self.pool(fmap_up)
            agg_features.append(fmap_agg)
        merged = torch.cat(agg_features, dim=1)
        return merged 

if __name__ == '__main__':
    extractor = efficientnet_feature_extractor()
    x = torch.randn(1, 3, 512, 512)
    out = extractor(x)
    print("Concat_out:", out.shape)
