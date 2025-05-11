import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights

class efficientnet_feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()

        weights = EfficientNet_B6_Weights.DEFAULT
        self.preprocess = weights.transforms()      
        self.model = efficientnet_b6(weights=weights)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        def get_features(module, input, output):
            self.features.append(output)
        for idx in range(1, 6):
            self.model.features[idx][-1].register_forward_hook(get_features)

        self.input_size = 512
        self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

    def forward(self, input):
        self.features = []
        with torch.no_grad():
            _ = self.model(input)
        agg = []
        for fmap in self.features:
            fmap_up = F.interpolate(fmap,
                                    size=(self.input_size, self.input_size),
                                    mode='bilinear')
            fmap_agg = self.pool(fmap_up)
            agg.append(fmap_agg)
        return torch.cat(agg, dim=1)

def compute_latent_dim(feature_matrix, threshold=0.90):
    pca = PCA(n_components=feature_matrix.shape[1])
    pca.fit(feature_matrix)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumvar >= threshold) + 1)


if __name__ == '__main__':
    feat_ext = efficientnet_feature_extractor()
    feat_ext.eval()

    image_folder = 'PATH'
    image_paths = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder)][:100]

    feats_list = []
    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            inp = feat_ext.preprocess(img).unsqueeze(0)  
            fmap = feat_ext(inp)                        
            arr = (fmap.cpu()
                       .numpy()
                       .transpose(0,2,3,1)
                       .reshape(-1, fmap.size(1)))   
            feats_list.append(arr)

    X = np.vstack(feats_list)
    print("Running PCA on feature matrix of shape:", X.shape)
    latent_dim = compute_latent_dim(X, threshold=0.90)
    print("→ Selected latent_dim =", latent_dim)
