
import os
import time
import datetime
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imagingcontrol4 as ic4

# --- Save Path ---
SAVE_DIR = Path(r"C:/Users/ZBook/Documents/Saved_detection")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
BEST_THRESHOLD = xxx
HEATMAP_MIN = xxx
HEATMAP_MAX = xxx
ZOOM_OPTIONS = {"1": 40, "2": 50, "3": 60}

# --- Preprocessing ---
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=2):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

def apply_clahe_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def test_enhancement(image):
    return unsharp_mask(adjust_gamma(apply_clahe_color(image), 0.8))

# --- Camera Capture Function ---
def capture_image_from_camera(zoom_level):
   
    grabber = ic4.Grabber()
    devices = ic4.DeviceEnum.devices()
    if len(devices) == 0:
        raise RuntimeError("No camera found.")
    grabber.device_open(devices[0])

    g = grabber.device_property_map
    g.set_value(ic4.PropId.OFFSET_X, 0)
    g.set_value(ic4.PropId.OFFSET_Y, 0)
    g.set_value(ic4.PropId.WIDTH, 1024)
    g.set_value(ic4.PropId.HEIGHT, 1024)
    g.set_value(ic4.PropId.ZOOM, zoom_level)
    time.sleep(0.4)

    try:
        g.set_value(ic4.PropId.EXPOSURE_TIME, 100000)
        g.set_value(ic4.PropId.FOCUS, 555)
        g.set_value(ic4.PropId.GAMMA, 1)
        g.set_value(ic4.PropId.IRIS, 786)
        g.set_value(ic4.PropId.CONTRAST, 0)
        g.set_value(ic4.PropId.GAIN, 0)
        g.set_value(ic4.PropId.SHARPNESS, 0)
        g.set_value(ic4.PropId.EXPOSURE_AUTO_REFERENCE, 0)
    except ic4.IC4Exception as ex:
        print(f"Camera config error: {ex.message}")

    sink = ic4.SnapSink()
    grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)

    try:
        image = sink.snap_single(3000)
        np_img = image.numpy_copy()
    except ic4.IC4Exception as ex:
        print(f"Error capturing image: {ex.message}")
        np_img = None

    grabber.stream_stop()
    grabber.device_close()
    return np_img

# --- Model Architecture ---
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.features = []

        def get_features(module, input, output):
            self.features.append(output)

        self.model.features[1].register_forward_hook(get_features)
        self.model.features[2][-1].register_forward_hook(get_features)
        self.model.features[3][-1].register_forward_hook(get_features)
        self.model.features[4][-1].register_forward_hook(get_features)
        self.model.features[5][-1].register_forward_hook(get_features)
        self.model.features[6][-1].register_forward_hook(get_features)

        self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

    def forward(self, x):
        self.features = []
        with torch.no_grad():
            _ = self.model(x)
        upsampled = [F.interpolate(fm, size=(170, 170), mode='bilinear', align_corners=False) for fm in self.features]
        pooled = [self.pool(fm) for fm in upsampled]
        return torch.cat(pooled, dim=1)

# --- Autoencoder ---
class FeatCAE(nn.Module):
    def __init__(self, in_channels=832, latent_dim=260, is_bn=True):
        super().__init__()
        mid1 = (in_channels + 2 * latent_dim) // 2

        enc_layers = [
            nn.Conv2d(in_channels, mid1, 1, 1, 0),
            nn.BatchNorm2d(mid1) if is_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(mid1, 2 * latent_dim, 1, 1, 0),
            nn.BatchNorm2d(2 * latent_dim) if is_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(2 * latent_dim, latent_dim, 1, 1, 0)
        ]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = [
            nn.Conv2d(latent_dim, 2 * latent_dim, 1, 1, 0),
            nn.BatchNorm2d(2 * latent_dim) if is_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(2 * latent_dim, mid1, 1, 1, 0),
            nn.BatchNorm2d(mid1) if is_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(mid1, in_channels, 1, 1, 0)
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

def decision_function(segm_map):
    return torch.stack([torch.sort(m.flatten(), descending=True)[0][:20].mean() for m in segm_map])

# --- Inference ---
def run_inference(image_pil, zoom, model, backbone, transform, save_dir):
    start = time.time()
    x = transform(image_pil).unsqueeze(0).cuda()
    with torch.no_grad():
        features = backbone(x)
        recon = model(features)

    segm_map = ((features - recon) ** 2).mean(1)[:, 20:-20, 20:-20]
    segm_map_full = ((features - recon) ** 2).mean(1)
    score_val = decision_function(segm_map)[0].item()

    if score_val < 0.028:
        cls = "OK"; color = "green"
    elif score_val <= 0.0305:
        cls = "CLOSE"; color = "yellow"
    else:
        cls = "NOK"; color = "red"

    np_img = x.squeeze().permute(1, 2, 0).cpu().numpy()
    heat_map = segm_map_full.squeeze().cpu().numpy()
    heat_map_resized = cv2.resize(heat_map, (np_img.shape[1], np_img.shape[0]))
    elapsed = time.time() - start

    plt.switch_backend('TkAgg')
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(np_img); plt.title('Input Image', fontsize=18); plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(np_img); plt.imshow(heat_map_resized, cmap='jet', alpha=0.4, vmin=HEATMAP_MIN, vmax=HEATMAP_MAX)
    plt.axis('off'); plt.title('Heatmap Overlay', fontsize=18)
    plt.subplot(1, 3, 3)
    plt.imshow(np_img)
    overlay = np.zeros_like(np_img)
    if cls == "OK": overlay[:, :, 1] = 1.0
    elif cls == "CLOSE": overlay[:, :, :2] = 1.0
    else: overlay[:, :, 0] = 1.0
    plt.imshow(overlay, alpha=0.2)
    plt.axis('off')
    ax = plt.gca()
    ax.text(0.95, 0.05, cls, transform=ax.transAxes, ha='right', va='bottom', fontsize=100, color=color, fontweight='bold')
    plt.title(f'Anomaly Score: {score_val/BEST_THRESHOLD:.4f} || Time: {elapsed:.3f} s', fontsize=18, fontweight='bold')
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"zoom_{zoom}_timestamp_{timestamp}_{cls}.png"


    if cls == "OK":
        subfolder = "OK"
    elif cls == "CLOSE":
        subfolder = "CLOSE"
    else:
        subfolder = "NOK"

    save_path = Path(save_dir) / subfolder / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)


    plt.savefig(save_path, bbox_inches='tight') 
    print(f"Saved to {save_path}")

    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')
    except:
        pass

    plt.show()
    return cls 

# --- Helper function for next higher zoom ---
def get_next_zoom_key(current_key, zoom_keys):
    idx = zoom_keys.index(current_key)
    if idx < len(zoom_keys) - 1:
        return zoom_keys[idx + 1]
    return None


# --- Main Loop ---

def main():
    ic4.Library.init()
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    backbone = EfficientNetFeatureExtractor().cuda().eval()
    model = FeatCAE().cuda().eval()
    model.load_state_dict(torch.load(r"C:/EffNet_170.pth"))

    while True:
        key = input("Choose zoom: 1=40, 2=50, 3=60, q=quit\n> ")
        if key == "q":
            break
        if key not in ZOOM_OPTIONS:
            continue

        zoom = ZOOM_OPTIONS[key]
        raw = capture_image_from_camera(zoom)
        if raw is None:
            print("Failed to capture image.")
            continue

        if len(raw.shape) == 2 or raw.shape[2] == 1:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        enhanced = test_enhancement(raw)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        cls = run_inference(
        	image_pil=pil_img,
        	zoom=zoom,
        	model=model,
        	backbone=backbone,
        	transform=transform,
        	save_dir=SAVE_DIR
         )

        if cls == "CLOSE":
                next_key = get_next_zoom_key(key, zoom_keys)
                if next_key is not None:
                	print(f"Result is CLOSE. Zooming in further to {ZOOM_OPTIONS[next_key]}.")
                	key = next_key
                	continue
        break
       
if __name__ == "__main__":
    main()
