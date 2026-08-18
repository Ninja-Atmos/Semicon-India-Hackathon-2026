import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Model Architecture
# ==========================================
class LightweightRestorationCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=6, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            ) for _ in range(num_blocks)
        ])

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        res = self.head(x)
        for block in self.body:
            res = block(res) + res

        res = self.upsample(res)
        
        # Using bicubic interpolation for sharper foundational upscaling
        x_upscaled = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        out = self.tail(res) + x_upscaled

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h * self.upscale_factor, :w * self.upscale_factor]

        return out

# ==========================================
# 2. Main Execution Logic
# ==========================================
def main():
    # Setup command-line arguments
    parser = argparse.ArgumentParser(description="KLA Problem Statement: Image Restoration")
    parser.add_argument("input_dir", type=str, help="Path to input directory containing noisy .npy files")
    parser.add_argument("output_dir", type=str, help="Path to output directory for restored .npy files")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Failsafe: Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model Offline
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'kla_checkpoint.pth')
    model = LightweightRestorationCNN().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model weights from {model_path}. \nDetails: {e}")
        return
        
    model.eval()

    # Process all .npy files in the input directory
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    print(f"Found {len(input_files)} files. Starting restoration...")

    for file_name in input_files:
        in_path = os.path.join(input_dir, file_name)
        out_path = os.path.join(output_dir, file_name)

        # Load data safely
        noisy_arr = np.load(in_path).astype(np.float32)
        
        # Normalize to [0.0, 1.0] if not already normalized
        if noisy_arr.max() > 1.0:
            noisy_arr = noisy_arr / 255.0

        # Tensor formatting
        input_tensor = torch.from_numpy(noisy_arr).to(device)
        if input_tensor.ndim == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Run Inference
        with torch.no_grad():
            restored_tensor = model(input_tensor)

        # Clean Output: Squeeze to (H, W) format
        restored_arr = restored_tensor.squeeze().cpu().numpy()
        
        # Strict Compliance: Replace any accidental NaNs or Infs and clip to [0,1] bounds
        restored_arr = np.nan_to_num(restored_arr, nan=0.0, posinf=1.0, neginf=0.0)
        restored_arr = np.clip(restored_arr, 0.0, 1.0)
        
        # Save output with identical filename
        np.save(out_path, restored_arr)

    print("✅ All files processed and successfully saved.")

if __name__ == "__main__":
    main()
