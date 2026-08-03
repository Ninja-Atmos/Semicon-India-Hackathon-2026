"""
AI-Based Restoration of Degraded Images for Semiconductor Inspection
i4c SEMICON India Hackathon 2026 - KLA Challenge
Author: Suvam Biswas (Jalpaiguri Government Engineering College)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ==========================================
# 1. Super-Resolution Architecture (592K Params)
# ==========================================
class LightweightRestorationCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=6, upscale_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor

        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            ) for _ in range(num_blocks)
        ])
        
        # 2x Upsampling Block for Super Resolution
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(0)
        _, _, h, w = x.shape
        
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        res = self.head(x)
        for block in self.body:
            res = block(res) + res
            
        res = self.upsample(res)
        x_upscaled = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        out = self.tail(res) + x_upscaled  

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h * self.upscale_factor, :w * self.upscale_factor]
        return out

# ==========================================
# 2. Hybrid Loss Function
# ==========================================
class RestorationLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_approx = torch.mean(torch.abs(F.avg_pool2d(pred, 3, 1, 1) - F.avg_pool2d(target, 3, 1, 1)))
        return self.l1_weight * loss_l1 + self.ssim_weight * loss_approx

# ==========================================
# 3. Dataset Loader
# ==========================================
class KLADataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.npy')])
        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.npy')])
        common_files = sorted(list(set(noisy_files).intersection(set(clean_files))))
        
        self.noisy_paths = [os.path.join(noisy_dir, f) for f in common_files]
        self.clean_paths = [os.path.join(clean_dir, f) for f in common_files]

    def __len__(self): return len(self.noisy_paths)
        
    def __getitem__(self, idx):
        noisy_tensor = torch.from_numpy(np.load(self.noisy_paths[idx]).copy()).float()
        clean_tensor = torch.from_numpy(np.load(self.clean_paths[idx]).copy()).float()
        if noisy_tensor.ndim == 2:
            noisy_tensor, clean_tensor = noisy_tensor.unsqueeze(0), clean_tensor.unsqueeze(0)
        return noisy_tensor, clean_tensor

# ==========================================
# 4. Core Execution Functions
# ==========================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KLADataset(args.noisy_dir, args.clean_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = LightweightRestorationCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = RestorationLoss()

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} (Limit 3.12M)")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {'in_channels': 1, 'out_channels': 1, 'num_features': 64, 'num_blocks': 6, 'upscale_factor': 2}
    }, args.save_path)
    print(f"Model saved to {args.save_path}")

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = LightweightRestorationCNN(**checkpoint['hyperparameters']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
    for f in test_files:
        input_tensor = torch.from_numpy(np.load(os.path.join(args.input_dir, f)).copy()).float().to(device)
        if input_tensor.ndim == 2: input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        np.save(os.path.join(args.output_dir, f), output_tensor.squeeze().cpu().numpy())
    print(f"Processed {len(test_files)} files to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLA Hackathon Image Restoration")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--noisy_dir", required=True, help="Path to noisy .npy files")
    parser_train.add_argument("--clean_dir", required=True, help="Path to ground truth .npy files")
    parser_train.add_argument("--epochs", type=int, default=100)
    parser_train.add_argument("--batch_size", type=int, default=16)
    parser_train.add_argument("--lr", type=float, default=1e-4)
    parser_train.add_argument("--save_path", default="kla_checkpoint.pth")

    parser_infer = subparsers.add_parser("infer")
    parser_infer.add_argument("--input_dir", required=True, help="Path to test .npy files")
    parser_infer.add_argument("--output_dir", required=True, help="Path to save restored .npy files")
    parser_infer.add_argument("--checkpoint", default="kla_checkpoint.pth")

    args = parser.parse_args()
    if args.mode == "train": train(args)
    elif args.mode == "infer": infer(args)
