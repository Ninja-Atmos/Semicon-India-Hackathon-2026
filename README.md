# AI-Based Restoration of Degraded Images for Semiconductor Inspection
**i4c SEMICON India Hackathon 2026 - KLA Challenge**  
**Team Member:** Suvam Biswas | Animesh Kullu | Debjit Das  
**Institution:** Jalpaiguri Government Engineering College (JGEC)  

## 📌 Project Overview
This repository contains a high-performance, lightweight Machine Learning solution designed to restore degraded, noisy, and low-resolution grayscale semiconductor inspection images. The model reverses signal degradation caused by electron scattering (speckle/Gaussian noise) and performs 2x Super-Resolution to recover spatial details without introducing artificial structural ringing. 

Optimized to maximize SSIM, PSNR, and LPIPS metrics, this pipeline meets strict industrial edge-deployment constraints.

## 🚀 Key Architectural Innovations
1. **Ultra-Lightweight Footprint (< 3.12M Limit):** 
   Utilizes a 6-block Residual CNN with only **592,065 parameters**, keeping it exceptionally lightweight for sub-millisecond edge inference.
2. **2x Super-Resolution (PixelShuffle):** 
   Addresses spatial resolution reduction by expanding feature maps (e.g., 128x128 → 256x256) dynamically within the network tail.
3. **Dynamic Resolution Guard:** 
   Implements automatic `mode='replicate'` padding during the forward pass to prevent tensor mismatch errors when processing out-of-distribution or arbitrary wafer scan resolutions.
4. **Hybrid Loss Function (L1 + MS-SSIM Proxy):**
   Optimizes for both pixel-perfect intensity and structural boundary preservation, vital for EUV lithography defect detection.
5. **Self-Describing Checkpoints:** 
   Model hyperparameters are embedded directly into `.pth` weights to prevent configuration drift during deployment.

## ⚙️ Repository Structure
- `KLA_Restored_Submission.zip`: [Download Restored Outputs Here](https://drive.google.com/file/d/1ks5DLZ-wNKiuBJp89Jfhrq1WMs2BiqAM/view?usp=sharing)
- `semicon_restore.py`: The core unified script containing the model architecture, training loop, and inference logic.
- `kla_checkpoint.pth`: The fully trained model weights (download from release / drive).
- `requirements.txt`: Python environment dependencies.
