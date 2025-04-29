import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# 1. Define Dataset
class SIDD_DenoiseDataset(Dataset):
    def __init__(self, root_dir, split="val"):
        self.data_dir = os.path.join(root_dir, "SIDD_Small_sRGB_Only", "Data")
        self.folder_names = sorted(os.listdir(self.data_dir))
        self.folder_names = self.folder_names[140:]  # Validation set

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        folder = self.folder_names[idx]
        folder_path = os.path.join(self.data_dir, folder)
        files = os.listdir(folder_path)
        noisy_file = [f for f in files if f.startswith("NOISY")][0]
        gt_file = [f for f in files if f.startswith("GT")][0]
        noisy_img = Image.open(os.path.join(folder_path, noisy_file)).convert("RGB")
        gt_img = Image.open(os.path.join(folder_path, gt_file)).convert("RGB")
        noisy_img = self.transform(noisy_img)
        gt_img = self.transform(gt_img)
        return noisy_img, gt_img

# 2. Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def conv_block(in_channels, out_channels, down=True, use_dropout=False):
            layers = []
            if down:
                layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
            else:
                layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU() if not down else nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        self.down1 = conv_block(3, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)

        self.up1 = conv_block(512, 256, down=False, use_dropout=True)
        self.up2 = conv_block(512, 128, down=False, use_dropout=True)
        self.up3 = conv_block(256, 64, down=False, use_dropout=True)

        self.final_up = nn.ConvTranspose2d(128, 3, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))

        return self.tanh(self.final_up(torch.cat([u3, d1], 1)))

# 3. Evaluate
def evaluate():
    dataset_path = "."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = SIDD_DenoiseDataset(dataset_path, split="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    generator = Generator().to(device)
    generator.load_state_dict(torch.load("saved_models/generator_epoch20.pth", map_location=device))
    generator.eval()

    os.makedirs("test_outputs", exist_ok=True)
    os.makedirs("ground_truth_outputs", exist_ok=True)

    psnr_scores = []
    ssim_scores = []

    for idx, (noisy_img, clean_img) in enumerate(val_loader):
        noisy_img = noisy_img.to(device)
        clean_img = clean_img.to(device)

        with torch.no_grad():
            fake_clean = generator(noisy_img)

        # Unnormalize [-1,1] → [0,1]
        fake_clean_vis = (fake_clean * 0.5) + 0.5
        clean_img_vis = (clean_img * 0.5) + 0.5

        # Resize (just to be safe, although all your data is now (256,256))
        fake_clean_vis = TF.resize(fake_clean_vis, [256, 256])
        clean_img_vis = TF.resize(clean_img_vis, [256, 256])

        # Save images
        save_image(fake_clean_vis, f"test_outputs/fake_{idx+1:03d}.png")
        save_image(clean_img_vis, f"ground_truth_outputs/real_{idx+1:03d}.png")

        # Prepare numpy arrays for metrics
        fake_clean_np = fake_clean_vis.squeeze(0).permute(1,2,0).cpu().numpy()
        clean_img_np = clean_img_vis.squeeze(0).permute(1,2,0).cpu().numpy()

        # Compute PSNR and SSIM
        psnr_val = compare_psnr(clean_img_np, fake_clean_np, data_range=1.0)
        ssim_val = compare_ssim(clean_img_np, fake_clean_np, channel_axis=2, data_range=1.0)

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

    # Average results
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)

    print(f"\n===== Evaluation Results =====")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("===============================")

if __name__ == "__main__":
    evaluate()