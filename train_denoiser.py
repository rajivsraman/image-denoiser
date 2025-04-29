import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# 1. Load the SIDD dataset
class SIDD_DenoiseDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        super().__init__()
        self.data_dir = os.path.join(root_dir, "SIDD_Small_sRGB_Only", "Data")
        self.folder_names = sorted(os.listdir(self.data_dir))

        if split == "train":
            self.folder_names = self.folder_names[:140]
        elif split == "val":
            self.folder_names = self.folder_names[140:]

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

# 2. Generator (U-Net style)
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

# 3. Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_img, target_img):
        return self.model(torch.cat([input_img, target_img], dim=1))

# 4. Training
def train():
    dataset_path = "."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SIDD_DenoiseDataset(dataset_path, split="train")
    val_dataset = SIDD_DenoiseDataset(dataset_path, split="val")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    num_epochs = 20

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        for i, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            valid = torch.ones(noisy.size(0), 1, 30, 30, device=device)
            fake = torch.zeros(noisy.size(0), 1, 30, 30, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            fake_clean = generator(noisy)
            pred_fake = discriminator(fake_clean, noisy)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_L1 = criterion_L1(fake_clean, clean) * 100
            loss_G = loss_GAN + loss_L1
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(clean, noisy)
            loss_real = criterion_GAN(pred_real, valid)

            pred_fake = discriminator(fake_clean.detach(), noisy)
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            # Print after each batch
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(train_loader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        # Save validation samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            generator.eval()
            val_noisy, val_clean = next(iter(val_loader))
            val_noisy = val_noisy.to(device)
            with torch.no_grad():
                val_output = generator(val_noisy)
            save_image(val_output * 0.5 + 0.5, f"outputs/fake_val_epoch{epoch+1}.png")
            save_image(val_clean * 0.5 + 0.5, f"outputs/real_val_epoch{epoch+1}.png")

        # Save models every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"saved_models/generator_epoch{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch{epoch+1}.pth")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    train()
