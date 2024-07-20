import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DAN_git.CGAN_toyproject.GAN_cartoon_face.model import Discriminator, Generator, initialize_weights
import os
import logging
from torchvision.utils import save_image
from tqdm import tqdm

# Create directories
directories = ["logs", "logs/real", "logs/fake", "real_images", "fake_images"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
    ),
])

# dataset = datasets.ImageFolder(
#     root="/fsx/homes/Sarim.Hashmi@mbzuai.ac.ae/GAN_toy_project/aladinpearson/cartoon_face/cartoonset100k_jpg", transform=transforms
# )
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

dataset_path = "/fsx/homes/Sarim.Hashmi@mbzuai.ac.ae/GAN_toy_project/aladinpearson/cartoon_face/cartoonset100k_jpg"

# Use ImageFolder, which can handle subdirectories
dataset = datasets.ImageFolder(root=dataset_path, transform=transforms)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    for batch_idx, (real, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        # Train Discriminator
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            log_message = f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            tqdm.write(log_message)
            logging.info(log_message)

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

    # Save images after each epoch
    with torch.no_grad():
        fake = gen(fixed_noise)
        save_image(torchvision.utils.make_grid(real[:32], normalize=True),
                   f"real_images/real_epoch_{epoch+1}.png")
        save_image(torchvision.utils.make_grid(fake[:32], normalize=True),
                   f"fake_images/fake_epoch_{epoch+1}.png")

    log_message = f"Epoch {epoch+1} completed. Images saved."
    tqdm.write(log_message)
    logging.info(log_message)