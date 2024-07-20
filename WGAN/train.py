import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from DAN_git.CGAN_toyproject.WGAN.model import Discriminator, Generator, initialize_weights
import os
import logging
from torchvision.utils import save_image
# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


directories = ["logs", "logs/real", "logs/fake", "real_images", "fake_images"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataset = datasets.ImageFolder(
    root="/home/sarim.hashmi/side_hussle/gan/DCGAN/animeface", transform=transforms
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)


opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):

    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]


        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        # if batch_idx % 100 == 0 and batch_idx > 0:
        #     print(
        #         f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
        #           Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        #     )

        #     with torch.no_grad():
        #         fake = gen(fixed_noise)
        #         # take out (up to) 32 examples
        #         img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
        #         img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

        #         writer_real.add_image("Real", img_grid_real, global_step=step)
        #         writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        #     step += 1
        if batch_idx % 100 == 0:
            log_message = f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
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