import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from DAN_git.CGAN_toyproject.conditional_GAN.model import Discriminator, Generator, initialize_weights

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset_path = "/fsx/homes/Sarim.Hashmi@mbzuai.ac.ae/GAN_toy_project/aladinpearson/cartoon_face/cartoonset100k_jpg"

# Use ImageFolder, which can handle subdirectories
dataset = datasets.ImageFolder(root=dataset_path, transform=transforms)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# initialize gen and disc
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMAGE_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initialize optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

# Create directories for saving images
os.makedirs("images/real", exist_ok=True)
os.makedirs("images/fake", exist_ok=True)

gen.train()
critic.train()

for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    for batch_idx, (real, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)):
        real = real.to(device)
        labels = labels.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, labels, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator
        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses and save images every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise, torch.randint(0, NUM_CLASSES, (32,)).to(device))
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                torchvision.utils.save_image(img_grid_real, f"images/real/real_epoch{epoch+1}.png")
                torchvision.utils.save_image(img_grid_fake, f"images/fake/fake_epoch{epoch+1}.png")

                writer_real.add_image("Real", img_grid_real, global_step=epoch)
                writer_fake.add_image("Fake", img_grid_fake, global_step=epoch)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed. Image grids saved.")

print("Training completed.")


















# import torch 
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from model import Discriminator ,Generator ,initialize_weights

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lr = 2e-4

# batch_size = 128

# image_size = 64

# channels_img = 1

# noise_dim = 100

# num_epochs = 5

# features_d = 64

# features_g = 64


# transforms = transforms.Compose(

# [


#     transforms.Resize(image_size),transforms.ToTensor(),transforms.Normalize(

#         [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
#     ),
# ]



# )

# dataset = datasets.MNIST(

#     root = "dataset/" , train = True , transform = transforms , download=True
# )

# dataloader = Dataloader(dataset, batch_size = batch_size , shuffle = True)

# gen = Generator(noise_dim , channels_img , features_g).to(device)

# disc = Discriminator(channels_img , features_d).to(device)

# initialize_weights(gen)
# initialize_weights(disc)

# opt_gen = optim.Adam(gen.parameters(),lr , betas = (0.5,0.999))

# opt_disc = optim.Adam(disc.parameters(), lr , betas = (0.5 , 0.999))

# criterion = nn.BCELoss()

# fixed_noise = torch.randn(32,noise_dimm,1,1).to(device)

# step = 0

# gen.train()
# disc.train()

# for epoch in range(num_epochs):

#     for batch_idx , (real , _) in enumerate(dataloader):

#         real = real.to(device)

#         noise = torch.randn(batch_size,noise_dim,1,1).to(device)
#         fake = gen(noise)

#         disc_real =disc(real).reshape(-1)

#         loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))

#         disc_fake = disc(fake.detach()).reshape(-1)
#         loss_disc_fake =  criterion(disc_fake , torch.zeros_like(disc_fake))

#         loss_disc = (loss_disc_real + loss_disc_fake) / 2
#         disc.zero_grad()
#         loss_disc.backward()
#         opt_disc.step()


#         output = disc(fake).reshape(-1)
#         loss_gen = criterion(output , torch.ones_like(output))
#         gen.zero_grad()
#         loss_gen.backward()
#         opt_gen.step()

#         if batch_idx % 100 == 0 :

#             print( f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
#                   Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")


#             with torch.no_grad():
#                 fake = gen(fixed_noise)
#                 # take out (up to) 32 examples
#                 img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
#                 img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

#                 writer_real.add_image("Real", img_grid_real, global_step=step)
#                 writer_fake.add_image("Fake", img_grid_fake, global_step=step)

#             step += 1
            

