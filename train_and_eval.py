import torch.optim as optim
from torch.autograd import grad, Variable
from torchvision.utils import make_grid

import imageio

import json
import timeit

from GAN import *

""" **General Parameters**"""

MainDir = ''
DataDir = MainDir + 'Data/'
ModelsPath = MainDir + 'Models/'
ParamsPath = MainDir + 'Params/'
OutputPath = MainDir + 'Outputs/'

image_size = 784 #28*28
wgangp_batch_size = 50
wgangp_num_of_ephocs = 130
dcgan_batch_size = 128
dcgan_num_of_ephocs = 60


""" **Model Trainer**"""
class WGANGP_Trainer(nn.Module):
    def __init__(self, name="WGANGP", gp_lambda=10, output_dim=image_size, latent_size=latent_size,
                 batch_size=wgangp_batch_size, load_from_file=False,
                 verbosity = 400):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.output_dim = output_dim

        self.gp_lambda = gp_lambda # Gradient penalty lambda - according to paper = 10

        ### Run parameters: ###
        self.critic_iters = 5  # num of ctiric iterations per generator interation
        self.num_steps = 0
        self.verbosity = verbosity

        ### Losses: ###
        self.Loss_dict = { 'G_loss' : [], 'D_loss' : [], 'GP_loss' : []}
        if load_from_file:
            loss_path = ParamsPath + self.name + "_losses.json"
            with open(loss_path, 'r') as json_file:
                self.Loss_dict = json.load(json_file)
        self.G = Generator_WGANGP(self.latent_size, self.output_dim)
        self.D = Discriminator_WGANGP(1 ,self.latent_size)

        if load_from_file:
            self.G.load_state_dict(torch.load(ModelsPath + self.name + "_generator.pt"))
            self.D.load_state_dict(torch.load(ModelsPath + self.name + "_discriminator.pt"))
        else:
            self.G.InitParameters()
            self.D.InitParameters()

        self.G = self.G.to(device)
        self.D = self.D.to(device)

        self.gen_opt = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.dis_opt = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))


    def discriminator_loss(self, real_output, fake_output, gradient_penalty):
        '''
        Calculates Discriminator loss, using gradient penalty
        '''
        return fake_output.mean() - real_output.mean() + gradient_penalty

    def generator_loss(self, fake_output):
        '''
        Calculates Generator loss, using specified criterion
        fake_output = Discriminator output over fake images for Generator
        '''
        return - fake_output.mean()

    def run_generator(self, num_samples):
        '''
        generates noise in the given size and generates images using Generator.
        '''
        latent_samples = Variable(self.G.GenerateNoise(num_samples)).to(device)
        generated_images = self.G(latent_samples)
        return generated_images

    def train_discriminator(self, real_images, generated_images):
        """
        Trains Discriminator for one iteration.
        real_images = real images from Fashoin MNIST
        """
        # Calculate probabilities on real and generated data
        real_images = Variable(real_images).to(device)
        d_real = self.D(real_images)
        d_generated = self.D(generated_images.detach())

        # Get gradient penalty
        real_data = real_images.data
        fake_data = generated_images

        # Sample uniformely along straight lines between RealImage distribution and FakeImage Distribution.
        batch_size = real_data.size()[0]  # number of data points
        alpha = torch.rand(batch_size, 1 ,1 ,1)  # random between [0,1)
        alpha = alpha.expand_as(real_data)  # same size as Data
        alpha = alpha.to(device)

        line_samples = real_data + alpha * (fake_data - real_data)  # sample from line
        line_samples = Variable(line_samples, requires_grad=True)
        line_samples = line_samples.to(device)

        D_sample = self.D(line_samples)

        gradients = grad(outputs=D_sample, inputs=line_samples, grad_outputs=torch.ones(D_sample.size()).to(device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda  # gp_lambda = 10 according to paper


        # gradient_penalty = CalculateGradientPenalty(self.D, real_images.data, generated_images)
        self.Loss_dict['GP_loss'].append(gradient_penalty.item())

        # Create total loss and optimize
        self.D.zero_grad()
        d_loss = self.discriminator_loss(d_real, d_generated, gradient_penalty)
        d_loss.backward()

        self.dis_opt.step()

        # Record loss
        self.Loss_dict['D_loss'].append(d_loss.item())

    def train_generator(self, data):
        """
        Trains Generator for one iteration.
        data - real images from FashionMNIST
        """
        self.G.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.run_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = self.generator_loss(d_generated)
        g_loss.backward()
        self.gen_opt.step()

        # Save loss
        self.Loss_dict['G_loss'].append(g_loss.item())

    def run_epoch(self, data_loader):
        for iteration, batch_data in enumerate(data_loader):
            self.num_steps += 1
            # print(self.num_steps , 'steps', 'data loader',enumerate(data_loader))
            size_of_batch = batch_data[0].size()[0]
            generated_data = self.run_generator(size_of_batch)
            # Train and update Discriminator
            self.train_discriminator(batch_data[0], generated_data)
            # Train and update Generator, only every critic_iters
            if 0 == self.num_steps % self.critic_iters:
                self.train_generator(batch_data[0])

            if self.verbosity and 0 == (iteration + 1) % self.verbosity:
                print("Iteration {:4d} | D: {:.3f} | GP: {:.3f} | G: {:.3f}".format(
                    iteration + 1,
                    self.Loss_dict['D_loss'][-1],
                    self.Loss_dict['GP_loss'][-1],
                    self.Loss_dict['G_loss'][-1]))

    def train(self, data_loader, num_of_epochs, create_gif=False):
        if create_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.GenerateNoise(120)).to(device)
            training_progress_images = []

        tic = timeit.default_timer()

        for epoch in range(num_of_epochs):
            if self.verbosity and 0 == epoch % 5:
                print("\nRunning Epoch {}".format(epoch))
            self.run_epoch(data_loader)
            if create_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data, nrow=10, normalize=True)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = (np.transpose(img_grid.numpy(), (1, 2, 0)) * 256).astype('uint8')
                # Add image grid to training progress
                training_progress_images.append(img_grid)
            toc = timeit.default_timer()
            print("Epoch {} ended, since beginning = {:d} mins".format(epoch, round((toc -tic ) /60)))

        if create_gif:
            imageio.mimsave((OutputPath +'WGANGP_training_for_{}_epochs.gif').format(num_of_epochs),
                            training_progress_images)
        print("\nDone training!")

    def save_models(self):
        '''
        Saves the trained models (generator and Discriminator) and all loss history.
        '''
        # Save Generator
        generator_path = ModelsPath + self.name + "_generator.pt"
        print("Saving Generator to {}...".format(generator_path))
        torch.save(self.G.state_dict(), generator_path)
        # Save Discriminator
        discriminator_path = ModelsPath + self.name + "_discriminator.pt"
        print("Saving Discriminator to {}...".format(discriminator_path))
        torch.save(self.D.state_dict(), discriminator_path)
        # Save losses
        loss_path = ParamsPath + self.name + "_losses.json"
        print("Saving Loss history to {}...".format(loss_path))
        with open(loss_path, 'w') as json_file:
            json.dump(self.Loss_dict, json_file)
        print("Saved models")


"""**Model Trainer**"""


class DCGAN_Trainer(nn.Module):
    def __init__(self, name="DCGAN", output_dim=image_size, latent_size=latent_size,
                 batch_size=dcgan_batch_size, load_from_file=False, dim=16,
                 verbosity=400):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.latent_size = latent_size
        self.dim = dim
        ### Run parameters: ###
        self.critic_iters = 1  # num of ctiric iterations per generator interation
        self.num_steps = 0
        self.verbosity = verbosity

        self.real_label = 1
        self.fake_label = 0

        ### Losses: ###
        self.Loss_dict = {'G_loss': [], 'D_loss': [], 'GP_loss': []}
        if load_from_file:
            loss_path = ParamsPath + self.name + "_losses.json"
            with open(loss_path, 'r') as json_file:
                self.Loss_dict = json.load(json_file)
        self.G = Generator_DCGAN(self.latent_size, self.dim)
        self.D = Discriminator_DCGAN(self.dim)

        if load_from_file:
            self.G.load_state_dict(torch.load(ModelsPath + self.name + "_generator.pt"))
            self.D.load_state_dict(torch.load(ModelsPath + self.name + "_discriminator.pt"))

        self.G = self.G.to(device)
        self.D = self.D.to(device)

        self.gen_opt = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.dis_opt = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.g_criterion = nn.BCELoss()
        self.d_criterion = nn.BCELoss()

    def discriminator_loss(self, real_output, fake_output):
        '''
        Calculates Discriminator loss, using specified criterion
        '''
        batch_size = real_output.size()[0]
        ones = torch.full((batch_size,), self.real_label, device=device)
        zeros = torch.full((batch_size,), self.fake_label, device=device)
        fake_bce = self.d_criterion(fake_output, zeros)
        real_bce = self.d_criterion(real_output, ones)
        return fake_bce + real_bce

    def generator_loss(self, fake_output):
        '''
        Calculates Generator loss, using specified criterion
        fake_output = Discriminator output over fake images for Generator
        '''
        batch_size = fake_output.size()[0]
        ones = torch.full((batch_size,), self.real_label, device=device)
        fake_output = fake_output.to(torch.float32)
        ones = ones.to(torch.float32)
        return self.g_criterion(fake_output, ones)

    def train_discriminator(self, real_images, generated_images):
        """
        Trains Discriminator for one iteration.
        real_images = real images from Fashoin MNIST
        generated_images = generated images from Generator
        """
        self.D.zero_grad()
        # Get generated images and real images
        size_of_batch = real_images.size()[0]

        real_images = Variable(real_images).to(device)

        # create real and fake labels to compare
        ones = torch.full((size_of_batch,), self.real_label, device=device)
        zeros = torch.full((size_of_batch,), self.fake_label, device=device)

        # Way num. 1: Calculate gradient seperately
        # Calculate probabilities on real data
        d_real = self.D(real_images)
        d_real = d_real.to(torch.float32)
        ones = ones.to(torch.float32)
        real_bce = self.d_criterion(d_real, ones)
        real_bce.backward()
        # Calculate probabilities on generated data
        d_generated = self.D(generated_images.detach())
        d_generated = d_generated.to(torch.float32)
        zeros = zeros.to(torch.float32)

        fake_bce = self.d_criterion(d_generated, zeros)
        fake_bce.backward()

        # Calculate combined loss
        d_loss = (fake_bce + real_bce) / 2
        # do step
        self.dis_opt.step()

        # Record loss
        self.Loss_dict['D_loss'].append(d_loss.item())

    def run_generator(self, num_samples):
        '''
        generates noise in the given size and generates images using Generator.
        '''
        latent_samples = Variable(self.G.GenerateNoise(num_samples)).to(device)
        generated_images = self.G(latent_samples)
        return generated_images

    def train_generator(self, generated_data):
        """
        Trains Generator for one iteration.
        generated_data - generated images by Generator.
        """
        self.G.zero_grad()

        # Calculate loss and optimize
        d_generated = self.D(generated_data)

        g_loss = self.generator_loss(d_generated)
        g_loss.backward()
        self.gen_opt.step()

        # Save loss
        self.Loss_dict['G_loss'].append(g_loss.item())

    def run_epoch(self, data_loader):
        for iteration, batch_data in enumerate(data_loader):
            self.num_steps += 1
            batch_size = batch_data[0].size()[0]
            generated_data = self.run_generator(batch_size)
            # Train and update Generator, only every critic_iters
            if 0 == self.num_steps % self.critic_iters:
                self.train_generator(generated_data)
            self.train_discriminator(batch_data[0], generated_data)
            if self.verbosity and 0 == (iteration + 1) % self.verbosity:
                print("Iteration {:4d} | D: {:.3f} | G: {:.3f}".format(iteration + 1,
                                                                       self.Loss_dict['D_loss'][-1],
                                                                       self.Loss_dict['G_loss'][-1]))

    def train(self, data_loader, num_of_epochs, create_gif=False):
        if create_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.GenerateNoise(120)).to(device)
            training_progress_images = []

        tic = timeit.default_timer()
        for epoch in range(num_of_epochs):
            if self.verbosity and 0 == epoch % 5:
                print("\nRunning Epoch {}".format(epoch))
            self.run_epoch(data_loader)
            if create_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data, nrow=10, normalize=True)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = (np.transpose(img_grid.numpy(), (1, 2, 0)) * 256).astype('uint8')
                # Add image grid to training progress
                training_progress_images.append(img_grid)
            toc = timeit.default_timer()
            print("Epoch {} ended, training for {:d} mins".format(epoch, round((toc - tic) / 60)))

        if create_gif:
            imageio.mimsave((OutputPath + 'DCGAN_training_for_{}_epochs.gif').format(num_of_epochs),
                            training_progress_images)
        print("\nDone training!")

    def save_models(self):
        '''
        Saves the trained models (generator and Discriminator) and all loss history.
        '''
        # Save Generator
        generator_path = ModelsPath + self.name + "_generator.pt"
        print("Saving Generator to {}...".format(generator_path))
        torch.save(self.G.state_dict(), generator_path)
        # Save Discriminator
        discriminator_path = ModelsPath + self.name + "_discriminator.pt"
        print("Saving Discriminator to {}...".format(discriminator_path))
        torch.save(self.D.state_dict(), discriminator_path)
        # Save losses
        loss_path = ParamsPath + self.name + "_losses.json"
        print("Saving Loss history to {}...".format(loss_path))
        with open(loss_path, 'w') as json_file:
            json.dump(self.Loss_dict, json_file)
        print("Saved models")
