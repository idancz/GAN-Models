# GAN-Models
Implementation of WGAN and DCGAN on Fashion-MNIST dataset, based on the paper "Improved Training of Wasserstein GANs", by Gulrajani et al.
<br />
Link to paper --> https://arxiv.org/pdf/1704.00028.pdf

# Program Description
## Code Blocks Hierachy 
1.	Initialization – mounting, package imports, seed, GPU allocation, general parameters, data extraction (Fashion-MNIST data set)
2.	WGAN GP – Residual block model, Generator_WGANGP model, Discriminator_WGANGP model, WGANGP_Trainer training class
3.	Training WGANGP – load the data set and train the WGAMGP model using WGANGP_Trainer class
4.	Save Model (WGANGP) – function of WGANGP_Trainer class that saves the trained model
5.	Graphs – plots graphs of the Generator and the Discriminator loss per iteration
6.	Image Generation – generates images using the trained generator (WGANGP)
7.	DCGAN -  Generator_DCGAN model, Discriminator_DCGAN model, DCGAN_Trainer training class
8.	Training DCGAN - load the data set and train the DCGAN model using DCGAN_Trainer class
9.	Save Model (DCGAN) - function of DCGAN_Trainer class that saves the trained model
10.	Graphs - plots graphs of the Generator and the Discriminator loss per iteration
11.	Image Generation – generates images using the trained generator (DCGAN)
12.	Test Section – loads WGANGP and DCGAN pre-trained models and generates new images

## Training
In order to train WGANGP you need to use WGANGP_Trainer class and run the train() method as follow:<br />
![image](https://user-images.githubusercontent.com/108329249/178118963-4e7884fb-fbf1-4862-a922-149500f3073b.png)


