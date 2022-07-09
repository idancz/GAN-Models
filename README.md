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
<br />
In order to train DCGAN you need to use DCGAN_Trainer class and run the train() method as follow:<br />
![image](https://user-images.githubusercontent.com/108329249/178119005-66e8cb65-7c42-4b91-a50e-81a19563db09.png)

## Load and Test
In order to load and use trained WGANGP/DCGAN model you need to use WGANGP_Trainer/DCGAN_Trainer class as follow:<br />
![image](https://user-images.githubusercontent.com/108329249/178119061-0307376e-64cc-4974-9fba-a3fc1d91bd22.png)

## Results
### WGANGP
I implemented WGANGP model with Residual block and gradient penalty as described in the paper "Improved Training of Wasserstein GANs", by Gulrajani et<br />
You can see that after ~5K iterations the generator and the discriminator stabilized.<br />
That’s makes sense because the generator generates more realistic images, and it becomes harder for the discriminator to distinguish.<br />
Epochs = 130, Batch size = 50<br />
![image](https://user-images.githubusercontent.com/108329249/178119128-3443949d-7695-4244-b63a-93400f7a121c.png)
<br />
![image](https://user-images.githubusercontent.com/108329249/178119194-2a123ce8-bf57-41f5-9538-b1445d541e5a.png)


### DCGAN
You can see from the graph below that when the generator gets better results (loss decreases) the discriminator loss increases and stabilizes around 0.5 loss.<br />
The main difference from the WGANGP is that after around 20K iteration the DCGAN generator loss increases.<br />
Epochs = 60, Batch size = 128<br />
![image](https://user-images.githubusercontent.com/108329249/178119184-e9e79bf6-3c10-4684-9208-425cc97bf6fb.png)
<br />
![image](https://user-images.githubusercontent.com/108329249/178119189-88b265dd-e178-40a9-bda6-a53fae0e787c.png)





