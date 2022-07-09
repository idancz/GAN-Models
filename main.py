import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from train_and_eval import *


def main():
    print(device)
    """ **Data Extraction**"""

    fashion_mnist_train = FashionMNIST(DataDir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    """ ***Training WGANGP***"""

    wgangp_data_loader = torch.utils.data.DataLoader(fashion_mnist_train, batch_size=wgangp_batch_size)
    wgangp = WGANGP_Trainer(verbosity=0)
    wgangp.train(wgangp_data_loader, wgangp_num_of_ephocs, create_gif=True)

    """**Save Model**"""

    wgangp.save_models()

    """**Graphs**"""

    palette = plt.get_cmap('tab20')
    BaseColor = 8
    figure = plt.figure(1, dpi=150)
    plt.rc('xtick', labelsize=6)    # fontsize of the axes labels
    plt.rc('ytick', labelsize=6)    # fontsize of the axes labels
    plt.rc('axes', labelsize=8)     # fontsize of the x and y labels

    WGANGP_data = (wgangp.Loss_dict['G_loss'], wgangp.Loss_dict['D_loss'])

    for count, loss_list in enumerate(WGANGP_data):
      axes = plt.subplot(2,1, count + 1)   # create a new subplot

      figure.tight_layout(pad=2.0) # add spacing between subplots

      plt.plot(range(1, 1 + len(loss_list)), loss_list,
               marker='', color=palette(BaseColor), linewidth=0.5, alpha=0.9)
      BaseColor += 1

      # Add axes titles
      plt.ylabel("Loss", color=palette(BaseColor), fontsize=6)
      plt.xlabel("Iterations", color=palette(BaseColor), fontsize=6)

      # Add grid
      plt.tick_params('x', colors=palette(BaseColor),
                      grid_color='grey', grid_alpha=0.2,
                      grid_linestyle="-")
      plt.tick_params('y', colors=palette(BaseColor),
                      grid_color='grey', grid_alpha=0.2,
                      grid_linestyle="-")
      plt.grid(b=True)

      # Add title and legend
      title = "Generator Loss" if count == 0 else "Discriminator Loss"
      plt.title(title, loc='left', fontsize=10, fontweight=0,
                color=palette(BaseColor))

      BaseColor += 3
    plt.savefig(OutputPath + 'WGANGPLossGraph.jpeg')
    plt.show()

    """**Image Generation**"""

    wgangp.G.GenerateNewImage(num_of_images=5)

    wgangp.G.GenerateNewImage(num_of_images=4)

    wgangp.G.GenerateNewImage(num_of_images=2)

    """ ***Training DCGAN***
    
    """

    dcgan_data_loader = torch.utils.data.DataLoader(fashion_mnist_train, batch_size = dcgan_batch_size)
    dcgan = DCGAN_Trainer(verbosity=0)
    dcgan.train(dcgan_data_loader, dcgan_num_of_ephocs, create_gif=True)

    """**Save Model**"""

    dcgan.save_models()

    """**Graphs**"""

    palette = plt.get_cmap('tab20')
    BaseColor = 0
    figure = plt.figure(1, dpi=150)
    plt.rc('xtick', labelsize=6)    # fontsize of the axes labels
    plt.rc('ytick', labelsize=6)    # fontsize of the axes labels
    plt.rc('axes', labelsize=8)     # fontsize of the x and y labels

    DCGAN_data = (dcgan.Loss_dict['G_loss'], dcgan.Loss_dict['D_loss'])

    for count, loss_list in enumerate(DCGAN_data):
      axes = plt.subplot(2,1, count + 1)   # create a new subplot

      figure.tight_layout(pad=2.0) # add spacing between subplots

      plt.plot(range(1, 1 + len(loss_list)), loss_list,
               marker='', color=palette(BaseColor), linewidth=0.5, alpha=0.9)
      BaseColor += 1

      # Add axes titles
      plt.ylabel("Loss", color=palette(BaseColor), fontsize=6)
      plt.xlabel("Iterations", color=palette(BaseColor), fontsize=6)

      # Add grid
      plt.tick_params('x', colors=palette(BaseColor),
                      grid_color='grey', grid_alpha=0.2,
                      grid_linestyle="-")
      plt.tick_params('y', colors=palette(BaseColor),
                      grid_color='grey', grid_alpha=0.2,
                      grid_linestyle="-")
      plt.grid(b=True)
      if count == 0: # Generator graph
        plt.yticks(np.arange(0, ceil(max(loss_list)), 0.5))

      # Add title and legend
      title = "Generator Loss" if count == 0 else "Discriminator Loss"
      plt.title(title, loc='left', fontsize=10, fontweight=0,
                color=palette(BaseColor))

      BaseColor += 3
    plt.savefig(OutputPath + 'DCGANLossGraph.jpeg')
    plt.show()

    """**Image Generation**"""

    dcgan.G.GenerateNewImage(num_of_images=6)

    dcgan.G.GenerateNewImage(num_of_images=2)

    dcgan.G.GenerateNewImage(num_of_images=4)

    """# **Test Section**
    
    **Load Pretrained Models**
    
    **WGANGP**
    """

    wgangp = WGANGP_Trainer(load_from_file=True)

    """**DCGAN**"""

    dcgan = DCGAN_Trainer(load_from_file=True)

    """**New Images Generation**
    
    **WGANGP**
    """

    wgangp.G.GenerateNewImage(num_of_images=5)

    wgangp.G.GenerateNewImage(num_of_images=4)

    wgangp.G.GenerateNewImage(num_of_images=2)

    """**DCGAN**"""

    dcgan.G.GenerateNewImage(num_of_images=1)

    dcgan.G.GenerateNewImage(num_of_images=4)

    dcgan.G.GenerateNewImage(num_of_images=2)

    num_rows = 1
    num_cols = 2
    for plot in range(1, 2 + 1):
      axes = plt.subplot(num_rows, num_cols, plot)   # create a new subplot
      plt.imshow(fashion_mnist_train.data[plot+15], cmap="gray")


if __name__ == "__main__":
    main()
