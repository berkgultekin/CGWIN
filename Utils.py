import torch
from torch.autograd import Variable
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import Datasets
from Parameters import hyper_parameters as parameters

device = 'cuda'
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

def set_device(value):
    cuda = value

    global device, FloatTensor, LongTensor
    device = 'cuda' if value else 'cpu'
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# ---------------------
#  Prints GAN progress
# ---------------------
def print_epoch_progress(epoch, d_loss, g_loss):
    print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
          % (epoch, parameters["n_epochs_GAN"], d_loss.item(), g_loss.item())
          )


def print_batch_progress(batch, d_loss, g_loss):
    print("[Batch %d/%d] [D loss: %f] [G loss: %f]"
          % (batch, len(Datasets.trainloader), d_loss.item(), g_loss.item())
          )

# ---------------------
#  Calculates Transformation Loss While Calculating Discriminator Loss
# ---------------------
def calculate_transformation_loss(img, svi, label):
    loss = 0
    loss += svi.evaluate_loss(img.flatten(1).to(device), label.to(device))
    return loss/(parameters["batch_size"]**2)


# ---------------------
#  Computes Gradient Penalty
# ---------------------
def compute_gradient_penalty(discriminator, real_samples, fake_samples, hot_labels):
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    gradients = autograd.grad(
        discriminator(interpolates, hot_labels),
        interpolates,
        Variable(FloatTensor(parameters["batch_size"], 1).fill_(1.0), requires_grad=False),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ---------------------
#  Creates One Hot Label Representation
# ---------------------
def create_one_hot_label(labels):
    hot_labels = []
    hot_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(labels)):
        hot_label[labels[i]] = 1
        hot_labels.append(hot_label)

    return FloatTensor(hot_labels)


# ---------------------
#  Calculates Discriminator Loss
# ---------------------
def calculate_discriminator_loss(discriminator, real_images, fake_images, hot_labels):
    real_validity = discriminator(real_images, hot_labels)
    fake_validity = discriminator(fake_images, hot_labels)

    # Calculating gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data, hot_labels)

    # Calculating total penalty
    total_loss = -torch.mean(real_validity) + \
                 torch.mean(fake_validity) + \
                 parameters["lambda_gp"] * gradient_penalty

    return total_loss


# ---------------------
#  Prints GAN progress
# ---------------------
def print_progress(epoch, d_loss, g_loss):
    print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
          % (epoch, parameters["n_epochs_GAN"], d_loss.item(), g_loss.item())
          )


# ---------------------
#  Converts Images and Labels
# ---------------------
def arrange_data_tensors(images, labels):
    images = Variable(images.type(FloatTensor))
    labels = Variable(labels.type(LongTensor))

    return images, labels

# ---------------------
#  Generates image from generator and saves if save parameter is true
# ---------------------
def sample_single_image(images, index, generator, image_path, save):
    z = Variable(FloatTensor(np.random.normal(0, 1, (parameters["batch_size"], parameters["latent_dim"]))))
    generated_images = generator(z, images)
    #if (save == True):
        #save_image(generated_images[index].data, image_path, nrow=1, normalize=True)

    return generated_images[index].data


# ---------------------
#  Shows the given image
# ---------------------
def show_image(img):
    img = img.view(parameters["img_size"], parameters["img_size"])
    img = img.type(torch.FloatTensor).detach().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()
