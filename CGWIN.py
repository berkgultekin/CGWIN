import os.path
import os
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import random
import subprocess
import pyro.distributions as dist
import pyro
from pyro import optim
from pyro.infer import SVI, Trace_ELBO
import torch.nn as nn
import torch
import numpy as np
import plotly.graph_objects as go

import Datasets
import Utils
from Parameters import hyper_parameters as parameters

os.makedirs("images", exist_ok=True)

cuda = True
device = 'cuda' if cuda else 'cpu'
Utils.set_device(cuda)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

img_shape = (parameters["channels"], parameters["img_size"], parameters["img_size"])


# ---------------------
#  BNN - Classifier Model
# ---------------------
class BNN(nn.Module):
    def __init__(self):
        super(BNN, self).__init__()

        self.out = nn.Sequential(
            nn.Linear(parameters["img_size"] ** 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = x.flatten(1)
        return self.out(x)


# ---------------------
#  Generator Class
# ---------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.fc = nn.Linear(parameters["latent_dim"], parameters["channels"] * parameters["img_size"] ** 2)

        self.l1 = nn.Sequential(nn.Conv2d(parameters["channels"] * 2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        self.conv_blocks_for_image = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=2, padding=1),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(16, 0.8),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
        )

        self.l2 = nn.Sequential(
            nn.Linear(4 * parameters["img_size"] ** 2, parameters["channels"] * parameters["img_size"] ** 2),
            nn.Tanh(),
        )

    def forward(self, z, img):
        # img = play_with_image(img)
        first_z = self.fc(z).view(
            [parameters["batch_size"], 1, int(parameters["img_size"]), int(parameters["img_size"])])
        gen_input = torch.cat((img, first_z), 1)
        out = self.l1(gen_input)
        generated_img = self.conv_blocks(out)
        return generated_img


# ---------------------
#  Discriminator Class
# ---------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + parameters["n_classes"], 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, labels):
        gen_input = torch.cat(
            (img.view(parameters["img_size"] * parameters["img_size"], parameters["batch_size"]),
             labels.view(parameters["n_classes"], parameters["batch_size"])), 0). \
            view(parameters["batch_size"], parameters["img_size"] * parameters["img_size"] + parameters["n_classes"])
        validity = self.model(gen_input)
        return validity


# ---------------------
#  Setting Datasets
# ---------------------
if parameters["run_download_sh"]:
    subprocess.call("download.sh", shell=True)


# ---------------------
#  Test
# ---------------------
def test_models():
    load_or_train_models()
    print("Generating images and printing results...")

    tableHeader = dict(values=['τ', '%Reject', 'BNN Acc.', 'BNN+GWIN Acc.', 'Rejected Acc.', 'Overall Acc.', '% Error'])
    thresholds = []
    rejects = []
    bnnAccs = []
    bnnGwınAccs = []
    rejectedAccs = []
    overallAccs = []
    errors = []
    showed_images = []
    showed_texts = []

    realValues = [["0.7 (Expected)", "0.8 (Expected)", "0.9 (Expected)"],
                  [1.83, 2.74, 4.39], [54.49, 58.91, 68.79],
                  [85.07, 86.30, 86.95], [30.59, 27.39, 18.16],
                  [0.56, 0.75, 0.80], [-27.55, -36.36, 40.26]]

    for threshold_index in range(len(parameters["threshold"])):
        print('Working on dataset with Threshold: %2f' % round(parameters["threshold"][threshold_index], 2))
        counter = 0
        probUnderThreshold = 0
        probUnderThresholdButCorrectClassified = 0
        probUnderThresholdThenClassifiedCorrectClassified = 0
        probAboveThresholdAndCorrectClassified = 0
        probAboveThresholdAlsoClassifiedCorrectClassified = 0

        for i in range(10):
            for j in range(int(len(Datasets.testset) / parameters["batch_size"])):
                # obtain one batch of training images
                dataiter = iter(Datasets.testloader)
                images, labels = dataiter.next()

                images, labels = Utils.arrange_data_tensors(images, labels)

                predsProbs, preds = predict(images, labels)

                for k in range(parameters["batch_size"]):

                    idx = (i * int(len(Datasets.testset))) + (j * parameters["batch_size"]) + k

                    trueLabel = labels[k].item()
                    modelsPrediction = preds[k].item()
                    modelsPredictionProb = predsProbs[k].item()

                    if (modelsPredictionProb < parameters["threshold"][threshold_index]):
                        probUnderThreshold = probUnderThreshold + 1
                        if (trueLabel == modelsPrediction):
                            probUnderThresholdButCorrectClassified = probUnderThresholdButCorrectClassified + 1
                    else:
                        if (trueLabel == modelsPrediction):
                            probAboveThresholdAndCorrectClassified = probAboveThresholdAndCorrectClassified + 1

                    if (predsProbs[k] < parameters["threshold"][threshold_index]):
                        save_image(images[k], "images/result_images/%d" % idx + "_original_image.png", nrow=1,
                                   normalize=True)

                        generatedImage = Utils.sample_single_image(images, k, generator,
                                                                   "images/result_images/%d" % idx + "_generated_image_.png",
                                                                   True)
                        lastPredsProbs, lastPreds = predict(generatedImage.view(1, 1, 28, 28), labels)

                        if not modelsPrediction == trueLabel and lastPreds.item() == trueLabel and counter < 5:
                            showed_images.append(images[k])
                            showed_images.append(generatedImage)
                            showed_texts.append("Old Label: %2d" %int(preds[k]) + "(Prob: %2f" %round(float(predsProbs[k]), 2) + ")" + " - " +
                                                "New Label: %2d" %int(lastPreds.item()) + "(Prob: %2f" %round(float(lastPredsProbs.item()), 2) + ")")
                            counter = counter + 1


                        if (lastPreds.item() == trueLabel):
                            probUnderThresholdThenClassifiedCorrectClassified = probUnderThresholdThenClassifiedCorrectClassified + 1

        total_images = 100000.0
        o = 0
        m = 0
        while o < len(showed_images):
            Utils.showGrid(showed_images[o], showed_images[o + 1], showed_texts[m])
            o = o + 2
            m = m + 1

        reject = (float(100 * probUnderThreshold) / total_images)
        bnnAcc = (100 * probUnderThresholdButCorrectClassified / probUnderThreshold)
        bnn_and_gwinAcc = (100 * probUnderThresholdThenClassifiedCorrectClassified / probUnderThreshold)
        overallAcc = (float(100 * (
                probUnderThresholdThenClassifiedCorrectClassified - probUnderThresholdButCorrectClassified)) / total_images)
        error = (100 * (bnnAcc - bnn_and_gwinAcc)) / probUnderThreshold

        thresholds.append(parameters["threshold"][threshold_index])
        thresholds.append(realValues[0][threshold_index])
        rejects.append(reject)
        rejects.append(realValues[1][threshold_index])
        bnnAccs.append(round(bnnAcc, 2))
        bnnAccs.append(realValues[2][threshold_index])
        bnnGwınAccs.append(round(bnn_and_gwinAcc, 2))
        bnnGwınAccs.append(realValues[3][threshold_index])
        rejectedAccs.append(round((bnn_and_gwinAcc - bnnAcc), 2))
        rejectedAccs.append(realValues[4][threshold_index])
        overallAccs.append(overallAcc)
        overallAccs.append(realValues[5][threshold_index])
        errors.append(round(error, 2))
        errors.append(realValues[6][threshold_index])

    tableCells = dict(values=[thresholds, rejects, bnnAccs, bnnGwınAccs, rejectedAccs, overallAccs, errors])
    resultsFig = go.Figure(data=[go.Table(header=tableHeader, cells=tableCells)])
    resultsFig.show("png")


# ---------------------
#  Trains or loads classifier model
# ---------------------
def create_classifier_model():
    if (not os.path.exists('ClassifierModel.pt') or not os.path.exists('paramstore.out')):
        train_classifier_model()
        test_classifier()
    else:
        pyro.get_param_store().load('paramstore.out', map_location=device)
        model.load_state_dict(torch.load('ClassifierModel.pt', map_location=device))


model = BNN()
if cuda:
    model.cuda()


# ---------------------
#  Stochastic Variational Inference's Module
# ---------------------
def module(x, y):
    priors = {}
    for iterator in model.named_parameters():
        name, param = iterator
        zeros = torch.zeros_like(param.data)
        ones = torch.ones_like(param.data)
        priors[name] = dist.Normal(loc=zeros,
                                   scale=ones)

    lifted_module = pyro.random_module("module", model, priors)
    lifted_module_method = lifted_module()
    lhat = F.log_softmax(lifted_module_method(x), 1)
    pyro.sample("obs", dist.Categorical(logits=lhat), obs=y)


# ---------------------
#  Stochastic Variational Inference's Guide
# ---------------------
def guide(x, y):
    priors = {}
    for iterator in model.named_parameters():
        name, param = iterator
        priors[name] = dist.Normal(loc=pyro.param(name + '.mu', torch.randn_like(param)),
                                   scale=F.softplus(pyro.param(name + '.sigma', torch.randn_like(param))))

    lifted_module = pyro.random_module('module', model, priors)
    return lifted_module()


opt = optim.Adam({'lr': parameters["lr_Classifier"]})
svi = SVI(module, guide, opt, loss=Trace_ELBO())


# ---------------------
#  Trains Classifier Model
# ---------------------
def train_classifier_model():
    pyro.clear_param_store()

    total_loss = 0
    for epoch in range(parameters["n_epochs_Classifier"]):
        loss = 0
        for x, y in Datasets.trainloader:
            loss += svi.step(x.flatten(1).to(device), y.to(device))
        total_loss = loss / len(Datasets.trainloader.dataset)
        print("Epoch: %d Loss: %f" % (epoch + 1, total_loss))

    pyro.get_param_store().save('paramstore.out')
    torch.save(model.state_dict(), 'ClassifierModel.pt')


# ---------------------
#  Classifier's prediction method
# ---------------------
def predict(x, y):
    sampled_models = [guide(None, None) for _ in range(parameters["n_classes"])]
    yhats = [model(x.to(device)).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    predsProbs, preds = torch.max(F.softmax(mean).to(device), 1)
    return predsProbs, preds


# ---------------------
#  Test for Classifier
# ---------------------
def test_classifier():
    correct = 0
    total = 0
    for x, y in Datasets.testloader:
        x, y = Utils.arrange_data_tensors(x, y)
        predsProbs, preds = predict(x.flatten(1), y)
        total += parameters["batch_size"]
        correct += (preds == y).sum().item()
    print("Accuracy: %f" % (correct / total))


create_classifier_model()
mask = []


# ---------------------
#  Creates Mask To Train Discriminator With P_Critic
# ---------------------
def create_critic_mask():
    print("Creating mask to train model with only critic dataset...")
    for i in range(int(len(Datasets.trainset) / parameters["batch_size"])):
        # obtain one batch of training images
        dataiter = iter(Datasets.trainloader)
        images, labels = dataiter.next()
        images, labels = Utils.arrange_data_tensors(images, labels)

        predsProbs, preds = predict(images, labels)
        # convert output probabilities to predicted class
        # predsProbs, preds = torch.max(output, 1)
        for j in range(parameters["batch_size"]):
            mask.append(1 if predsProbs[j].item() > parameters["critic_threshold"] else 0)


create_critic_mask()
mask = FloatTensor(mask)


# ---------------------
#  Sampler Class to use mask
# ---------------------
class SpecialSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(mask)])

    def __len__(self):
        return len(self.data_source)


# ---------------------
#  Loads Models Or Trains Classifier, Generator, Discriminator Models
# ---------------------
def load_or_train_models():
    create_classifier_model()

    if parameters["continue_on_existing_training"] or (
            not (os.path.exists('DiscriminatorModel.pt') or os.path.exists('GeneratorModel.pt'))):

        if parameters["continue_on_existing_training"] == True and (
                os.path.exists('DiscriminatorModel.pt') and os.path.exists('GeneratorModel.pt')):
            load_GAN_models()

        train_GAN()
    else:
        load_GAN_models()


def play_with_image(img):
    for x in range(parameters["batch_size"]):
        for _ in range(200):
            i = random.randint(0, 27)
            j = random.randint(0, 27)
            rand = random.uniform(-1, 1)
            img[x][0][i][j] = rand
    return img


generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=parameters["lr_GAN"],
                                       betas=(parameters["b1"], parameters["b2"]))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=parameters["lr_GAN"],
                                           betas=(parameters["b1"], parameters["b2"]))


# ---------------------
#  Trains Generative Adverserial Network model
# ---------------------
def train_GAN():
    sampler = SpecialSampler(mask, Datasets.trainset)
    criticLoader = torch.utils.data.DataLoader(
        Datasets.trainset,
        drop_last=True,
        batch_size=parameters["batch_size"],
        sampler=sampler,
        shuffle=False
    )
    batches_done = 0
    for epoch in range(parameters["n_epochs_GAN"]):
        for i, (imgs, labels) in enumerate(criticLoader):
            hot_labels = Utils.create_one_hot_label(labels)

            real_images = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            discriminator_optimizer.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (parameters["batch_size"], parameters["latent_dim"]))))

            generated_images = generator(z, real_images)

            d_loss = Utils.calculate_discriminator_loss(discriminator, real_images, generated_images, hot_labels)

            d_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()

            if i % parameters["n_critic"] == 0:
                dataiter = iter(Datasets.trainloader)
                pr_images, pr_labels = dataiter.next()

                pr_images = Variable(pr_images.type(FloatTensor))
                pr_labels = Variable(pr_labels.type(LongTensor))

                generated_images = generator(z, pr_images)
                pr_hot_labels = Utils.create_one_hot_label(pr_labels)

                fake_validity = discriminator(generated_images, pr_hot_labels)
                # Calculating transformation penalty
                transformation_loss = Utils.calculate_transformation_loss(generated_images, svi, pr_labels)

                g_loss = -torch.mean(fake_validity) + (parameters["lambda_loss"] * transformation_loss)

                g_loss.backward()
                generator_optimizer.step()

                Utils.print_batch_progress(i, d_loss, g_loss)
                if batches_done % 100 == 0:
                    save_image(generated_images.data[:25], "images/wgan/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += batches_done + parameters["n_critic"]

        Utils.print_epoch_progress(epoch + 1, d_loss, g_loss)

    save_GAN_models()


# ---------------------
#  Loads Generator and Discriminator Models
# ---------------------
def load_GAN_models():
    generator.load_state_dict(torch.load('GeneratorModel.pt', map_location=device))
    discriminator.load_state_dict(torch.load('DiscriminatorModel.pt', map_location=device))


# ---------------------
#  Saves Generator and Discriminator Models
# ---------------------
def save_GAN_models():
    torch.save(generator.state_dict(), 'GeneratorModel.pt')
    torch.save(discriminator.state_dict(), 'DiscriminatorModel.pt')


# train_classifier_model()
# load_or_train_models()
test_models()
