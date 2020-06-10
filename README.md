# Paper Information
* Name: Generative Well-intentioned Networks
* Link: https://papers.nips.cc/paper/9467-generative-well-intentioned-networks.pdf

# Authors of Code
* Yasin Berk Gültekin - e194211@metu.edu.tr
* Hasan Ali Duran - e194199@metu.edu.tr

# Paper Summary
Paper proposes a new framework in order to increase the classifier’s accuracy rate. It uses conditional GAN to understand the distribution of the data which classifier labels with high certainty. It adds an rejection option to classifier since making a prediction about a uncertain input would decrease the accuracy. These rejected inputs translated by the GAN to high-certainty inputs and then relabeled by the classifier.
They combined different loss functions in the paper:
 
![alt text](https://github.com/berkgultekin/CGWIN/blob/master/LossFunction.PNG?raw=true)

Paper uses Wasserstein GAN with gradient penalty and adds Transformation Penalty. Transformation Penalty is a new loss function with a transformation penalty that encourages the conditional generator to produce images that the classifier will label correctly. This penalty is the loss of the classifier when labeling the transformed observations in the current training batch
The WGAN-GP critic is typically trained on both generated data and real data. However, they want the GWIN to generate images from the classifier’s confident distribution. Thus, they prefilter the training data to create a confident distribution Pc containing all images that the classifier labels correctly with high certainty. The critic is then trained exclusively on samples drawn from Pc and generated images by the generator.
Paper uses two different datasets; the MNIST handwritten digits dataset and the Fashion-MNIST clothing dataset.

# Implementation Steps
First of all, we had to calculate a classification precision score for mnist data. For this reason, we had to first implement a classifier. We have implemented our Bayesian Neural Network classifier. Then we train the classifier using all the mnist data. Afterwards, our classifier has calculated the score for each data separately. We used these scores in order to create a subset of our data. And we called this subset critic dataset, we structured this critic data to consist only of those with very high classification scores.
Then we wanted to recreate the data whose classification scores are below a certain threshold, through our generative model. To achieve this, we first implemented a vanillia gan. Later, we updated the discrimator to add contidition to vanillia gan and updated the images to use class label information as input. Then we updated Generator to receive an image as input because we wanted to produce the given image in a better quality. Later, we updated our GAN model by adding the loss functions mentioned in the paper. First of all, we made Wasserstein GAN. We then added a gradient penalty to develop the discriminator. Finally, we calculated the transformation loss and thus, we enabled our Generator to improve for the images we have generated. The model we have has turned into a conditionally customized version of Wasserstein Gan with Gradient Penalty. We then trained only the discriminator using the critic data, but the generator was trained using all the data.
After the entire train process was over, we sent and recreated the data that the classifier did not give enough confidence score to our model. We then calculated the score using the classifier again for the re-created images and also classified them. Then, we compared the original results using the classification results of the new score values and the recreated images. Then, we presented these results to users for comparison.
       
# Workflow of Code
![alt text](https://github.com/berkgultekin/CGWIN/blob/master/CodeStructure.PNG?raw=true)

Our system first checks to see if we have a train model. If there is a model that has already been trained, it will load or load these models separately as Classifier and GAN models. If the train operation is to be performed again, it creates the critic dataset because the discriminator in Gan model becomes the train using this data. It also saves the related models after the train operation is finished.
When an image comes next, this image is first examined in the classifier model and an attempt is made to estimate a class label, as well as a confidence score is calculated. If this score is below a certain threshold, the classifier rejects this image and the rejected image is sent to the GAN model. The GAN model uses a rejected image to produce a new image to preserve the image features. This expects the produced image to have a higher confidence score and to be correctly classified.
Then, the re-created image is sent to the classifier and expected to classify again. Classifier's last results are compared with the first results and accordingly, it is calculated how our model offers an overall improvement.

# Challenges Encountered When Implementing Paper
* There was not enough detail about how BNN was implemented. The number of layers was not specified. We could not fully obtain the BNN results mentioned in Paper. The BNN we have implemented makes predictions with higher scores. This situation caused difficulties in the exact occurrence of qualitative results.
* The biggest problem we encountered during GAN implementation was that it was not clear enough how the Generator and Discriminator inputs should be processed in the model. It was not specified how many layers or what types of layers were used. In addition, the pictures produced by the generator appeared similar to those given as input to the Generator. There was no explanation for how this problem was solved in paper implementation. The new method which is transformation loss used during the Discriminator's loss calculation was not sufficiently explained.
You can find our assumptions for the models(like number of layers and type of layers) in the implementation.

