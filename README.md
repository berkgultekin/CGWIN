# CGWIN

Paper Information :
    Generative Well-intentioned Networks 
    https://papers.nips.cc/paper/9467-generative-well-intentioned-networks.pdf

Authors of code:
    Yasin Berk Gültekin
    Hasan Ali Duran
    
    
Challenges Encountered When Implementing Paper:
* There was not enough detail about how BNN was implemented. The number of layers was not specified. 
We could not fully obtain the BNN results mentioned in Paper. The BNN we have implemented makes predictions with higher scores. 
This situation caused difficulties in the exact occurrence of qualitative results.

* The biggest problem we encountered during GAN implementation was that it was not clear enough how the 
Generator and Discriminator inputs should be processed in the model. It was not specified how many layers 
or what types of layers were used. In addition, the pictures produced by the generator appeared similar to 
those given as input to the Generator. There was no explanation for how this problem was solved in paper implementation. 
The new method which is transformation loss used during the Discriminator's loss calculation was not sufficiently explained.
(You can find our assumptions for the models(like number of layers and type of layers) in the implementation.
