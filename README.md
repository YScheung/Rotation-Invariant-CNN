# Rotation Invariant CNN network 

## Introduction: 
One of the major weaknesses of CNN is that it is not rotation invariant. CNN couldn’t detect images that are rotated. To tackle the problem, most support data augmentation, feeding normal and rotated images to the network, yet the training time and resources needed will be hugely increased. 

In this project, I would like to create a CNN network that could classify rotated photos of dogs and cats (by 180 degrees) without the need of data augmentation and with a CNN network that is trained on unrotated images 

## Main ideas: 
After the CNN model is trained on unrotated images: 
1) Rotate CNN filters by 180 degrees to detect features in rotated images
2) Rotate the resulting feature map back by 180 degrees (resulting feature map for rotated and unrotated images are now the same)
3) Input the rotated feature map to the fully connected CNN network for classification.


## Results and conclusion 

|   Accuracy (Normal image)   |   Accuracy (Rotated image)  |
| --------------------------- | --------------------------- |
|            85%              |             85%             |


Accuracy of model before rotation in feature maps and filters in classifying normal images is 85%
Accuracy of model after rotation in feature maps and filters in classifying rotated images is 85%

Most importantly, likelihood scores between the above 2 models in classifying the same image of their respective kind have a difference of less than 0.01
