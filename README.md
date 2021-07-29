Introduction:

One of the major weaknesses of CNN is that it is not rotation invariant. CNN couldn’t detect images that are rotated. To tackle the problem, most support data augmentation, feeding normal and rotated images to the network in the training process, yet the training time and resources needed will be hugely increased. 

In this project, I would like to create a CNN network that could be used to classify photos of dogs and cats that are rotated by 180 degrees without the need of data augmentation.   

Major idea: 

1) Rotate CNN filters by 180 degrees to detect features in rotated images
2) Rotate the resulting feature map by 180 degrees so that the feature map for both normal and rotated images are the same.
3) The feature map will then be inputted to the fully connected CNN network for classification.

