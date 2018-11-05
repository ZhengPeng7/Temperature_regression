# temperature_regression
> Put scanned hand back images into InceptionV4 to predict the inside temperature [with inside factor[add at the beginning / end] / without factor K].

---

## Input:

+ Image: large quantities of images of scanned hand back | 3d arrays.
+ K_factor: a factor which reflects the property of the indoor environment | scalars.

## Method:

### ![#1589F0](https://placehold.it/15/1589F0/000000?text=+)Network:

Based on the [Inception-v4](https://github.com/kentsommer/keras-inceptionV4), which is then modified into a regression model:

0. Inputs: {scanned_hand_back_images, indoor_factor_K}, outputs:{temperature_prediction},

1. Modification 0: Only images,
2. Modification 1: At the beginning, expand the scalar and then use the 1x1 convolution to combine the channels of images and Ks from 4 channels into 3 ones, which can be directly used into Inception-v4.
3. Modification 2: At the end, concatenate the image flow and K into some additional dense layers.

### ![#12F943](https://placehold.it/15/12F943/000000?text=+)Data format:

#### Example:

In scene 1:

K: 97.1443351260611,

Image_1:

![hand_1_1](./images/1_1_1.jpg)

![#F51234](https://placehold.it/15/F51234/000000?text=+)Results:

Now the result is sort of acceptable after **1** epoch over the whole dataset:

> Without K

![res_1](./images/inceptionV4_no_K_labels_and_preds.jpg)

![res_1](./images/inceptionV4_no_K_AE.jpg)

> With K conjuncted before the additional dense layers at the end.

![res_1](E:/researchLab/projects_cloned/temperature_regression/images/inceptionV4_tail_K_labels_and_preds.jpg)

![res_1](E:/researchLab/projects_cloned/temperature_regression/images/inceptionV4_tail_K_AE.jpg)





