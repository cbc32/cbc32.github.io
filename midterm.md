# Project Midterm Report

> [Proposal](https://cbc32.github.io/proposal.html)
> [Final Report](https://cbc32.github.io)

## Contribution Table

<table>
  <tr><th>Task</th><th>Member</th></tr>
  <tr><td>Implement Background Removal</td> <td>Aiden</td></tr>
  <tr><td>Implement Brightness/Contrast Processing</td> <td>Divyam</td></tr>
  <tr><td>Implement Augmentation</td> <td>Ray, Charlie</td></tr>
  <tr><td>Implement AlexNet Method</td> <td>Ray</td></tr>
  <tr><td>Update Background, Problem Definition, Data Collection</td> <td>Prithvi</td></tr>
  <tr><td>Add to Methods & Results</td> <td>Ray</td></tr>
  <tr><td>Github Organization</td> <td>Charlie</td></tr>
</table>


## Background

Deaf and hard-of-hearing people face issues with communication every day, relying only on sign language. However, with barely any education in schools, people without these disabilities often fail to interpret sign language. This issue is very common for Bengali Sign Language (BdSL) users due to the lack of research in the area and the number of BdSL interpreters. We aim to mitigate this problem. Our system would rely on images of bare hands in varied positions, providing a natural experience for users.


## Problem Definition

Beginners may find it challenging to differentiate between the Bengali sign language alphabet's symbolic variations. A program that generates a real-time analysis of hand movement will help consumers to increase the efficiency and accuracy of their learning process while giving them valuable input to improve the experience. Additionally, we believe that people who do not understand the Bengali sign language will be able to converse with the users of this language easily. We aim to build a computer vision-based system for the automatic recognition of Bengali sign languages to mitigate this problem.

## Data Collection

_Bengali Sign Language Dataset._ (2020, March 8). Kaggle. Retrieved October 3, 2022, from https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset

The data was published on Kaggle with a [paper in IEEE](https://www.researchgate.net/publication/337366713_Image-based_Bengali_Sign_Language_Alphabet_Recognition_for_Deaf_and_Dumb_Community). The research was sponsored by the Bangladesh University of Engineering and Technology  in collaboration with the National Federation of the Deaf (Bangladesh). 

We used the dataset directly in a Kaggle notebook. We also downloaded it and uploaded it to GitHub. The dataset contains 12,581 different hand images belonging 38 BdSL signs.

## Methods

### Preprocessing: Background and Shadow Removal
Most of the images we take in real life contain various backgrounds and shadows. To reflect this condition, our data set contains various backgrounds and shadows. For this reason, to increase the accuracy of data set analysis, it is essential for us to remove the background and shadows. Background and shadow removal removes any kind of objects, lighting, and shadows that we do not need. To remove the background and shadows, we used the rembg tool. Unlike other background and shadow removal tools that use simple mathematical calculations, the rembg tool produces much more accurate results by using a neural network-based U2Net. However, the rembg tool does not output 100% accuracy. If you look at the images below, you can see that the image 1 has the background and shadows removed well, but the image 2 has a little background left. However, the result of the rembg tool does not change that it produces much more accurate results than the results of other background and shadow removal tools. As a result of the inspection, the rembg tool completely removed the background and shadows of more than 98% of the images.

![Example 1 Before](/assets/ex1_before.jpg)
![Example 1 After](/assets/ex1_after.jpg)

![Example 2 Before](/assets/ex2_before.jpg)
![Example 2 After](/assets/ex2_after.jpg)

### Preprocessing: Brightness and Contrast Enhancer
To train our model properly based on the dataset, we increased the brightness of the images and enhanced contrasts for the same. This allows for the model to correctly identify and differentiate the different alphabets. We used the Python Imaging Library to perform these tasks on our dataset.


### Preprocessing: Data Augmentation
To increase the size of our dataset and reduce the possibility for overfitting, we add Keras preprocessing layers that randomly rotate and flip images horizontally/vertically. We believe adding random rotation and flipping makes sense in the real world since if we were to deploy this computer vision model to read Bengali Sign Language, it is possible that the image captured by the camera could be from different angles and sometimes even upside down.

### Models
We will use two model types, convolutional neural network (CNN) and support vector machine (SVM). For CNN, we will use famous architectures such as AlexNet (Krizhevsky et al., 2017), VGG16 (Simonyan et al., 2014), ResNet50 (He et al., 2016), and MobileNet (Howard et al., 2017). We use Tensorflow Keras to train these CNN models. SVM could be implemented using scikit-learn. A series of narrowing GridSearchCV could tune the SVM.

## Results

To evaluate the performance of the models, we measured their accuracy and loss. We also measured their average f1 and roc auc score across all classes, which can be calculated by passing in `macro` as argument to the average parameter in the associated [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics). Additionally, we looked at the confusion matrix to inspect the classification result for each class in more detail. 

So far we have trained several models from scratch using the AlexNet (Krizhevsky et al., 2017) architecture. We used [SGD optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (learning rate 0.001, momentum 0.9) and stopped training after no improvement in training loss for 10 epochs and restored the weights from the epoch with minimal loss (use [tf.keras.callbacks.EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)). We didn’t use adam optimizer since when trying adam optimizer, we found that the model often got stuck and training loss wasn’t decreasing. 

The following table compares the results of training with different data preprocessing techniques and data augmentation settings. 

![Results Table](/assets/midterm_results_table.png)

Overall, our result indicates that AlexNet is able to achieve around 90% accuracy regardless of the data preprocessing techniques or data augmentation settings. Compared to the results from Rafi et al. (2019), where they achieved a test accuracy of 89.6% using an architecture similar to VGG19, we were able to achieve similar results with a simpler architecture as AlexNet is only about ⅓ the size of VGG19 in terms of number of parameters (AlexNet: 46,905,446 parameters; VGG19: 139,725,926 parameters).

Comparing different data preprocessing techniques, background removal seems to be the most effective in terms of improving testing accuracy while enhancing contrasts doesn’t seem to help much and even has an adverse effect. After introducing data augmentation to combat overfitting, we observed that the training loss increased, and the testing loss decreased in cases where we applied background removal. However, testing loss increased in other cases.  Overall, applying background removal plus data augmentation seems to give us the best result. And we will use this setting to train future models.

The following is the confusion matrix from the AlexNet model trained on background removed dataset with data augmentation.

![Confusion Matrix](/assets/confusion_matrix_mid_term_report.png)

![Confusion Matrix Example](/assets/confusion_matrix_example.png)

### Next Steps
- We want to see if we are able to train even simpler models such as MobileNet (Howard et al., 2017) architecture that can be deployed on mobile devices while achieving a similar level of accuracy.
- We want to see if training bigger models (like imageNet) would give us even higher testing accuracy.
- We want to try traditional machine learning techniques such as SVM and see how they perform.


## References

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017). Imagenet classification with deep convolutional neural networks. Communications of the ACM, 60(6), 84-90.
https://dl.acm.org/doi/abs/10.1145/3065386

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv.
https://arxiv.org/abs/1704.04861

Rafi, A. M., Nawal, N., Bayev, N. S. N., Nima, L., Shahnaz, C., & Fattah, S. A. (2019, October). Image-based bengali sign language alphabet recognition for deaf and dumb community. In 2019 IEEE global humanitarian technology conference (GHTC) (pp. 1-7). IEEE.
https://ieeexplore.ieee.org/abstract/document/9033031/?casa_token=fqX0wWHAUfcAAAAA:UGYwm_6jlOVwANq9kCi146mVqPfanS5w47Gp9tPwh3Eh-yCcakPnY5uakjePHMkMYtob2Yfdxw

Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv. https://arxiv.org/abs/1409.1556
