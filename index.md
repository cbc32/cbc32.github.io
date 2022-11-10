# Project Midterm Report

> [Proposal](https://cbc32.github.io/proposal.html)

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

![Results Table](/assets/midterm_results_table.png)

## References

Amor, E. (2021, March 12). _4 CNN Networks Every Machine Learning Engineer Should Know._ TOPBOTS. Retrieved October 3, 2022, from https://www.topbots.com/important-cnn-architectures/.

_Bengali Sign Language Recognition Using Deep Convolutional Neural Network._ (2018, June 1). IEEE Conference Publication. Retrieved October 3, 2022, from https://ieeexplore.ieee.org/document/8640962.

Goonewardana, H. (2019, January 14). _Evaluating Multi-Class Classifiers._ Apprentice Journal. Retrieved October 3, 2022, from https://medium.com/apprentice-journal/evaluating-multi-class-classifiers-12b2946e755b.

Rafi, A., & Nawal, N. (2019, October) _Image-based Bengali Sign Language Alphabet Recognition for Deaf and Dumb Community._ ResearchGate. Retrieved October 3, 2022, from https://www.researchgate.net/publication/337366713_Image-based_Bengali_Sign_Language_Alphabet_Recognition_for_Deaf_and_Dumb_Community.
