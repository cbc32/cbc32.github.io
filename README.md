# Group 37 Project Proposal: BdSL
- Introduction/Background

Deaf and hard-of-hearing people face issues with communication every day, relying only on sign language. However, with almost no teaching in school, people without these disabilities often fail to interpret sign language. This issue is very common for Bengali Sign Language (BdSL) users due to the lack of research and BdSL interpreters. We aim to mitigate this problem. Our system would rely on images of bare hands in varied positions, providing a natural experience for users. There are a total of 12,581 different hand signs for the 38 BdSL alphabets included in the dataset. 

- Problem Definition

Beginners may find it challenging to differentiate between the Bengali sign language alphabet's symbolic variations. A program that generates a real-time analysis of hand movement will help consumers to increase the efficiency and accuracy of their learning process while giving them valuable input to improve the experience. Additionally, we believe that people who do not understand the Bengali sign language will be able to converse with the users of this language easily. We aim to build a computer vision-based system for the automatic recognition of Bengali sign languages to mitigate this problem.

- Methods

The image data will be pre-processed to reduce the effect of backgrounds, shadows, unwanted parts, 3D rotation, and left vs. right hand on classification. We will perform image augmentation by shifting, flipping, and rotating images to increase the dataset size.

The dataset could be split into training and validation sets. Within the training set, the models will train their parameters on training sets and be evaluated on test sets for various train-test splits. The model with the best performance will be trained on the entire training set and evaluated on the validation set.

We will use two model types, convolutional neural network (CNN) and support vector machine (SVM). For CNN, we will use famous architectures such as AlexNet (Krizhevsky et al., 2017), VGG16 (Simonyan et al., 2014), and ResNet50 (He et al., 2016). SVM could be implemented using scikit-learn. A series of narrowing GridSearchCV could tune the SVM.

- Potential results and Discussion

This is a multi-class classification project. Among the models we trained, we will compare the accuracy, loss, average precision, average recall, average F1 score, and average ROC AUC, to see if certain models perform better than others and why. At the current moment, we expect CNN-based models to perform better than SVM models. Additionally, we will also generate a confusion matrix for each model as well as looking at the precision and recall for each individual BdSL alphabet. We want to know if some BdSL letters are more easily confused than others, and if so, why. We will also be checking if our models are biased under certain conditions. 

- Dataset

Bengali Sign Language Dataset. (2020, March 8). Kaggle. Retrieved October 3, 2022, from https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset

- References

Amor, E. (2021, March 12). 4 CNN Networks Every Machine Learning Engineer Should Know. TOPBOTS. Retrieved October 3, 2022, from https://www.topbots.com/important-cnn-architectures/.

Bengali Sign Language Recognition Using Deep Convolutional Neural Network. (2018, June 1). IEEE Conference Publication | IEEE Xplore. Retrieved October 3, 2022, from https://ieeexplore.ieee.org/document/8640962.

Goonewardana, H. (2019, January 14). Evaluating Multi-Class Classifiers. Apprentice Journal. Retrieved October 3, 2022, from https://medium.com/apprentice-journal/evaluating-multi-class-classifiers-12b2946e755b.

He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Communications of the ACM 60.6 (2017): 84-90. 

Research Gate Publication. Retrieved October 3, 2022, from https://www.researchgate.net/publication/337366713_Image-based_Bengali_Sign_Language_Alphabet_Recognition_for_Deaf_and_Dumb_Community
Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
