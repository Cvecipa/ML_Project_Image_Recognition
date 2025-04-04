# ML_Project_Image_Recognition
Image Recognition for cracks in walls and pavements - aiming to automate, to some degree, the process of checking structural stability. The following ReadMe will outline how this project sets about achieving this and what steps were employed to get there.
## Project Introduction
The following project aims to create and train a machine learning (henceforth ML) model able to classify an image into one of two categories, either cracked or not. 

As such a relevant dataset was selected containing images of walls being both cracked and not (consisting of over 18,000 images) which will be used to teach, train and evaluate the model. 

Throughout the course of this project both CRISP-DM (Cross-Industry Standard Process for Data Mining) and the ML Pipeline will govern the steps and process plan undertaken as seen below.


<hr>
<hr>

# Table of Contents
## Section 1 - Project Review
### - Reviewing Relevant Projects
<hr>

## Section 2 - Project Structure Breakdown
### - CRISP-DM
### - My ML Pipeline
### - About The Dataset
### - Setting Up My Workspace
<hr>

## Section 3 - Jupyter Notebooks
### - Data Collection
### - Data Visualisation
### - Modelling And Evaluation
<hr>

## Section 4 - Project Deployment
### - Unavailable
<hr>

## Section 5 - Appendices
### - Work in Progress

<hr>
<hr>

# Project Review and Abstract
Abstract -> Structural defects such as cracks in walls pose risks to the safety and longevity of buildings if not identified early, traditional inspection methods are often time-consuming, costly and inconsistent and with growing demand for rapid assessments, automated approaches using machine learning and image recognition are gaining interest. This project explores the use of computer vision to detect wall cracks from images - aiming to reduce reliance on manual inspections and improve the speed and accuracy of defect identification.
A supervised learning approach is applied using a labeled dataset of wall images categorized as cracked or non-cracked; the images are preprocessed to identify visual patterns that may indicate structural defects, however, data augmentation is not used as the dataset is sufficiently large to train the model effectively without it. The dataset is split into training, validation and testing sets to ensure reliable performance evaluation. TensorFlow is used, among other vital libraries to develop and train the model and the development process is structured across three Jupyter notebooks covering data collection, visualization and model training and evaluation.
As the project is in its early stages I am currently preparing the dataset and building the initial version of the model. Once implemented, the CNN will be trained and evaluated using metrics such as accuracy, precision and recall and a prediction accuracy above 50% is considered useful in this context as even moderate performance can support users by flagging potential defects for further inspection - future work will focus on tuning the model and assessing its consistency.
The intended outcome is a working image classification tool to support construction and maintenance professionals, by automating the initial detection of cracks the system could improve inspection efficiency and help prioritize areas requiring closer attention. As development progresses, I plan to explore the model’s practical value in real-world scenarios.

# Project Breakdown Structure
### CRISP-DM 

<hr>

Business Understanding -> The objective of this project is to develop a ML learning based image classifier capable of automatically detecting cracks in various structural elements; the solution aims to assist construction and maintenance experts in identifying structural defects as early as possible, reducing not only cost for the business but also inspection time on the whole and improving structure safety. Success of the project will be defined by not only the model accuracy and recall in detecting cracks, but also its ability the classify new, unseen images in the form of different structural elements (ie. paths and roads). As such, the following are the business aims of this project: 
1. Enable Predictive Maintenance 
-> By indentifying cracks early on the business is able to prioritise at risk areas and allocate its resources most effectively.

2. Improving Precision to Cost Ratio
-> By somewhat automating time consuming and labour intensive inspections done manually the business is able to both accelerate the inspection process while reducing require staff and prevent large scale structural issues by method of prevention.

3. Improving Safety and Risk Management
-> By detecting defects early the business can mitigate risks posed by structural instability and reduce or avoid dangerous situations.

In this project, a prediction probability threshold of 50% will be applied, with values exceeding this threshold deemed statistically significant (ie. a successful model).

<hr>

Data Understanding -> In preparation for the project a number of datasets were reviewed on Kaggle, typically containing just cracked or non-cracked images of various structural elements. The deciding factor for deciding upon a dataset was the size and variety of the given dataset and as such the chosen dataset for this project is the [Structural Defects Network](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images?select=Pavements) dataset. Given its overall total of 56,000 images and 18,000 of those being of the primary structural element this project will cover, walls, it seemed the natural choice; with an additional 38,000 images that we can further test our models with it fulfills all requirements for a dataset. Some additional things to note: all images were pre-sized by the owner of the dataset as 256 x 256 which will streamline image pre-processing; images contain a wide variety of crack size in the range of 0.06-25mm allowing for a diverse set of photos and the photos have already been classified as cracked and non-cracked. 

<hr>

Data Preparation -> To prepare the dataset for the project we employed some generic steps for cleaning the dataset; initially just checking for and removing any non-images from the dataset and following which a manual check of the images to verify the set is clean and ready for use. The dataset is then split into three partitions - a training, validation and test set in the ratio of 0.7, 0.1 and 0.2 respectively. 

<hr>

Modelling -> *TBC

<hr>

Evaluation -> *TBC

<hr>

Deployment -> *TBC

<hr> 

### My ML Pipeline

<hr>

Introduction: The following explores the process plan adopted in this project, explaining step-by-step what was done and what this project set out to achieve.

<hr>

Data Collection -> 
1. => Collecting dataset containing images of cracked and non-cracked structural surfaces.
2. => Ensuring variety within the dataset, key focal points being things such as lighting; angle of the crack on the surface; external environmental conditions that could be present.
3. => Checking dataset to remove any non-image files and removing them. 
4. => Labeling images as cracked or non-cracked. 
5. => A final visual check of all aforementioned steps to ensure accuracy.
6. => Splitting the checked dataset into the desired subsets of train, test and validation. 

These steps can be seen in the first of the jupyter notebooks in this project titled 1_data_collection. 

<hr>

Data Preprocessing and Augmentation ->


<hr>

Feature Extraction ->


<hr>

Model Selection and Training ->


<hr>

Evaluation ->


<hr>

Hyperparameters ->


<hr>

Deployment


<hr>

Future Monitoring and Potential Improvements ->


<hr>

## About the Dataset
The chosen dataset was decided upon for a number of reasons, primarily the notably large size of the dataset and the subdivisions within it. The model will be tested on cracked and non-cracked images of walls but the dataset contains also images of cracked and non-cracked pavements and decks meaning after training and evaluation the best model can also be tested for transferability on these other structures. Additionally, the large file count, 56000, will save some effort upfront as when training the models it can be observed that the bigger the better - having a smaller dataset could impose issues unless the dataset is augmented to provide more images, ie, the same images alongside warped or rotated versions to give the dataset depth. 

<hr>

## Setting Up My Workspace
After deciding project aims and finding a suitable dataset, setting the workspace was next. Consisting mostly of creating the repo itself and including within this a few key elements for a ML project - a .gitignore file for sensitive data to not be pushed to git; the jupyter notebooks I would be working out of organised as: 1_data_collection, 2_data_visualisation and 3_modelling_and_evaluation; and a README file to document the project itself alongside various other instructions. 

<hr>

# Jupyter Notebooks

<hr>

## Data Collection
To begin the first notebook I set out the aims of the notebook, namely finding a suitable dataset; downloading and preprocessing said dataset; and finally splitting it into three subsets for training later, these subsets being a test, train and validation set. 

<hr>

## Data Visualisation
In this notebook, I set out the aims of exploring and visualizing the dataset to better understand its characteristics. This involves generating summary statistics, examining class distributions, and identifying potential issues such as class imbalance or missing values. I also create various plots and visual representations to gain insights into the data before proceeding to the modeling phase.

<hr>

## Modelling and Evaluation
In this final notebook, I set out the aims of training, evaluating, and refining a machine learning model using the preprocessed dataset. I begin by selecting an appropriate model architecture, followed by training it on the training set and validating its performance. I then evaluate the model using various metrics and, if necessary, fine-tune it to improve accuracy and generalizability. Finally, I test the model on the test set to assess its real-world performance.

<hr>