# ML_Project_Image_Recognition
Image Recognition for cracks in walls and pavements - aiming to automate the process of checking structural stability.
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

# Project Review
*Finish word doc and commit here

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


<hr>

## Modelling and Evaluation


<hr>