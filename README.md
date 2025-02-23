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

Data Preparation -> 
