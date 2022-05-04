# Disaster Response Pipeline Project
### Table of Contents
1. [Instructions](#installation)
2. [Project Objectives](#objectives)
3. [File Descriptions](#files)


## 1.Instructions <a name='installation'></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## 2.Project Objectives <a name='objectives'></a>
The project main objective is to build the machine learning model to classify disaster message data from [Figture Eight](https://appen.com), and deploy the web app, where an potential emergency worker can input a new message and get relevant classification results in several categories as well as see the data visualization of original dataset.

The project consists of the following components:
1. ETL Pipeline - Load and clean data, and the store the data in a SQLite database.
2. ML Pipeline - Text processing, feature extraction with BoW and TF-IDF, and machine learning processes. This process includes the model tuning by GridSearchCV with a few additional features.
3. Flask Web App - Deployment of web app for users and visualization of original dataset using Plotly.

## 3.File Descriptions <a name="files"></a>
There are three folders for each project components.
- data
  - original dataset from Figure Eight
    - [disaster_categories.csv](data/disaster_categories.csv)
    - [disaster_messages.csv](data/disaster_messages.csv)
  - [process_data.py](data/process_data.py) - code for ETL pipleline
  - [DisasterResponse.db](data/DisasterResponse.db) - cleaned data stored in SQLite database.
