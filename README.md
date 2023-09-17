# Real-Time-Facial-Emotion-Recognition

This Machine Learning project is the implementation of real-time facial expression/emotion recognition using machine learning. The model is trained on a pre-existing facial expression dataset from KAGGLE and utilizes Tensorflow, Keras, OpenCV, and Scilet. The model also utilizes the pre-loaded scripts that have 62% accuracy.


# INSTALLATION
Recommended to check requirements and proceed further
To install dependencies simply run
```
pip install -r requirements.txt
```
in an environment of your choosing.

# USAGE
Open Jupyter Notebook by using the following command:
```
jupyter notebook
```
2. Look up for "trainmodels.ipynb" notebook and open it.
3. Follow the step-by-step instructions
     - Load and preprocess the dataset
     - Train the machine learning model
     - Use **OpenCV** to access your webcam and initialize real-time detection.
4. Experiment with different settings, hyperparameters, and architectures to fine-tune the model for your specific use case.

# DATASET
The facial expression dataset used in this project is sourced from Kaggle. It contains labeled facial images for various emotions, such as happiness, sadness, anger, etc. 
Dataset: kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

To use your custom dataset, update the data loading and preprocessing steps in the Jupyter Notebook accordingly.

# MODEL
The machine learning model is built using TensorFlow and Keras. It consists of a  convolutional neural network (CNN) architecture optimized for facial emotion recognition. At this moment, the model has been optimized for 62% accuracy only. However, accuracy and precision will improve in later updates.

# CONTRIBUTION
Contributions or Collab are always welcome. For any questions, suggestions, or collaborations, please feel free to open an issue or create a pull request.

# LICENSE
Licensed under MIT License  - GitHub 2023-2024
