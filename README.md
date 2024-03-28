
***Harmful Brain Activity Classification with KerasCV***
https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview

Link to Kaggle Contest Source

***Objective***
The aim of this project is to classify harmful brain activity patterns in critically ill patients using EEG data.  Using EfficientNetV2, this notebook guides through the process of training a deep learning model on spectrogram images of EEG data.

***Dataset***
The dataset consists of EEG recordings transformed into spectrogram images, allowing us to use image classification techniques for identifying harmful brain activity. The data is split into training and test sets, with the training set used for model training and the test set for model evaluation.  **THE DATA IS NOT IN THE REPO**  must be run from the Notebook.


***Methodology***
Data Preparation: EEG spectrograms are loaded, preprocessed, and formatted for the deep learning model. Data loading is optimized using tf.data for efficient pipeline processing.
*Modeling*: Utilize KerasCV's EfficientNetV2 pre-trained on ImageNet for feature extraction, fine-tuning it to classify different patterns of brain activity.
*Training*: Implement a custom learning rate scheduler to optimize model training, employing techniques like MixUp and RandomCutout for data augmentation.
*Evaluation*: Use Kullback-Leibler Divergence as the loss function and the primary metric for model performance evaluation.
**Results**
The model's performance is evaluated using the KL Divergence metric

***How to Run***
Install the necessary libraries using pip install.
Ensure data sources are correctly linked and accessible in the /kaggle/input directory.
  for this run the first part of notebook, ports over dataset from Kaggle
Run the notebook cells sequentially, from data loading to model inference.


**REFERENCES**
HMS-HBAC: ResNet34d Baseline [Training]
EfficientNetB2 Starter - [LB 0.57]
