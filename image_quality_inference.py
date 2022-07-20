# -*- coding: utf-8 -*-
"""image_quality_inference.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_3RAjL8kOvZ5RvOtBmlTEyFUHoTBP7Zk
"""

#INference of image quality

!pip install -q datasets transformers

import torch

import transformers
transformers.__version__

torch.__version__

from transformers import AutoModelForImageClassification, AutoFeatureExtractor
repo_name = r"./models/DiT_model0"

feature_extractor = AutoFeatureExtractor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)

import os
os.getcwd()

from transformers import pipeline

pipe = pipeline("image-classification", "shivarama23/swin-tiny-patch4-window7-224-finetuned-image_quality")

pipe = pipeline("image-classification", 
                model=model,
                feature_extractor=feature_extractor)

from google.colab import drive

drive.mount('/content/gdrive')

import os
os.listdir('/content/gdrive/MyDrive/image_quality/test_folder')

from PIL import Image
image_folder = r'/content/gdrive/MyDrive/image_quality/test_folder'
category_list = ["good", "bad"]
predicted_labels = []
gt_labels = []
for category_ in category_list:
  image_list = os.listdir(os.path.join(image_folder, category_))
  for image_ in image_list:
    image_path = os.path.join(image_folder, category_, image_)
    image = Image.open(image_path)
    output = pipe(image)[0]['label']
    predicted_labels.append(output)
    gt_labels.append(category_)
    print('pred:', output, 'actual:', category_, 'image_name:', image_)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(gt_labels, predicted_labels, labels=category_list)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_list)

disp.plot()

