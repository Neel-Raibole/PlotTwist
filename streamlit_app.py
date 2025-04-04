#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
from ultralytics import YOLO
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from paddleocr import PaddleOCR

from datasets import load_dataset, concatenate_datasets
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
import torch

from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

import streamlit as st
from Capstone_Project_Code import process_line_chart_without_data_points_full, process_line_chart_with_data_points_full
from Capstone_Project_Code import process_bar_chart_without_data_points_full, process_bar_chart_with_data_points_full
from Capstone_Project_Code import process_full_pie_chart


# In[2]:


# Calling YOLO Model
# Path to your trained model weights and test image
model_path = "run2/detect/final_model_continued/weights/best.pt"
# Load the trained model
model = YOLO(model_path)


# In[6]:


st.title("Chart Analyzer App")


# In[7]:


# Image upload box
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Dropdown for chart type selection
chart_type = st.selectbox("Select Chart Type", ["Pie Chart", "Line Chart", "Bar Chart"])
# Debugging Statement: st.write("Chart Type selected:", chart_type)

# Dropdown for data points option
data_points_option = st.selectbox("Select Data Points Option", ["With Data Points", "Without Data Points"])
# Debugging Statement: st.write("Data Points Option selected:", data_points_option)


# In[8]:


# Submit button to trigger processing
if st.button("Submit"):
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption = "Uploaded Image", use_column_width = True)
        
        # Save the uploaded file temporarily
        image_path = "temp_image.png"
        image.save(image_path)
        
        # Nested if-else to call the appropriate function based on dropdown selections
        result = None
        
        if chart_type == "Pie Chart":
            if data_points_option == "With Data Points":
                result = process_full_pie_chart(image_path)
            
            else:  # Without Data Points
                result = process_full_pie_chart(image_path)

        
        elif chart_type == "Line Chart":
            if data_points_option == "With Data Points":
                result = process_line_chart_with_data_points_full(image_path)
            
            else:  # Without Data Points
                result = process_line_chart_without_data_points_full(image_path)

        
        elif chart_type == "Bar Chart":
            if data_points_option == "With Data Points":
                result = process_bar_chart_with_data_points_full(image_path)
            
            else:  # Without Data Points
                result = process_bar_chart_without_data_points_full(image_path)
        
        # Display output area with results
        st.write("### Extracted Chart Data:")
        st.write(result)
    else:
        st.error("Please upload an image.")

