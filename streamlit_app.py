#!/usr/bin/env python
# coding: utf-8

import os
import json
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from paddleocr import PaddleOCR

from datasets import load_dataset, concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import torch

from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

from llama_cpp import Llama

import streamlit as st
from Capstone_Project_Code import (
    process_line_chart_without_data_points_full,
    process_line_chart_with_data_points_full,
    process_bar_chart_without_data_points_full,
    process_bar_chart_with_data_points_full,
    process_full_pie_chart,
    create_chart_prompt
)

# Load models
model_path = "run2/detect/final_model_continued/weights/best.pt"
model = YOLO(model_path)

llm_model_path = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
llama = Llama(model_path=llm_model_path)

# App UI
st.title("Plottwist (Chart Analyzer App)")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
chart_type = st.selectbox("Select Chart Type", ["Pie Chart", "Line Chart", "Bar Chart"])
data_points_option = st.selectbox("Select Data Points Option", ["With Data Points", "Without Data Points"])

if st.button("Submit"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_path = "temp_image.png"
        image.save(image_path)

        table_1, table_2, prompt, response, response_filtered = None, None, None, None, None

        try:
            # Routing based on chart type
            if chart_type == "Pie Chart":
                table_1, table_2 = process_full_pie_chart(image_path)

            elif chart_type == "Line Chart":
                if data_points_option == "With Data Points":
                    table_1, table_2 = process_line_chart_with_data_points_full(image_path)
                else:
                    table_1, table_2 = process_line_chart_without_data_points_full(image_path)

            elif chart_type == "Bar Chart":
                if data_points_option == "With Data Points":
                    table_1, table_2 = process_bar_chart_with_data_points_full(image_path)
                else:
                    table_1, table_2 = process_bar_chart_without_data_points_full(image_path)

            # Display extracted data
            st.subheader("Extracted Chart Data")
            st.write(table_1)

            # Generate prompt and get LLM output
            prompt = create_chart_prompt(table_1, table_2, chart_type)
            st.info("Prompt Sent to LLM:")
            st.code(prompt)

            response = llama(prompt, max_tokens=1500)
            st.success("LLM call successful.")

            # Check and extract response
            response_filtered = response.get("choices", [{}])[0].get("text", "No text returned.")
            st.markdown("### 📌 LLM Summary Output")
            with st.expander("Click to view LLM output"):
                st.markdown(str(response_filtered))

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Prompt:", prompt)
            st.write("Raw response:", response)

    else:
        st.error("Please upload an image.")