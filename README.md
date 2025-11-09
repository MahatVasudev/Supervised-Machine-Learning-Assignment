# Spatial-Temporal Forest Fire Spread Prediction

## Problem Statemnt

<img width="612" height="417" alt="image" src="https://github.com/user-attachments/assets/5f8d57a0-fb0a-498a-988f-3639fdf320f2" />

Wildfires are becoming more frequent at an alarming rate and causing severe damage to environment due to changing climatic conditions, or human caused fires.
Early prediction of fire-prone regions can significantly help in disaster preparedness and resource allocation.

The wider picture of this project is to cover all fronts of the problem, not only how environmental factors can influence the fire spread, and amplification of the fire, 
but also about the wildlife and how they are affected by it, making a system that can predict the probability of species presence in a certain space and time...

This project will be bit bigger, so I am dividing them to different versions

---

## Dataset (As of 9 November 2025)

<img width="500" height="323" alt="image" src="https://github.com/user-attachments/assets/accacb67-cd94-47d9-90e2-249bb6464e40" />

- MODIS (FIRMS Dataset) From NASA [Dataset Link](https://firms.modaps.eosdis.nasa.gov/country/) 

## Tech Used

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/38fefcf2-3271-4aed-8f5b-07ba6df92bb9" />

- Programming Language: Python
- Deep Learning: PyTorch
- Data Manipulating/Processing: Pandas, Numpy
- Data Visualization: Matplotlib, Cartopy

## Version 1: ConvLSTM

For the 1st Version i decided to work with not so complicated model, which captures both the essence of time and space, Using LSTM and Convolutional Networks

<img width="860" height="346" alt="image" src="https://github.com/user-attachments/assets/e5039718-f777-4951-a2e9-1524a1e3b0e9" />

### Process

So the Process goes as follows

#### Training Phase

- Preprocessing: Combined Each country to a single dataset by year (2020 global dataset for eg.) this gave me the ability to look at the world as a whole 
and catch patterns of how fire spreads across the world

- During the training phase: a pivot table type of data is made from bins of $0.5^{\circ}\times 0.5^{\circ}$ of lat and long of every single day with the values being the fire count
  - Making these training data on the fly is very expensive so a cache is made in .npy file, which is saved under .cache folder (hidden here)
  - The data is then standardized with what i have calculated mean and standard deviation across the years (2020,2021,2024)
- Training Strategy: The X or input tensor here is 7 days worth of data, CONVLSTM has to predict the 8th day

### Best Model Details

Current Best Model Details

CONVLSTM
- 1 LSTM Layer
- Batch Size (during training): 4
- Inputs: \[days, 1, 269, 687\]
- Outputs: \[days, 1, 269, 687\]
- Epoch: 45

### Results

<img width="1384" height="812" alt="image" src="https://github.com/user-attachments/assets/0e3d7baa-338d-4a6d-ab60-d5fed0db2a90" />

- **Real**
<img width="1125" height="653" alt="image" src="https://github.com/user-attachments/assets/3a925ac3-cc8c-4f75-8d92-a1e699847dc5" />

- **Predicted**
<img width="1092" height="592" alt="image" src="https://github.com/user-attachments/assets/79c6e2ce-45db-468f-b5f3-6cdad200ee35" />

The model is able to decently simulate the fire intensity spread, but still is incomplete
