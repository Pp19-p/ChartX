# ChartX
ChartX is a workflow for extracting bar chart data from scientific literature.

# Evns
CUDA12.8,Python3.8;
numpy 1.24.4
keras 2.9.0
matplotlib 3.5.3
opencv 4.10.0
pandas 1.5.3
scikit-learn 0.24.2
tensorflow 2.13.1
yaml 0.2.5

# Dataset
Public dataset URL:
https://codeload.github.com/paulorscj/Chart-dataset/zip/refs/heads/master

http://vis.stanford.edu/papers/revision

https://chartinfo.github.io/toolsanddata_2022.html

# Chart Classification
Before training the model, please preprocess your dataset, perform classification, split the dataset into training and validation sets, run ImageName.py, save the output results to the corresponding directory, modify the relevant paths in ChartClassification.ipynb, replace them with your_dir, and run ChartClassification.ipynb to start training the Xception model.

# Axis Positioning and Subgraph Segmentation
There are multiple subgraphs that can be processed. For a single chart image, no additional processing is required.

# Chart Element Detection
We recommend using LabelImg for annotation. The annotation interface is shown below:
<img width="2544" height="1606" alt="Lamblimg" src="https://github.com/user-attachments/assets/acd9f4cb-c664-424c-b5ce-603151d1d07a" />
(Ensure annotated results are saved in YOLO format(red box area). Additionally, save the original annotated images and the generated annotations separately.)
Divide the annotated dataset and corresponding annotations into training and validation sets(8:2), place them in your_dir, modify the relevant paths in detection.py, then run the script to begin training the YOLOv11 model for object detection.

# Text 
Access link for Panddle OCR: https://github.com/PaddlePaddle/PaddleOCR
Based on Algorithm 1 and specific requirements, Panddle OCR can be trained or directly invoked.

# Color
K-Means clustering and DBSCAN clustering are combined to extract chart colors. K-Means clustering primarily extracts the colors of detected Legend colors and Bars, while DBSCAN clustering is used to extract colors for Legend labels when Legend colors are not detected. This prevents Bars from being misclassified due to omitted colors.

# Data Extraction-Mapping
Algorithm 2 performs the final data integration, mapping the extracted content from the preceding steps based on pixel coordinates and real-world coordinates. It reconstructs new visualization charts and saves the extracted content to a .csv file, enabling seed chart extraction—the complete ChartX workflow.
