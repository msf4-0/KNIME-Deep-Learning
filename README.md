# KNIME-Deep-Learning

This repository is going to contain all the relevant documentation to my KNIME workflows, which focuses on computer vision and deep learning.
Each file will contain one workflow, and documentation that will support that details the process in KNIME.
The following are links to all datasets and workflows on the KNIME HUB, of which you can then download the full workflow from.
The categories for the workflows in this repository are as follows: <br/>

1.Image Classification <br/>
2.Image Segmentation & Prediction <br/>
3.Sentiment Analysis <br/>
4.Time Series Classification<br/>
5.Time Series Forecasting <br/>

Disclaimer: Most machine learning workflows will include data augmentation as part of their project. However, KNIME has some issues with augmenting image data, therefore I have chosen to skip this step in all my workflows. Augmentation is still a very important part of machine learning and it would bode well to understand its importance, but for the very purpose of the following workflows it has not been used. The training still works fine as all the datasets are still quite large regardless.<br/>


# Dataset Links
Sign language (gesture recognition):  https://drive.google.com/file/d/1EAcId2AJefByuUvDAL_6Ee5QdWo-ABSd/view  <br/>
Skin cancer classification: https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign  <br/>
Malaria classification: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria <br/>
Breast cancer classification: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data <br/>
Brain MRI Segmentation: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation  <br/>
Drug Review Dataset: https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018  <br/>
ECG Classification: http://www.timeseriesclassification.com/description.php?Dataset=ECG5000   <br/>
Daily weather forecast (India): https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data  <br/>

# Workflow Links
CNN Meta node for skin cancer Workflow: https://tinyurl.com/2p97rckw  <br/>
Skin Cancer (Transfer Learning) Workflow: https://tinyurl.com/72br59ss  <br/>
Malaria Binary Image Classification Workflow: https://tinyurl.com/3zcva9y8  <br/>
Sign Language Image Recognition Workflow: https://tinyurl.com/5n97ntk2 <br/>
Breast Cancer Binary Classification Workflow: https://tinyurl.com/59xfxsay  <br/>
MRI Scan Segmentation & Prediction Workflow: https://tinyurl.com/yhrfs8y8 <br/>
Drug Review Sentiment Analysis Workflow: https://tinyurl.com/2p8krhx5 <br/>
ECG Classification Workflow: https://tinyurl.com/2854dwxp <br/>
Daily Weather Forecast (India) Workflow: https://tinyurl.com/2p9bywnr <br/>

# Additional Information
Some of the following workflows are inspired by Anson’s Github and will be referencing that from time to time in the document. Make sure to read through his explanations for detailed understanding and to compare the utilization of deep learning both in KNIME and Python. 

Anson’s Links -  https://docs.google.com/document/d/1rPQbKr5YXVL83wVZ9ta6qcHr3gBJJwnmXnbDZyOqo-w/edit 

I also have a google drive folder link with all the workflows I’ve worked on, these files contain the conda environment propagation node so that the user does not need to configure their own environment. They can merely run this node and all the relevant package version will be downloaded automatically into a makeshift environment for the workflow. 

Google Drive Folder Link: https://drive.google.com/drive/folders/1qUwNDe8rpB9AYclQbCvrkLpzVYO3PiHO?usp=sharing


# Acknowledgements
Thank you to Anson, who has helped me a lot in the beginning of my deep learning journey and provided me with Python examples. <br/>
Thank you to Warren (Nien Loong Loo, Ph.D.) for helping me in fixing my workflows and answering all my deep learning/machine learning questions. <br/>
Thank you to Dr Chua Wen Shyan who has given me the opportunity for this internship.<br/>
