# Time Series Classification - ECG Time Series using Simple AutoEncoder
The following workflow will demonstrate how to use a simple auto encoder to do deep learning in KNIME for time series classification of a ECG dataset.

###### Dataset Link
ECG Classification: http://www.timeseriesclassification.com/description.php?Dataset=ECG5000   <br/>

###### Workflow Link
ECG Classification Workflow: https://tinyurl.com/2854dwxp <br/>

# Binary & Multilabel ECG Time Series Classification - Deep Learning 
![image](https://user-images.githubusercontent.com/94952931/156868395-36f0ead3-3ae8-4090-b04f-df1fcf60eadf.png)

CSV Data Snippet <br/>

The following workflow will use a simple autoencoder made by Keras layers in order to classify a dataset containing ECG Heartbeats. Each heartbeat classification consists of 140 data points, and can be classified to be either a normal heartbeat or abnormal heartbeat - of which consists a few more subcategories. The workflow shows similar models being used to do binary (normal/abnormal) classification and multilabel classification.

### Data Pre-Processing
![image](https://user-images.githubusercontent.com/94952931/156868245-7f6fa70c-7f1d-4b70-ac62-392ece6a74b4.png)

We will need to read in both the testing and training data, do some minor pre-processing and combine the 140 data points into one list (or tensor) before feeding it into the models. Using the ARFF Reader node, we will read in both the training and test data, before concatenating them into one list. We then use the missing value node to filter any rows with missing data. The first rule engine is to append a new column called “Annotation” and classify the heartbeat as either normal or anomaly.

![image](https://user-images.githubusercontent.com/94952931/156868251-212c9f00-4144-4f40-8204-75187d29b917.png)

This will now be diverged into the multi-label classification model pathway. See the Multi-label Training section for a continuation.
The second rule engine node is to replace the “target” column with either Normal or Anomaly so that we can do a binary classification. We then use the column aggregator node in order to aggregate all 140 data point columns into one list.

![image](https://user-images.githubusercontent.com/94952931/156868256-3ca708e1-8197-479a-b258-4e6eaf6ec16f.png)

### Binary Label Training
![image](https://user-images.githubusercontent.com/94952931/156868263-d791ad71-8b7a-4de2-86da-91db89552e8f.png)

The string to number node is used to convert the “target” column to integer values. We then partition the data twice in a 70-30 divide with stratified sampling and random seed.

![image](https://user-images.githubusercontent.com/94952931/156868273-b393201e-29f4-41ea-a71b-6f26252f9cc1.png)

Next, we build the CNN. We are first going to place an input layer of SHAPE = 140 and BATCH SIZE = 32. We will then put 6 Dense layers sequentially after the input layer. The dense layers each have SHAPE = 128, 64, 32, 64, 1 respectfully. Next, follow the configurations below for the Keras Network Learner and Executor nodes.

![image](https://user-images.githubusercontent.com/94952931/156868278-be1a0f21-847b-4065-bd46-c25cd8a1fe16.png)

![image](https://user-images.githubusercontent.com/94952931/156868282-dd229822-b34c-4068-a7f6-7ac417f8d82b.png)

![image](https://user-images.githubusercontent.com/94952931/156868288-95911f44-4219-4321-afda-84e5043f55a3.png)

![image](https://user-images.githubusercontent.com/94952931/156868293-255f541f-1eaf-47ce-96ba-a29e8381590e.png)

![image](https://user-images.githubusercontent.com/94952931/156868295-90e6867f-4007-4fb4-849f-55d9e277dd17.png)

Finally, we convert the output_0 column to an integer so that it can be put into the scorer configuration. 

### Multi-label Training
![image](https://user-images.githubusercontent.com/94952931/156868298-207df84b-926b-4c0d-95e8-5639a313296b.png)


We first use the One to Many node to do KNIME’s version of one-hot encoding the intended labels. Next we use the column aggregator to aggregate all the attribute columns. The CNN metanode in the multi-label training is very similar to the binary label training, with the only difference being the last dense layer’s shape is 5 instead of 1. Follow the configurations above for the Keras Network Learner and Executor.

After training, use the Many to One and String Manipulation nodes to have this output.

![image](https://user-images.githubusercontent.com/94952931/156868300-8b40c076-2f23-4ed9-bd54-fa3604026dda.png)

### Post Processing & Model Evaluation

Here we can see the confusion matrix for both the binary label and multi-label training. The binary label training has slightly higher accuracy compared to the multi-label training and this is to be expected.

![image](https://user-images.githubusercontent.com/94952931/156868321-fd32b967-7734-4580-ad7e-13484d6f7eb2.png)

Confusion Matrix for binary label training<br/>

![image](https://user-images.githubusercontent.com/94952931/156868326-cdee420e-7d9c-4fa8-a595-f536e5db655a.png)

Confusion Matrix for multi-label training<br/>

