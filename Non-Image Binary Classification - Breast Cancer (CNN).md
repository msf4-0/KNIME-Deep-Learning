# Non-Image Classification - Breast Cancer using a CNN
The following workflow will demonstrate how to use a CNN to do deep learning in KNIME for image classification of a sign language dataset.

###### Dataset Link
Breast cancer classification: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data <br/>

###### Workflow Link
Breast Cancer Binary Classification Workflow: https://tinyurl.com/59xfxsay  <br/>

# Sign Language Alphabet Recognition - Deep Learning 
The breast cancer data is a csv file with 32 columns containing important information about each cancer tumour. The diagnosis column contains the labels, either M for malignant or B for benign.<br/>

Class distribution: 357 benign, 212 malignant.

![image](https://user-images.githubusercontent.com/94952931/156778449-465c3dd2-aedd-483d-887f-a675c76bb1c0.png)

### Data Acquisition and Pre-Processing

![image](https://user-images.githubusercontent.com/94952931/156778573-f8527fc0-1e9d-46b6-9959-80738eafd45b.png)

Use the CSV Reader nodes to read both the data file. 

![image](https://user-images.githubusercontent.com/94952931/156778619-9e3a48a7-99c9-4865-8abc-b6df6d2f835d.png)

Use the missing value node to filter out any rows that might have missing data. Alternatively, you can fill in missing data by using maximum, mean, median, minimum, etc. The method you choose will depend on your dataset type.

![image](https://user-images.githubusercontent.com/94952931/156778710-3936b62d-c618-496e-b79a-64fe01ee7881.png)

Under the column settings tab, you can also individually select what will happen to missing values in each column.
For now, we are just going to remove the row if there are any missing values.

The category to number node works similar to the many to one node, it will convert all the different labels in one column to a number.
We pass the diagnosis column into this node to obtain M = 0, B = 1.

![image](https://user-images.githubusercontent.com/94952931/156778990-c01b6f48-b45a-4242-bfb5-aac2c9912819.png)

### Partitioning
Partitioning the data into test, train, and validation allows us to effectively evaluate the model’s performance during and after training.

![image](https://user-images.githubusercontent.com/94952931/156779062-09f4668c-747c-4c09-b09e-715b0bd7a50e.png)

How we partition the data is also very important. Depending on the type of data set, we may need to use stratified sampling or we might need to take from the top.
“Use random seed” is selected to specify a seed for random number generation for the partitioning. Setting this option results in the same records being assigned to the same set on successive runs.

![image](https://user-images.githubusercontent.com/94952931/156779121-98743f8a-be27-4fbb-ac0c-d1ce6ccb9908.png)

We are going to partition the data 70-30 using stratified sampling from the diagnosis column. The column that we take the stratified sampling from needs to be a string column. 

### Convolutional Neural Network (CNN)
![image](https://user-images.githubusercontent.com/94952931/156779228-12384597-a5f8-4c77-a4e6-23cc5938a02a.png)

The input layer will have the same shape as the amount of columns in the csv data file.

![image](https://user-images.githubusercontent.com/94952931/156779349-4aefe74c-037f-485c-a406-bc69c6043f77.png)

![image](https://user-images.githubusercontent.com/94952931/156779270-8d7c036b-01e6-4b43-af6c-90eccb0cad6d.png)

The convolution block will be made out of a dense layer, batch normalization layer and dropout layer.
The first block will have a dense layer of 256 units.

![image](https://user-images.githubusercontent.com/94952931/156779381-c6eacb2d-70bc-4137-a8c3-04389ad3a35f.png)

![image](https://user-images.githubusercontent.com/94952931/156779298-7e6e5745-8bdc-45b9-adf5-f8adaae1a356.png)

The batch normalization layer will have an axis of -1 and tick “center” and “scale.”

![image](https://user-images.githubusercontent.com/94952931/156779437-82e28379-6a8d-4051-9765-ba150b29516c.png)

![image](https://user-images.githubusercontent.com/94952931/156779451-1d7054ea-13a8-4213-8ad0-dc6d6ef18e1a.png)

The second convolutional block has the same components as the first block, with almost all the same configurations. For the dense layer in the 2nd convolutional block, you can either keep it 256 or decrease it to 128.

The final dense layer will have an activation function of sigmoid and 1 unit, as we only have one classification problem.

![image](https://user-images.githubusercontent.com/94952931/156779489-e0881617-4f9f-4271-a408-e0994d5da858.png)

![image](https://user-images.githubusercontent.com/94952931/156779532-d9419d5f-08cb-4787-81d4-6cc162bfdd68.png)

You can use the DL Python Network Editor node in order to check for a summary of the model, by connecting this to the end of the CNN. It will show the total trainable and non-trainable parameters.

![image](https://user-images.githubusercontent.com/94952931/156779629-de6d4207-71b3-4614-a500-556e3671e047.png)

![image](https://user-images.githubusercontent.com/94952931/156779640-178151df-9f2c-4330-a509-0cad6a94eabe.png)

### Training & Evaluation
![image](https://user-images.githubusercontent.com/94952931/156779684-4896be2a-b797-4891-bf4d-4caf90fb20fb.png)

We are going to use mean absolute error as our standard loss function to get a forecast of the predicted mean temperature.

![image](https://user-images.githubusercontent.com/94952931/156779721-5bed9650-77aa-492f-87b4-61ed41bc5c67.png)

![image](https://user-images.githubusercontent.com/94952931/156779745-4ee40f76-cf52-4fd6-9171-ae30d6b75455.png)

The epoch required for the deep learning model will vary according to the application, so it is best to try with smaller numbers first as training can take a very long time. But for now, let’s try using an epoch of 100.
When possible, keep the random seed number used throughout the workflow the same, and although it is preferable to use Adam as the optimizer feel free to experiment and use other optimizers.

![image](https://user-images.githubusercontent.com/94952931/156779783-29ad4082-dd04-4dba-a33f-511f7f870654.png)

This executor node will take in the partitioned test set and use the model that was just trained to fit the new data.
Follow the configurations in the image.

![image](https://user-images.githubusercontent.com/94952931/156779849-b489950f-5df6-4f90-8a78-1dc41b5dfba1.png)

![image](https://user-images.githubusercontent.com/94952931/156779857-7a7a9915-dfc0-4245-9f84-eeb5f18d7f58.png)

The output of the Keras Network Executor would look something like this. The original labels are under the diagnosis column, and the output_0 column is the probability of the breast cancer data being Benign.

![image](https://user-images.githubusercontent.com/94952931/156779893-f0fe6e77-6cfe-49b9-9828-7fad49885717.png)

If the predicted_0 column has a less than 0.5 probability, then the rule engine will append a prediction column with the input as M for malignant, otherwise it is B for benign.

![image](https://user-images.githubusercontent.com/94952931/156779941-fe8c3b61-33fa-462a-affc-c4b4cb0d5e72.png)

![image](https://user-images.githubusercontent.com/94952931/156779951-25a17536-acd2-4b61-8cda-5594babcf1fa.png)

The scorer will use the selected columns to determine the model’s accuracy. Follow the configurations. To see the confusion matrix, right click on the node and select “confusion matrix.”

![image](https://user-images.githubusercontent.com/94952931/156779975-91687008-c1dc-442b-b589-c0b901e7b19d.png)

![image](https://user-images.githubusercontent.com/94952931/156779992-19526ca0-edc1-463d-911c-a7877b1630b2.png)


