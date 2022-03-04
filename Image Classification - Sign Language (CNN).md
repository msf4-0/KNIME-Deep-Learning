# Image Classification - Sign Language using a CNN
The following workflow will demonstrate how to use a CNN to do deep learning in KNIME for image classification of a sign language dataset.

###### Dataset Link
Sign language (gesture recognition):  https://drive.google.com/file/d/1EAcId2AJefByuUvDAL_6Ee5QdWo-ABSd/view  <br/>

###### Workflow Link
Sign Language Image Recognition Workflow: https://tinyurl.com/5n97ntk2 <br/>

# Sign Language Alphabet Recognition - Deep Learning 
![image](https://user-images.githubusercontent.com/94952931/156772166-3f10ebb2-79b3-45c2-9bc7-fb0408757e40.png)
Letter for “A”<br/>
![image](https://user-images.githubusercontent.com/94952931/156772212-2bc68d25-8e94-42e7-b94b-8578972ac88a.png)
Letter for “K”<br/>

All images taken are already the same size in grayscale, but pre-processing must still be done for good practice.

The letters “J” and “Z” are omitted from the dataset as they are “moving” signs and cannot be captured as one static image.

### Image Pre-Processing
Before anything, the images from the dataset has to be cleaned and processed in a way that the model can accept it.

![image](https://user-images.githubusercontent.com/94952931/156770243-6d43fc0e-5f8c-433b-b2b6-e4ce239ef4f8.png)<br/>

The image pre-processing node is made up out of 5 nodes. Image reader, image resizer, string manipulation, one to many, and column splitter. The image reader node is quite self–explanatory, so just select all the pictures that will go into the deep learning network, which includes all test and train data sets pictures. Within the image reader’s configuration, select pixel type as “FLOAT TYPE.” Make sure all the configurations follows below. Click APPLY then OK. <br/>
![image](https://user-images.githubusercontent.com/94952931/156772629-3ba0dd70-2d96-45ec-81fe-1acd57078eef.png)

![image](https://user-images.githubusercontent.com/94952931/156770387-d526e49c-23e1-4997-968f-cf3a340ced8d.png)

Next, connect the image resizer to the previous node and ensure that the new dimension sizes are X = 80.0, Y = 80.0, and CHANNEL = 1. Resizing the images will keep the data consistent, and the channel is equal to 1 as it is a black and white image. If you are feeding RGB images, make sure to have CHANNEL = 3. The new dimension size has to be set as ABSOLUTE IMAGE SIZE, and not the default setting of RELATIVE SCALING FACTOR. 

![image](https://user-images.githubusercontent.com/94952931/156772802-0166d37e-21ab-4410-9e9c-7ac1a4d06da9.png)

![image](https://user-images.githubusercontent.com/94952931/156772845-0ca7be94-8636-4b26-9807-f3247154a86e.png)


The string manipulation node is to create class columns for each image based on the file name. As each image starts with the letter of the alphabet it belongs to followed by a unique ID number, we will take the first character of each image name and create a substring that will be appended as a string value. The string manipulation node only works when there is 1 expression. So you might need to use expressions within expressions to achieve the desired output with just one node.
To both capitalize and extract the relevant substring, put in the following line of code.

capitalize(substr($$ROWID$$,0 ,1 ))

0 is for the starting position index, and 1 is for the length of characters we want to extract. You can now choose to append a new column or replace a column containing the new class names. In our case, we want to append a new column named CLASS and check the box that says Syntax check on close. Refer to the figure to check the configurations. <br/>

![image](https://user-images.githubusercontent.com/94952931/156770503-5a486c63-2931-4f08-80c0-fb10313ca1a7.png)

The One to Many node transforms all possible values in a chosen column into a new column each. Move the Class column into the green include box using the arrows. In the configurations, select manual selection. The INCLUDE box must contain the CLASS column. An example of the output is shown below in the figure. <br/>

![image](https://user-images.githubusercontent.com/94952931/156770613-42a9ec8f-9d21-4973-a55a-2d08f1395fbf.png)

The column splitter node separates the IMAGE column from the CLASS column so that once we feed the output of this node into the scorer to verify the accuracy, it will only use the CLASS and ROW ID column as references. Refer to the figure to check the configurations.  <br/>

![image](https://user-images.githubusercontent.com/94952931/156770687-bbb74394-7263-49a0-9121-e9e9416b1f3d.png)

### Partitioning
Partitioning the data into test, train, and validation allows us to effectively evaluate the model’s performance during and after training.

We are going to import 2 partitioning nodes into this section so we can partition the test set from the non-test set, and the training and validation sets from the non-test set. We do an 80/20 partition first by selecting RELATIVE[%] = 80. We then select STRATIFIED SAMPLING and check the box that says use a random seed. Here you can put any number as it does not matter. Once you have done all the configurations, you can use the same configurations for the next partitioning node, making sure that the output of the first node (top) goes into the input of the second partitioning node. Refer to the figures to check the configurations. 
 
![image](https://user-images.githubusercontent.com/94952931/156770825-f9027ff4-046c-4119-877e-89cc3918caf9.png)

 ![image](https://user-images.githubusercontent.com/94952931/156770841-3e5c6fbb-6b96-4c8b-880f-112da82f3433.png)

### Building a Neural Network
To construct the convolutional neural network (CNN), we refer to Anson’s Github to construct the layers. The layout of the neural network is attached below in Figure X.xx. We will use a mix of the following Keras layers.
1.	Keras Input
2.	Keras Convolutional 2D Layer
3.	Keras Max Pooling 2D Layer
4.	Keras Flatten Layer
5.	Keras Dense Layer

![image](https://user-images.githubusercontent.com/94952931/156770910-5d59ac20-dcf3-40cf-a7bb-8bd5d9bbfe8c.png)

We first place the INPUT layer with the following configurations in Figure X.xx. The shape 80,80,1 comes from the size of the image and the channel, so if you are doing this for your own dataset, make sure the shape of the input layer matches the image size. Check the batch size box and key in 32 for now, this can be changed later on but I highly recommend you to keep it at 32.

![image](https://user-images.githubusercontent.com/94952931/156770943-1f85748c-5539-4356-a955-5a43227b3854.png)

The first CONVOLUTION 2D layer will have FILTER = 32, KERNEL SIZE = (3,3), STRIDES = (1,1), PADDING = SAME, and ACTIVATION FUNCTION = ReLU. All these configurations will except the FILTER remain the same throughout the neural network. The second CONVOLUTION 2D layer will have a configuration of FILTER = 64, and all other settings same as the previous convolution layer. The third CONVOLUTION 2D layer will have a configuration of FILTER = 128, and the fourth layer will have FILTER = 256.

![image](https://user-images.githubusercontent.com/94952931/156770975-386e06ed-dae1-4a5a-bda7-7d8c75c237e0.png)

The MAX POOLING 2D layer will have a POOL SIZE = (2,2), STRIDES = (2,2) and PADDING = VALID. All configurations will remain the same throughout the neural network.

![image](https://user-images.githubusercontent.com/94952931/156771005-8467817c-b6bf-4dad-9a07-7c06486cef17.png)

The FLATTEN layer has no configuration to change so we can leave it as so. The first DENSE layer has UNITS = 256, and an activation function of ReLU. The last layer which is the 2nd DENSE layer, will need to have the UNITS amount equal to the number of classes we have which is 24 for this example. If you are using your own dataset, make sure to change this last layer before executing the deep learning network.

### Training a Neural Network
![image](https://user-images.githubusercontent.com/94952931/156771058-d3300e74-057d-4c5f-8716-42b7bcb745a1.png)

Our network training section consists of 3 parts; the DL Network Metanode which we have previously configured, the Keras Network Leaner and the Keras Network Executor. Within the Keras Network Leaner configurations, under the input data tab,  make sure to select CONVERSION = FROM IMAGE, and include the IMAGE column in the green box. Under the target data tab, select CONVERSION = FROM NUMBER(DOUBLE) and include all the classes of your dataset in the green box. Below that select STANDARD LOSS FUNCTION = CATEGORICAL CROSS ENTROPY. The accuracy and loss of your Network learner will depend heavily on the loss function, so make sure to select the correct one depending on your dataset. Under the options tab, select the number of EPOCHS = 10. Check the SHUFFLE TRAINING DATA BEFORE EACH EPOCH and USE RANDOM SEED boxes. Choose OPTIMIZER = ADAM, and click Apply then ok. Refer to the figures to check the configurations. 

![image](https://user-images.githubusercontent.com/94952931/156771117-afcaea9a-bb42-48f3-8018-169a15ba759d.png)

![image](https://user-images.githubusercontent.com/94952931/156771127-e9b0f414-9d67-4e04-95d0-15ae32535307.png)

Under the options tab in the Keras Network Executor, select CONVERSION = FROM IMAGE, and make sure the Image column is inside the green box.  Below that in the Outputs section, select CONVERSION = TO NUMBER (DOUBLE) and click Apply. Optionally, you can add a prefix if you wish for the new appended column.

![image](https://user-images.githubusercontent.com/94952931/156771160-3882d34d-c39e-45d9-8e1d-5c05b7dd7d8b.png)

Once all this is done, execute all nodes. Depending on the GPU on your device, training the data set can take up to 5 minutes. If you wish to view how the training is going, select the VIEW : LEARNING MONITOR option by right-clicking the Keras Network Leaner. Here you can view the training and validation data loss/accuracy. If all the configurations are correct and optimised for the dataset, the training and validation accuracy graph should go up exponentially till it reaches a value of 1.

### Post Processing & Model Evaluation
![image](https://user-images.githubusercontent.com/94952931/156771452-98e2978e-d172-4256-829e-5282920452c4.png)

The many to one node will be used to append a PREDICTION column and list down the probabilities of each image into a certain class. Select INCLUDE METHOD = MAXIMUM, as this will ensure each image is classified into the class that has the highest probability. You can choose to check KEEP ORIGINAL COLUMNS, but if you’d like a cleaner output then leave it unchecked.

![image](https://user-images.githubusercontent.com/94952931/156771477-cecec9fb-79f3-468e-a357-c61b8f3b7098.png)

The rule engine is used to convert the prediction column back to the original class columns. Here simply put the relevant syntax and choose to REPLACE COLUMN. Click apply then ok.

![image](https://user-images.githubusercontent.com/94952931/156771514-b53fda2b-fa27-4b06-8166-5377f0f13b31.png)

The joiner node is used to combine both the predictions and original labels. Follow the configutations below.

![image](https://user-images.githubusercontent.com/94952931/156771553-034a299f-10e4-4634-a0ff-a73110dc3374.png)

![image](https://user-images.githubusercontent.com/94952931/156771580-bf24f1ba-be1d-4254-b539-aaf7d097ef19.png)

![image](https://user-images.githubusercontent.com/94952931/156773148-5164e311-7e97-4ba1-ac1a-40f3f4f3a070.png)
