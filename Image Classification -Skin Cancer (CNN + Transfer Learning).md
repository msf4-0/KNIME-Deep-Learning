# Image Classification - Skin Classification using CNN
The following workflow will demonstrate how to use a CNN to do deep learning in KNIME for image classification of a skin cancer dataset.

###### Dataset Link
Skin cancer classification: https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign  <br/>

###### Workflow Link
CNN Meta node for skin cancer Workflow: https://tinyurl.com/2p97rckw  <br/>
Skin Cancer (Transfer Learning) Workflow: https://tinyurl.com/72br59ss  <br/>

### Image Pre-Processing & Partitioning

![image](https://user-images.githubusercontent.com/94952931/156773742-efa293c4-73b6-4a10-a433-76e75d510849.png)

Here we have two image readers as opposed to one because the source files have already segregated the test and training data hence we have to read those two files separately. They both go through the same rule engine, where we append their class column with the following syntax. The rule engine can take multiple expressions, so this node is used to see whether the file path has the word “benign” or “malignant” in order to append a new “Class” column.

$$ROWID$$ LIKE “benign” => “B”
TRUE => “M”

In this case, the statement “TRUE => “M”” means that if the input RowID does NOT equal benign or have benign in it’s path file name, then the second statement is true and M is appended instead of B.

![image](https://user-images.githubusercontent.com/94952931/156773883-0b0e7e4b-a2ea-4e67-8ee1-4c5d3b5e97e5.png)

The category to number node takes columns with nominal data and maps every category to an integer. Use the arrows to move the Class column to the green include box and tick “append columns.” The output will show that “Benign” = 0, and “Malignant” = 1.

![image](https://user-images.githubusercontent.com/94952931/156774154-b2e6e15b-6ad6-40f8-8caa-e4ae35f5afd3.png)

We then normalize the image between 0 and 1. The reason we normalize the images is to make the model converge faster. When the data is not normalized, the shared weights of the network have different calibrations for different features, which can make the cost function to converge very slowly and ineffectively. We put in the following expression, 

$Image$/255.0

into the expressions box. You can then choose to replace or append column. The result pixel type should be FLOATTYPE.

![image](https://user-images.githubusercontent.com/94952931/156774249-a84171e8-0c08-4a5a-b1d8-9e9a85b42f2b.png)

Next, the image resizer should be used to resize the image. You can put the size you want (in pixels) for the image. We are resizing it 224x224x3.
Channel is to indicate whether the picture should be in RGB (3) or B&W (1).

![image](https://user-images.githubusercontent.com/94952931/156774356-50b0e3d8-d700-43a5-b8de-3e8b856382b4.png)

Partitioning the data into test, train, and validation allows us to effectively evaluate the model’s performance during and after training. As we already have our test data ready, we just need one partitioning node to partition the training and validation set.
How we partition the data is also very important. Depending on the type of data set, we may need to use stratified sampling or we might need to take from the top.
“Use random seed” is selected to specify a seed for random number generation for the partitioning. Setting this option results in the same records being assigned to the same set on successive runs.Partition the data 80-20 using stratified sampling from the Class column.

![image](https://user-images.githubusercontent.com/94952931/156774562-63878fb2-cce0-480d-95c9-017fd3155cf7.png)

### Convolution Neural Network

![image](https://user-images.githubusercontent.com/94952931/156774699-db5c9a1c-36ad-48ff-956e-3100f4773119.png)

Next the CNN is made using the method we explained in the sign language workflow. Just drag and drop the necessary nodes and configure the same as what is listen in the description. It is important to include dropout nodes after certain layers in order to prevent overfitting. Another way to prevent overfitting would be to include regularisation techniques that could be modified in the Keras Dense layers.  The final dense node has a unit of 1 because we only have 2 classes under 1 column, and so the sigmoid activation would be better suited for this scenario.

### Training & Evaluation
![image](https://user-images.githubusercontent.com/94952931/156774947-33c469a4-b92f-4634-8199-b5c04ed83eed.png)

The Keras Network Learner node will be configured similarly to our sign language workflow. Under the input data, select CONVERSION = FROM IMAGE and include IMAGE inside the greEn box. Under the target data tab, select CONVERSION = FROM NUMBER(INTEGER) and make sure the CLASS(TO NUMBER) COLUMN is included under the green box. Select the STANDARD LOSS FUNCTION = BINARY CROSS ENTROPY as we only have 2 classes and this would be best suited for our current dataset. You can choose for the epochs to be any number. With 15 epochs you will still get a good accuracy of more than 80% but with 20 epochs it is possible to achieve 86%. 

We include one more rule engine to convert the probability columns back to their classes. Since we are using sigmoid activation in the final dense layer, then the output will be shown a bit differently than our previous workflow. Instead of being either 0 or 1 in their class columns, if the output probability <= 0.5, then it is classified as benign. If the output probability > 0.5, then it is classified as malignant.

# Image Classification - Skin Classification using Transfer Learning
![image](https://user-images.githubusercontent.com/94952931/156776077-28242dbc-c9d9-4c38-8fc6-6fa4d7597cad.png)

For transfer learning, since the model is already trained online we are just going to import it into KNIME and edit a few of the last layers to better suit our dataset and workflow. We use the DL Python Network Creator and Editor Nodes to do this. 
Using the script, we are importing the VGG16 model into KNIME.  We changed the input shape to suit our skin cancer images and did not include the top.

![image](https://user-images.githubusercontent.com/94952931/156776129-36d0ca3e-3e36-4b22-98e9-72e72be2f20d.png)

Since we did not include the top when importing the VGG16 model, we create a simple network using the following Python script.

![image](https://user-images.githubusercontent.com/94952931/156776151-81eb2f03-9040-475f-a291-7b2ca7b1bc77.png)

The configurations for the learner are quite straightforward. Since we have a binary classification, we will use binary cross entropy as our loss function.

![image](https://user-images.githubusercontent.com/94952931/156776170-57cc60a4-b86f-4c5d-83ff-8203b7883eef.png)

![image](https://user-images.githubusercontent.com/94952931/156776184-652d38c9-1af0-4255-8e97-7505fff72632.png)

The epoch required for the deep learning model will vary according to the application, so it is best to try with smaller numbers first as training can take a very long time.
Keep the random seed number used throughout the workflow the same, and although it is preferable to use Adam as the optimizer feel free to experiment and use other optimizers.
For now, follow the configurations in the image.

![image](https://user-images.githubusercontent.com/94952931/156776231-36f6d1ce-8e6c-4696-b5a2-6129d5208015.png)

Turning down the learning rate reduces the random fluctuations in the error due to the different gradients on different mini-batches.
You can later untick this configuration to see how it will affect the outcome.

![image](https://user-images.githubusercontent.com/94952931/156776277-acd045ad-ab6f-48cd-b21a-9a4e9d8ef2cb.png)

The keras network executor will take in the test set and use the model that was just trained and fit the new data. Follow the configurations in the image.

![image](https://user-images.githubusercontent.com/94952931/156776367-90ac8dfb-f3e9-4861-8956-dfd33451aec2.png)

![image](https://user-images.githubusercontent.com/94952931/156776379-9ac6e4c3-c597-43e7-9785-25bcd2459eed.png)

### Post Processing & Evaluation
In this section we evaluate the accuracy of the model based on the partitioned test data.

![image](https://user-images.githubusercontent.com/94952931/156776465-2821cc29-7885-40b7-9079-c5199a90e620.png)

The rule engine can take multiple expressions, so this node is used to convert the numbers back to letters, which will then be used to score against the original class. Follow the syntax in the image.

![image](https://user-images.githubusercontent.com/94952931/156776526-1d223351-2856-4378-bc01-9af460b3b859.png)

The scorer will use the selected columns to determine the model’s accuracy.  Follow the configurations. To see the confusion matrix, right click on the node and select “confusion matrix.”

![image](https://user-images.githubusercontent.com/94952931/156776550-94393dce-882c-46c2-988c-768613d3e351.png)

![image](https://user-images.githubusercontent.com/94952931/156776562-44202885-7993-4ac4-a0f9-d7439347500c.png)

