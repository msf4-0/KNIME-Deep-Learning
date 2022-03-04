# Image Classification - Malaria using Transfer learning (VGG-16)
The following workflow will demonstrate how to use transfer learning to do image classification for a binary malaria dataset.

###### Dataset Link
Malaria classification: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria <br/>

###### Workflow Link
Malaria Binary Image Classification Workflow: https://tinyurl.com/3zcva9y8  <br/>

# Sign Language Alphabet Recognition - Transfer Learning
The malaria classification workflow is similar to the skin cancer workflow, as we also use transfer learning to import the VGG16 model. Therefore, similar sections would be the same as previous workflows and explanations will just be carried over. The only difference would be the pre-processing and some minor configurations in the machine learning portion of the workflow.

### Image Pre-processing & Partitioning
Our goal in this portion of the workflow is to read in the images from our dataset and append their labels which we can get from their original file names. 

![image](https://user-images.githubusercontent.com/94952931/156777520-729dedd3-2562-427f-baee-714840dc62a8.png)

We need to read in the image files, and append the folder they came from as their label. For example, if they came from the Uninfected folder, then their label would be “Uninfected” as well. We first drag 2 of the List Files/Folders node into the workflow. We configure the first node to read all the image files in the “Parasitized” folder and the other node the “Uninfected” file. We then use the path to string node to convert so that the image reader (table) node can read in the files. We then use the column appender node to append the initial string that reads in the file location and filter out the unwanted columns. 

![image](https://user-images.githubusercontent.com/94952931/156777565-18b43a83-1679-4cad-864f-6ee4d54847e2.png)

Next we concatenate both the uninfected and parasitized images into one table. Now, use the rule engine node to append the class columns. Place the following syntax into the rule engine node,

$Location$ LIKE “*Uninfected*” => “N”
TRUE => “P”

The above syntax means that if there is the word “uninfected” within the file path (under the location column) to label it as N for Negative. The second line states that if the preceding line is false, therefore making the second line TRUE, then name it P for Positive.

![image](https://user-images.githubusercontent.com/94952931/156777612-f5145634-06c2-4ba0-adc1-bdca39a9865b.png)

Use the category to number node to change the class encoding from N and P to 0 and 1. This will make it easier for machine learning later on. Place the image calculator and image resizer nodes to do the rest of the pre-processing for the data. Follow the configurations below.

![image](https://user-images.githubusercontent.com/94952931/156777651-e66b5cf7-b8af-4608-bc95-5a92a05275fe.png)

![image](https://user-images.githubusercontent.com/94952931/156777665-c9c2c988-5f1c-4c79-9917-1cd4eb742b94.png)

Use the row filter node in order to filter any missing rows. Finally, use 2 partitioning nodes to partition the data with an 80-20 ratio using stratified sampling based on class (string).

### Transfer Learning
![image](https://user-images.githubusercontent.com/94952931/156777771-922948e8-4694-4a7d-ba8f-675cebb359ec.png)

We only have two nodes in the transfer learning section. The first node is to import the architecture we plan on using which is VGG16. The second node is to create additional head layers and freeze some of them so that their weights will not be damaged. The DL Python Network Editor can be exchanged with a combination of Keras DL nodes and the Keras Freezing Layer node, this is another alternative way of doing it. 

![image](https://user-images.githubusercontent.com/94952931/156777801-8de97e7e-1d03-4578-a964-0fa24f31febf.png)

![image](https://user-images.githubusercontent.com/94952931/156777825-41c20b58-cba9-4663-8974-113a93d5cdd9.png)

### Training & Model Evaluation
![image](https://user-images.githubusercontent.com/94952931/156777882-e6de0426-5801-4db0-a8bc-18c0247ec3f9.png)

All the nodes in the Training & Model Evaluation will have the similar configurations as the previous workflow. With 20 epochs, an accuracy of 92% can be achieved. Follow the configurations below for the Keras Network Learner and Executor.

![image](https://user-images.githubusercontent.com/94952931/156777918-52f1a132-18e8-478a-9c45-b7ba463c44b8.png)

![image](https://user-images.githubusercontent.com/94952931/156777926-50e08955-0cb4-4a2d-a0d3-abd38c065a53.png)

![image](https://user-images.githubusercontent.com/94952931/156777940-8d3b44f6-d786-4a98-b4ed-64d9401a0713.png)

![image](https://user-images.githubusercontent.com/94952931/156777947-f95fd454-181a-4065-a045-0770be0228e0.png)

If you’d like to save the model, simply drag and drop the Keras Network Writer node and connect it to the output of the Keras Network Learner. Specify the file path destination and the model should be saved as a .h5 file.

![image](https://user-images.githubusercontent.com/94952931/156777979-7a45a9d0-21a4-406a-a962-071feda36ec8.png)

After the Keras Network Executor, use the rule engine node to specify that if the output is less than 0.5 (probability is less than 50%) then the prediction should be “P” for Positive. Finally, drag the scorer and include the two columns for comparison. 

![image](https://user-images.githubusercontent.com/94952931/156778012-a8261d18-c9b5-4f8e-bbc0-f233a20f9b9b.png)

![image](https://user-images.githubusercontent.com/94952931/156778035-681c1ebe-4572-43f9-9ba1-ea44c662468a.png)




