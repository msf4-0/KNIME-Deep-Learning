# Time Series Forecasting - Daily Weather Prediction using a Simple LSTM Model
The following workflow will demonstrate how to use a CNN to do deep learning in KNIME for image classification of a sign language dataset.

###### Dataset Link
Daily weather forecast (India): https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data  <br/>

###### Workflow Link
Daily Weather Forecast (India) Workflow: https://tinyurl.com/2p9bywnr <br/>

# Daily Weather Prediction - Deep Learning 
![image](https://user-images.githubusercontent.com/94952931/156868658-7b801486-a97b-499c-9ca5-252ed79a753d.png) <br/>
CSV Data Snippet<br/>

All the training data is in a csv format with the following columns:<br/>
1.date<br/>
2.mean temperature<br/>
3.humidity<br/>
4.wind speed<br/>
5.mean pressure<br/>

There are 1462 rows with the dates spanning from 2013-01-01 to 2017-01-01. 
The testing data has the same columns with dates spanning from 2017-01-01 to 2017-04-24.

### Data Pre-Processing
Before anything, the images from the dataset has to be cleaned and processed in a way that the model can accept it.

![image](https://user-images.githubusercontent.com/94952931/156868486-6f5c8d41-36dc-40aa-b8fa-9ff45ca8724b.png)

We will first use two CSV Reader nodes in order to read both the training and test csv files from the downloaded dataset. We can then use the Concatenate node to combine both tables together.

![image](https://user-images.githubusercontent.com/94952931/156868494-c8524c1e-015f-4604-8900-2984ed2db65d.png)

Inside the Pre Processing Metanode, we will clean the data up before feeding it into the model. We use Sorter to sort the data ascending by date (as time series is usually highly dependent on the date/time, this step is crucial). 

![image](https://user-images.githubusercontent.com/94952931/156868689-5ef4ae54-7505-4be7-8a09-2c792abfc850.png)

We then convert the date column - originally a string - to KNIME’s version of date/time using the String to Date&Time node. This node will convert the date column (string) to KNIME’s date and string object. You can configure the date format and type as well as the execution (whether it should fail on error). Follow the configurations on the image.

![image](https://user-images.githubusercontent.com/94952931/156868707-d01f4461-8e42-4635-94e5-21c0ddfeacad.png)

![image](https://user-images.githubusercontent.com/94952931/156868713-8b25371e-485e-4595-91a9-293492d69b8b.png)

Now we can use the column filter node to filter out the unneeded data. In our case we are going to use the Mean Temperature column to train our model, so we are only passing the date and mean temperature column. 

![image](https://user-images.githubusercontent.com/94952931/156868731-1cd9c380-f49b-40a0-8dfd-a4ead696db9b.png)

The timestamp alignment node hecks whether the selected timestamp column is uniformly sampled in the selected time scale. Missing values will be inserted at skipped sampling times. Select Period = Day, Timestamp column = date and tick replace timestamp column.

![image](https://user-images.githubusercontent.com/94952931/156868736-4b660ad6-3572-44f4-936e-c69d925df15a.png)


### Lagging & Partitioning
![image](https://user-images.githubusercontent.com/94952931/156868515-52cc1342-d9c8-4c99-8079-6e44bf93b54a.png)

We are now going to lag and partition the data before feeding it into our model. Using the lag column, we are going to put in the following configurations.

![image](https://user-images.githubusercontent.com/94952931/156868527-b83683c1-6560-4e6c-9dd1-010c7f6ea5db.png)

This will lag our input column by 1,200 times. We then aggregate all the lagged columns so that they fit into one list column. 

![image](https://user-images.githubusercontent.com/94952931/156868540-fcb08b1d-803b-4952-8b4e-9da4ca4f9ed4.png)

![image](https://user-images.githubusercontent.com/94952931/156868541-c717f614-bca5-4321-ac2b-aa1d2d88da61.png)

Finally, partition the data into a 80-20 split using TAKE FROM TOP.
  
### LSTM Model
![image](https://user-images.githubusercontent.com/94952931/156868556-fc71c273-baf7-4c0a-b30f-42d027438b54.png)

Follow the configurations listed below for all 3 layers of the LSTM model.

![image](https://user-images.githubusercontent.com/94952931/156868567-550daa83-1eab-4bae-a18b-b42a3a61e6ae.png)

![image](https://user-images.githubusercontent.com/94952931/156868573-bef41bd5-c9d6-443e-b1ff-bc4537e22c33.png)

![image](https://user-images.githubusercontent.com/94952931/156868577-28adcf04-612d-4c18-82da-add0df522afd.png)

### Training & Post Evaluation
![image](https://user-images.githubusercontent.com/94952931/156868581-bce38c26-1b92-4927-88a0-142ff372813c.png)

Follow the configurations below for the Keras Network Learner.

![image](https://user-images.githubusercontent.com/94952931/156868607-6289e54a-957c-4b38-9998-822ae77a9ca7.png)

![image](https://user-images.githubusercontent.com/94952931/156868610-14a8c853-5f34-446b-bd95-058a66c24a96.png)

![image](https://user-images.githubusercontent.com/94952931/156868611-37669b06-aaaf-4984-ab02-0afc4cbf6f24.png)

The Deployment loop component can be taken from the original cited workflow, or made from scratch. Essentially, this component will first convert the Keras network to Tensorflow. Then to execute the network, we start with an input of the same length as the training. We apply our network to predict the next character, delete the first character, and apply the network again to our new sequence and so on.

![image](https://user-images.githubusercontent.com/94952931/156868624-b3bf9a0b-bf12-4102-97b3-5d141492efdd.png)

The scorer will use the selected columns to determine the model’s accuracy. Follow the configurations. To see the confusion matrix, right click on the node and select “confusion matrix.”

![image](https://user-images.githubusercontent.com/94952931/156868631-2c3f0127-8c27-49d2-9a6f-d3940b043509.png)

![image](https://user-images.githubusercontent.com/94952931/156868637-314591cc-1b8f-4458-8e7f-ae0bcbd02619.png)

This nline plot node will present a visual plot of how accurate the predicted forecast was in comparison with the actual values (mean temperature). To see this plot, just right click on the node and select view line plot.

![image](https://user-images.githubusercontent.com/94952931/156868648-438e8ffc-f921-46b1-ac2a-6be66338e2be.png)

![image](https://user-images.githubusercontent.com/94952931/156868650-39567ce3-ec6a-4602-b398-aacfb7174fcb.png)


