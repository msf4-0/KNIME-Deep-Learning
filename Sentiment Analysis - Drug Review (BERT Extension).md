# Sentiment Analysis - Drug Review using BERT Extension
The following workflow will demonstrate how to use a CNN to use the BERT extension within KNIME to do sentiment analysis for a drug review dataset.

###### Dataset Link
Drug Review Dataset: https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018  <br/>

###### Workflow Link
Drug Review Sentiment Analysis Workflow: https://tinyurl.com/2p8krhx5 <br/>


# Drug Review - BERT 
To do sentiment analysis on a dataset containing drug reviews, we will use the BERT Extension that exists within KNIME. There are many different types of BERT models, but we will be using a general one to do our analysis. BERT stands for Bidirectional Encoder Representations from Transformers, and it has been pretrained on a very large dataset - including the entire Wikipedia. BERT looks at both the right and left of the token’s context during the training phase, hence the first part of it’s name is Bidirectional.

![image](https://user-images.githubusercontent.com/94952931/156868029-bb8ed582-e313-409a-ac85-c1fce0aa0c9f.png)

### Image Pre-processing & Partitioning
![image](https://user-images.githubusercontent.com/94952931/156868037-d78108ac-5855-4c2e-8a7c-b1ee931b3127.png)

The pre-processing done for this workflow is not very complicated as we only need to do minimal things to prepare the data to be fed into the BERT Extension. We first use the row filter nodes to make sure there are no empty rows in each of the following columns: text, drugs, sentiment.

![image](https://user-images.githubusercontent.com/94952931/156868044-94b9e5b0-7a4f-4f9e-8f60-b2f2e0f7b50e.png)

We then use the string manipulation node to make all the texts lowercase - if you are using the distil-bert-uncased model it is not necessary to do this, however, it is still good practice to do so. Next, we use the shuffle node to shuffle all the texts and use the number to string node to convert the sentiment - originally an integer - to a string. We then partition our model into an 80-20 divide.

![image](https://user-images.githubusercontent.com/94952931/156868052-90675c16-6394-4464-bb23-c5f145cb3567.png)

### Conda Environment Propagation & BERT Extension
![image](https://user-images.githubusercontent.com/94952931/156868061-f4c4d7aa-9ee9-46bf-a3fc-6db7497cdb5d.png)

Drag and drop the Conda Environment Propagation Node into the workflow. This ensures that all the necessary packages needed in the Conda environment will be installed. Link the variable to the BERT Model Selector Node. Configure the BERT Model Selector Node to the following:

![image](https://user-images.githubusercontent.com/94952931/156868068-bf54e150-4970-47dc-a495-ce0bbaf3dfb0.png)

Use the following configurations for the BERT Classification Learner Node. You can choose to fine-tune in order to get better results (around 8% better accuracy) but this comes at the expense of a much longer computation time.

![image](https://user-images.githubusercontent.com/94952931/156868078-e39074ad-03ab-44ca-83aa-46c45be42aeb.png)

![image](https://user-images.githubusercontent.com/94952931/156868081-33cae716-9b88-4111-b71f-702eeb242e53.png)

### Post Processing & Model Evaluation
Finally, drop the Scorer into the workflow.  This workflow should demonstrate an accuracy of 82% for 2 classes, and 74% for 3 classes (within the sentiment column).



