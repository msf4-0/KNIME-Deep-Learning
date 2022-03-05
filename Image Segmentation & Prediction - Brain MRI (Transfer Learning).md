# Image Segmentation & Prediction - Brain MRI Scan using Unet Architecture
The following workflow will demonstrate how to use transfer learning in KNIME for image segmentation and prediction in a brain MRI scan dataset.

###### Dataset Link
Brain MRI Segmentation: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation  <br/>

###### Workflow Link
MRI Scan Segmentation & Prediction Workflow: https://tinyurl.com/yhrfs8y8 <br/>

# Brain MRI Scan Prediction & Segmentation - Transfer Learning using Unet Architecture
![image](https://user-images.githubusercontent.com/94952931/156867517-b1221e21-7b41-4dfc-85c8-55df3adafbc2.png)
![image](https://user-images.githubusercontent.com/94952931/156867523-b23d309f-c745-4901-8a22-a34857be0028.png)

This is what the dataset looks like. Each MRI Scan contains an associated tumour mask. <br/>

### Image Pre-Processing

![image](https://user-images.githubusercontent.com/94952931/156867601-2f529b36-afda-4935-92d5-4616a0d4b930.png)

We first read all the tif image files using the image reader node. For the image pre-processing, we are mainly doing the 4 following things:
1.	Appending each mask next to their MRI scan so we can view them per row
2.	Filtering out the images that do not have tumours
3.	Resizing the images to 256x256
4.	Normalizing the images

Although a bit tedious, we have to use a combination of string manipulation, row filter, column rename and joiner nodes to append the mask of each MRI scan next to each other, as they are read as individual pictures the first time. There’s no correct way to do this, as long as we arrive at the same output as seen below.

![image](https://user-images.githubusercontent.com/94952931/156867612-ba9961ad-1ee9-4ef8-a6fc-795e6ede15b1.png)

After appending the mask images in the adjacent column as so,  we can then use the image features node to filter out the images that have a blank (black) mask, indicating there are no tumours. Under the column selection, select IMAGE = IMAGE_MASK and under features, tick FIRST ORDER STATISTICS. Select MAX under features and click ok. 

![image](https://user-images.githubusercontent.com/94952931/156867621-0f2b6d3d-2d93-4bec-ab69-3bb6025eba8f.png)

Drag and drop a row filter after the image feature node and filter out any rows that contain a lower bound of 255.0. This will filter out all the rows where the labeling mask is empty. Before we move on to #3 and #4, we need to convert the mask to a labeling so that KNIME can superimpose the tumour mask on top of the Brain MRI for viewing. Use the image to labeling node and include the IMAGE MASK within the green box. Under the options tab, tick USE BACKGROUND VALUE AS BACKGROUND and select BACKGROUND VALUE = 0. You can now append the labeling to the same row as the brain MRI scan, and you can use the interactive segmentation view to see the tumours, superimposed on top of the original image.

![image](https://user-images.githubusercontent.com/94952931/156867637-5761daf2-603b-43ff-bfe9-a47f71d47444.png)
![image](https://user-images.githubusercontent.com/94952931/156867664-888cc71f-19bb-429a-9797-f7d4937323a2.png)
![image](https://user-images.githubusercontent.com/94952931/156867668-f1b148f3-6a7e-4b54-a07a-65097a35a6e4.png)

Now use the image calculator and normalize both the original MRI scan and the mask. You can do this by including the following expression in the expressions column.

$Image_mask_img$/255.0

The RESULT PIXEL TYPE = FLOAT TYPE. We then resize the image to 256x256x3 for the original brain scan, and 256x256x1 for the mask. Use a joiner to  combine these 2 tables back together.

### Partitioning
For partitioning we are going to use a 70-30 split. First we partition the test and non-test data set. Then from the non-test data set, we partition the training and validation set.

![image](https://user-images.githubusercontent.com/94952931/156867697-77b80174-1544-4ada-a00c-668f34319022.png)

### Transfer Learning
We are going to use the DL Python Network Creator to import the Unet architecture. You would need to import it using segmentation_models, and this has to also be installed on your virtual environment. We are also going to implement LeakyReLU in the architecture, which is an advanced loss function. Follow the figure below to put in the code for this node.

![image](https://user-images.githubusercontent.com/94952931/156867725-59770310-e1f6-47fb-801d-a3bc06787e56.png)

The DL Python Network Editor is used to implement the LeakyReLU in the architecture, by simply using the following code.

![image](https://user-images.githubusercontent.com/94952931/156867736-832862b4-6924-47ac-86b2-1d7c4fc405aa.png)


The Keras Network Learner will be configured similarly to our previous workflows, other than the epoch of which we will change to 80. After testing, a lower epoch count has been shown to sometimes give a blank prediction for the output mask for smaller tumours which leads to lower accuracy. Increasing the epochs has been shown to solve the issue, so 80 epochs are the recommended minimum. The Keras Network Writer node is optional, as this will save your trained model to a .h5 file to be used in other workflows or projects if needed.

The Keras Network Executor will be configured as follows,

![image](https://user-images.githubusercontent.com/94952931/156867743-e01c2359-c4f3-45b4-b642-2b120416f89d.png)

### Post Processing & Evaluation
![image](https://user-images.githubusercontent.com/94952931/156867754-6df85925-be8d-482d-a712-48bb46243c54.png)

We will first convert the predictions (which we named sigmoid) and the masks to an unsigned byte type using the image converter node. We then normalize both items using the following configuration in the image normalizer node,

![image](https://user-images.githubusercontent.com/94952931/156867792-66ec30a5-b4a6-4f46-8d51-7ebb49f88ed5.png)

Then, by using the image calculator node we can then subtract the predicted mask from the original label so that we can see the difference in the predictions visually. This is more of a subjective measure on how to visually see the difference, and more can be done in order to numerically calculate the accuracy if needed. We then convert the image converter to convert the images to bit type and use the compare segments node to see the pixel agreement between the original mask and the predicted. You should receive an output such as this at the end,

![image](https://user-images.githubusercontent.com/94952931/156867802-78179bc4-57a3-4e76-9fd5-f67882ef94bb.png)
![image](https://user-images.githubusercontent.com/94952931/156867812-ca407b5c-3173-43d6-b29b-31f4041c31ac.png)


