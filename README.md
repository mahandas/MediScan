![alt text](https://github.com/mahandas/MediScan/blob/main/Screen%20Shot%202021-01-24%20at%203.14.28%20PM.png)

## Inspiration
We noticed the lack of easy to use machine learning tools in medical imaging. So we decided to make MediScan, a tool for fast and easy detection of cancerous regions in breast mammograms. We believe MediScan as a screening tool will be very useful to radiologists.

## What it does
MediScan is web application powered by Deep Learning Models for predicting cancerous regions in breast mammograms. A user, for example a radiologist, would first select images to upload, a threshold value and then press submit. On submission, our backend will pass on the images to our deep learning models and will generate new images with bounding boxes if any are found. The bounding boxes will be clearly visible on the output images alongside with their probabilities.

## How we built it
We created a pipe for analysis of breast mammograms. We utilized 2 models. First an image goes through a Densenet model which predicts whether it is malignant or not. If malignant, the image is then passed into a FasterRCNN model which predicts cancerous regions with their corresponding probabilities. These models were both trained on DDSM dataset and were part of our research projects. We utilized PyTorch for training and inference of the models. We utilized Flask as the web framework for the web application.

## Challenges we ran into
Building a functional UI in a short period of time with authentication was a challenge. Generating new images from the predictions of our model was a difficult task. Building a machine learning pipe with 2 models was also a challenge.

## Accomplishments that we're proud of
We are proud of a fully functioning product, with proper user authentication and elegant UI. We are also proud of the pace at which we were able to build the machine learning pipeline.

## What we learned
We learned about web application development and building machine learning pipelines. We also learned how to integrate the two to produce a well functioning tool.

## What's next for MEdiScan: Breast Lesion detection
MediScan can be used with any machine learning model and we plan to use it for various applications like Covid-19 classification of X-rays and MRI scan tumor segmentation in the future.

![alt text](https://github.com/mahandas/MediScan/blob/main/Screen%20Shot%202021-01-24%20at%201.00.10%20PM.png)

![alt text](https://github.com/mahandas/MediScan/blob/main/Screen%20Shot%202021-01-24%20at%201.16.39%20PM.png)

![alt text](https://github.com/mahandas/MediScan/blob/main/Screen%20Shot%202021-01-24%20at%201.18.24%20PM.png)
