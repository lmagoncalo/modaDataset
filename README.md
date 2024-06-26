# Moda Dataset

Steps to reproduce results:

1. download_dataset.py - It downloads all the images presents in the csv files to later be used.
2. prepare_dataset.py - For each line of the dataset, we open the image, pass it through the CLIP and store the returned embeddings and store also the avaliation of the piece. It returns two files (x_ModaDataset_CLIP_L14_embeddings.npy and y_ratings_X.npy) which contains these values.
3. The final step is decided by the type of evaluation performed:
 - If we want to finetune a pre-trained model we need to use the finetune.py
 - If we want to infer without any training on the current dataset we can use the inference.py
 - Finally, the train_predictor.py trains a similar model to the one presented in the original repository and the one used on the finetunning from scratch soleny based on the current dataset.
 - Note: Each file can receive a set of parameters to reproduce the available results namely, the traininf learning rate, the existance or not of a ReLU activation funtion in the model and the type of loss used to train.