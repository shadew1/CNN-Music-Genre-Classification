# Music genre classification using CNNs

The goal of the assignment is to classify different music genres based on their spectrograms. 

Download the **template notebook** as a start for your assignment: there, you will find the way to download the data and some helpful functions already coded for you. Upload it on Google Colab and modify the name of the file as "ST456_WT2024_Assignment1_YourLSECandidateNumber.ipynb", where `YourLSECandidateNumber` corresponds to your 5-digit candidate number available on LfY.

## Data

Download the [training, validation and test data](https://drive.google.com/file/d/1D2ddb2vqRF6tyju8mkrM3R-Rxdrk6qX6/view?usp=share_link). It is recommended you download the file and then place it in the Google Drive associated with your Colab account. Then, once you mount your Drive on Colab, you can easily access them in the form of numpy arrays.

The data represent the log-transformed Mel spectrograms derived from the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). The original GTZAN dataset contains 30-second audio files of 1,000 songs associated with 10 different genres (100 per genre). We have reduced the original data to 4 genres (400 songs) and transformed it to obtain, for each song, 15 log-transformed Mel spectrograms. Each Mel spectrogram is an image file which describes the time, frequency and intensity of a song segment. In particular, the x-axis represents time, the y-axis is a transformation of the frequency (to log scale and then the so-called mel scale) and the color of a point represents the decibels of that frequency at that time (with darker colours indicating lower decibels). Here you can see an example of Mel spectrogram (x and y ticks identify the pixels making up the picture):

![alt text](https://github.com/lse-st456/assignment1-2024/blob/main/mel_spectrogram_example.png)

The training data represent approximately 66% of the total number of data points, the validation set 14% and test set 30%.

The labels of the classes are such that: 
- the first class corresponds to classical music
- the second to disco music
- the third to metal music
- the fourth to rock music

# P1 - CNN

For this exercise, you **must use the CPU runtime**.

The goal is to train a CNN-based classifier on the Mel spectrograms to predict the corresponding music genres. Implement the following CNN architecture:

- a 2D convolutional layer with 4 channels of squared filters of size 5, padding, default stride, ReLU activation function and default weight and bias initialisations.
- a 2D max pooling layer with size 2 and stride 2.
- a 2D convolutional layer with 8 channels of squared filters of size 5, padding, default stride, ReLU activation function and default weight and bias initialisations.
- a 2D max pooling layer with size 2 and stride 2.
- a 2D convolutional layer with 16 channels of squared filters of size 5, padding, default stride, ReLU activation function and default weight and bias initialisations.
- a 2D max pooling layer with size 2 and stride 2.
- a layer transforming the output filters to a 1D vector.
- a dense layer made of 50 neurons, ReLU activation and L2 regularisation with a penalty of 0.01
- an output layer with the required number of neurons and activation function.

Compile the model using an appropriate evaluation metric and loss function. To optimise, use the mini-batch stochastic gradient descent algorithm with batch size 32. Train the model for 20 epochs. 

**IMPORTANT:** - For reproducibility of the results, before training your model you must run the following 2 lines of code to fix the seeds:

`tf.keras.utils.set_random_seed(42)`

`tf.config.experimental.enable_op_determinism()`

Answer to the following questions:
1. How many parameters does the model train? Before performing the training, do you expect this model to overfit? Which aspects would influence the overfitting (or not) of this model?
2. Plot the loss function and the accuracy per epoch for the train and validation sets.
3. Which accuracy do you obtain on the test set?
4. Using the function `plot_confusion_matrix` plot the confusion matrices of the classification task on the train set and test set. What do you observe from this metric? Which classes display more correct predictions? And wrong?
5. Using the function `ind_correct_uncorrect` extract the indexes of the training data that were predicted correctly and incorrectly, per each class. For each music genre, perform the following steps:
   - Using the function `plot_spectrograms` plot the 12 mel spectrograms of the first 6 data points which were predicted correctly and the first 6 which were predicted wrongly. Do you observe some differences among music genres?
   - Using the function `print_wrong_prediction` print the predicted classes of the first 6 data points which were predicted wrongly.
   - Using the Grad-CAM method, implemented in the function `plot_gradcam_spectrogram`, print the heatmaps of the last pooling layer for the same 12 extracts (6 correct + 6 wrong). Comment on the heatmaps obtained. Do you observe differences among the heatmaps of different music genres? Can you understand why the model got some predictions wrong?
6. Comment on the previous question: what are your thoughts about the applicability of the Grad-CAM tool on these data?

# P2 - Disentangling time and frequency

The images we are using in this assignment are different from a usual picture: the x and y axes carry different meanings. With the tools we are exploring during lectures and seminars, can you propose a CNN architecture that takes into account differently the time and frequency components of the spectrograms?

Present and describe the architecture you have chosen and justify the rationale behind it. Plot training and validation loss and accuracy over 20 epochs (this time you can use the GPU runtime if the model is slow to train). Print the accuracy on the test set and the confusion matrices on the training and test sets.
