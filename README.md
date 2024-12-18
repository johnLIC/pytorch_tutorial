# pytorch_tutorial
I worked through the first tutorial on pytorch.org.

train.py - import the testing and training data, put it into Dataloader objects, define a loss function and an optimizer - in this case, stochastic gradient descent.  Part of training the model is checking the performance against the test data set.  That happens in this file.  The model is saved out after training.  Running

>python ./train.py

will train the model and output a model.pth file which contains the entire model, and can be loaded later.

classNN.py - defines the neural network - the weight layers and activation functions, and forward pass function.

load_saved.py - this is just a way to test that the model was saved out correctly.

infer.py - This is a file that can be used to make predictions with the trained model.  Run 

> python ./infer.py

to infer the type of clothing of a randomly chosen image from the dataset.  It will display the test image so you can verify the prediction.

I also did some work to visualize the training with a tensorboard, but that is not included in this example.
