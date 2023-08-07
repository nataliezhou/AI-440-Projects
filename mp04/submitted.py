# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

"""
1.  Build a neural network class.
"""
class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 5)
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################

        batch_size = x.shape[0]
        unflatten = nn.Unflatten(-1, (3, 31, 31)) # 1 is the batch size
        x = unflatten(x)

        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))

        x = torch.flatten(x,1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)

        
        return x

        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################


"""
2. Train your model.
"""
def fit(train_dataloader, test_dataloader, epochs):
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
        loss_fn:            your selected loss function
        optimizer:          your selected optimizer
    """
    
    # Create an instance of NeuralNet, don't modify this line.
    model = NeuralNet()


    ################# Your Code Starts Here #################
    """
    2.1 Create a loss function and an optimizer.

    Please select an appropriate loss function from PyTorch torch.nn module.
    Please select an appropriate optimizer from PyTorch torch.optim module.
    """
    #loss_fn = None
    #optimizer = None

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    #raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################


    """
    2.2 Train loop
    """
    for epoch in range(epochs):
        print("Epoch #", epoch)
        train(train_dataloader, model, loss_fn, optimizer)  # You need to write this function.
        test(test_dataloader, model, loss_fn)  # optional, to monitor the training progress
    return model, loss_fn, optimizer


"""
3. Backward propagation and gradient descent.
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################

    for batch, (X, y) in enumerate(train_dataloader):
        y_pred = model(X)
        error = loss_fn(y_pred, y)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    
    #raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################


def test(test_dataloader, model, loss_fn):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions
        loss_fn:            loss function
    """

    # test_loss = something
    # print("Test loss:", test_loss)
