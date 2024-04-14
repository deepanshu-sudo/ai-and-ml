import torch

"""
The provided Python function train is used to train a machine learning model. It takes six parameters: trainloader and valloader (which are PyTorch DataLoader objects for the training and validation datasets, respectively), model (the machine learning model to be trained), optimizer (the optimization algorithm to be used for training), criterion (the loss function), and device (the computing device where the training will be performed, typically a CPU or GPU).

The function starts by setting the model to training mode with model.train(). It then initializes variables to keep track of the total training loss, total validation loss, and the number of correct predictions in the training and validation sets.

The function then enters a loop over the training data. For each batch of data, it moves the inputs and labels to the specified device, computes the model's predictions with model(inputs), and calculates the loss with criterion(preds, labels). It then clears any gradients from the previous step with optimizer.zero_grad(), computes the gradients of the loss with respect to the model parameters with loss.backward(), and updates the model parameters with optimizer.step(). It also updates the total training loss and the number of correct predictions.

After going through all the training data, the function sets the model to evaluation mode with model.eval(), and enters a loop over the validation data. The process is similar to the training loop, but without the gradient computation and parameter update steps, since we don't want to change the model while validating it.

Finally, the function calculates the average training and validation losses by dividing the total losses by the number of batches, and the accuracy on the training and validation sets by dividing the number of correct predictions by the total number of samples. It then returns these four metrics as a list.
"""

def train(trainloader, valloader, model,optimizer, criterion, device):
    model.train()
    totalTrainLoss = 0
    totalValLoss = 0
    trainCorrect = 0
    valCorrect = 0

    for (inputs, labels) in trainloader:
        (inputs, labels) = (inputs.to(device), labels.to(device))
        # labels = torch.tensor(labels)
        preds = model(inputs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalTrainLoss += loss
        trainCorrect += (preds.argmax(1) == labels).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()
        for (inputs, labels) in valloader:
            (inputs, labels) = (inputs.to(device), labels.to(device))
            preds = model(inputs)
            loss = criterion(preds, labels)
            totalValLoss += loss
            valCorrect += (preds.argmax(1) == labels).type(torch.float).sum().item()

    avgTrainLoss = totalTrainLoss / len(trainloader)
    avgValLoss = totalValLoss / len(valloader)

    avgTrainLoss = avgTrainLoss.cpu().detach().numpy()
    avgValLoss = avgValLoss.cpu().detach().numpy()

    trainCorrect = trainCorrect / len(trainloader.dataset)
    valCorrect = valCorrect / len(valloader.dataset)

    return [avgTrainLoss, trainCorrect, avgValLoss, valCorrect]
