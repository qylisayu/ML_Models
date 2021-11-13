# import Torch packages and its submodules
import torch
from torch import nn, optim

# import TorchVision and its submodules
import torch.nn.functional as F
from torchvision import datasets, transforms

# import other packages
import numpy as np
import math

import matplotlib
from matplotlib import pyplot as plt

import os
import splitfolders

from timeit import default_timer as timer
import pandas as pd

# Task 1
#-----------------------------------------------------------------------------------------------------
DATA_PATH = "notMNIST_small"

SPLIT_DATA_PATH = "notMNIST_small_split"

splitfolders.ratio(DATA_PATH, output=SPLIT_DATA_PATH, seed=1337, ratio=(15000/18720, 1000/18720, (18720-15000-1000)/18720)) 

transform = transforms.Compose([                          
                                transforms.Grayscale(num_output_channels=1),                        
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

TRAIN_DATA_PATH = SPLIT_DATA_PATH + "/train"
VAL_DATA_PATH = SPLIT_DATA_PATH + "/val"
TEST_DATA_PATH = SPLIT_DATA_PATH + "/test"

trainset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)

valset = datasets.ImageFolder(root=VAL_DATA_PATH, transform=transform)

testset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)

# Task 2
#-----------------------------------------------------------------------------------------------------
def new_model():
  input_size = 784
  hidden_sizes = 1000
  output_size = 10
  model = nn.Sequential(nn.Linear(input_size, hidden_sizes),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes, output_size))
  return model

# This function is similar to the tutorial 6 code (with some edits).
def train(model,
          criterion,
          optimizer,
          trainloader,
          validloader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2):
    

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(trainloader):
            data = torch.squeeze(data)
            data = data.reshape(len(data), 784)
            
            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()
            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            m = nn.Softmax(dim=1)
            _, pred = torch.max(m(output), dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            #print(
            #    f'Epoch: {epoch}\t{100 * (ii + 1) / len(trainloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
            #    end='\r')

        # After training loops ends, start validation
        else:
            #model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in validloader:

                    data = torch.squeeze(data)
                    data = data.reshape(len(data), 784)

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(m(output), dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(trainloader.dataset)
                valid_loss = valid_loss / len(validloader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(trainloader.dataset)
                valid_acc = valid_acc / len(validloader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}% (training acc: {100 * history[best_epoch][2]:.2f}%)'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history
  

def test(model, testloader, criterion):
  
  test_loss = 0.0
  test_acc = 0.0
  model.eval()

  for data, target in testloader:

      data = torch.squeeze(data)
      data = data.reshape(len(data), 784)

      # Forward pass
      output = model(data)

      # Validation loss
      loss = criterion(output, target)
      # Multiply average loss times the number of examples in batch
      test_loss += loss.item() * data.size(0)
      m = nn.Softmax(dim=1)
      
      # Calculate validation accuracy
      _, pred = torch.max(m(output), dim=1)
      correct_tensor = pred.eq(target.data.view_as(pred))
      accuracy = torch.mean(
          correct_tensor.type(torch.FloatTensor))
      # Multiply average accuracy times the number of examples
      test_acc += accuracy.item() * data.size(0)

  # Calculate average losses
  test_loss = test_loss / len(testloader.dataset)

  # Calculate average accuracy
  test_acc = test_acc / len(testloader.dataset)

  print(f'\tTest Loss: {test_loss:.4f}')
  print(f'\tTest Accuracy: {100 * test_acc:.2f}%')
  print(f'\tTest Error: {100-100 * test_acc:.2f}%')
  
  return test_loss, test_acc, 1-test_acc
  
  
  
  
  
loss = nn.CrossEntropyLoss() 
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
trainloader64 = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
validloader64 = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
trainloader32 = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
validloader32 = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)



# model_1
print("model1")
model1 = new_model()
optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.1)
model1, history1 = train(model1, loss, optimizer1, trainloader64, validloader64, save_file_name='MODEL1', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history1[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model 1)')
plt.savefig('Task2_model1_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * history1[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model 1)')
plt.savefig('Task2_model1_error')


# model_2
print("model12")
model2 = new_model()
optimizer2 = optim.SGD(model2.parameters(), lr=0.05)
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.1)
model2, history2 = train(model2, loss, optimizer2, trainloader64, validloader64, save_file_name='MODEL2', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history2[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model 2)')
plt.savefig('Task2_model2_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * history2[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model 2)')
plt.savefig('Task2_model2_error')

# model_3
print("model3")
model3 = new_model()
optimizer3 = optim.SGD(model3.parameters(), lr=0.07)
scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=5, gamma=0.1)
model3, history3 = train(model3, loss, optimizer3, trainloader32, validloader32, save_file_name='MODEL3', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history3[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model 3)')
plt.savefig('Task2_model3_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * history3[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model 3)')
plt.savefig('Task2_model3_error')

# model_4
print("model4")
model4 = new_model()
optimizer4 = optim.SGD(model4.parameters(), lr=0.0001)
scheduler4 = optim.lr_scheduler.StepLR(optimizer4, step_size=5, gamma=0.1)
model4, history4 = train(model4, loss, optimizer4, trainloader32, validloader32, save_file_name='MODEL4', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history4[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model 4)')
plt.savefig('Task2_model4_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * history4[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model 4)')
plt.savefig('Task2_model4_error')

# model_5
print("model5")
model5 = new_model()
optimizer5 = optim.SGD(model5.parameters(), lr=0.001)
scheduler5 = optim.lr_scheduler.StepLR(optimizer5, step_size=5, gamma=0.1)
model5, history5 = train(model5, loss, optimizer5, trainloader64, validloader64, save_file_name='MODEL5', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history5[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model 5)')
plt.savefig('Task2_model5_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * history5[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model 5)')
plt.savefig('Task2_model5_error')


T2Model_best = new_model()
T2Model_best.load_state_dict(torch.load("MODEL2"))
test_loss, test_acc, test_error = test(T2Model_best, testloader, loss)
plt.clf()
plt.figure(figsize=(8, 6))
plt.hlines(100*test_error, 0, 1, label='test error')
plt.legend()
plt.ylabel('test error %')
plt.title('Test Error (Model 2)')
plt.savefig('Best Test Error (Model 2)')


# Task 3
#-----------------------------------------------------------------------------------------------------
def T3new_model(hidden_size):
  input_size = 784
  hidden_sizes = hidden_size
  output_size = 10
  model = nn.Sequential(nn.Linear(input_size, hidden_sizes),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes, output_size))
  
  return model

#Model with 100 hidden
print("model with 100 hidden units:")
T3model100 = T3new_model(100)
optimizer100 = optim.SGD(T3model100.parameters(), lr=0.05)
scheduler100 = optim.lr_scheduler.StepLR(optimizer100, step_size=5, gamma=0.1)
T3model100, history100 = train(T3model100, loss, optimizer100, trainloader64, validloader64, save_file_name='MODEL100', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history100[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model with 100 hidden units)')
plt.savefig('Task3_model100_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * history100[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model with 100 hidden units)')
plt.savefig('Task3_model100_error')


#Model with 500 hidden
print("model with 500 hidden units:")
T3model500 = T3new_model(500)
optimizer500 = optim.SGD(T3model500.parameters(), lr=0.05)
scheduler500 = optim.lr_scheduler.StepLR(optimizer500, step_size=5, gamma=0.1)
T3model500, history500 = train(T3model500, loss, optimizer500, trainloader64, validloader64, save_file_name='MODEL500', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history500[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model with 500 hidden units)')
plt.savefig('Task3_model500_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * history500[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model with 500 hidden units)')
plt.savefig('Task3_model500_error')

T3_100Model_best = T3new_model(100)
T3_100Model_best.load_state_dict(torch.load("MODEL100"))
test_loss, test_acc, test_error = test(T3_100Model_best, testloader, loss)
plt.clf()
plt.figure(figsize=(8, 6))
plt.hlines(100*test_error, 0, 1, label='test error')
plt.legend()
plt.ylabel('test error %')
plt.title('Test Error (Model with 100 hidden units)')
plt.savefig('Best Test Error (Model 100)')

T3_500Model_best = T3new_model(500)
T3_500Model_best.load_state_dict(torch.load("MODEL500"))
test_loss, test_acc, test_error = test(T3_500Model_best, testloader, loss)
plt.clf()
plt.figure(figsize=(8, 6))
plt.hlines(100*test_error, 0, 1, label='test error')
plt.legend()
plt.ylabel('test error %')
plt.title('Test Error (Model with 500 hidden units)')
plt.savefig('Best Test Error (Model 500)')


# Task 4
#--------------------------------------------------------------------------------------------------
def T4new_model():
  input_size = 784
  hidden_sizes = 500
  output_size = 10
  model = nn.Sequential(nn.Linear(input_size, hidden_sizes),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes, hidden_sizes),
                      nn.ReLU(),                      
                      nn.Linear(hidden_sizes, output_size))
  
  return model


#Model with 2 hidden layers
print("model with 2 hidden layers:")
T4model = T4new_model()
optimizerT4 = optim.SGD(T4model.parameters(), lr=0.05)
schedulerT4 = optim.lr_scheduler.StepLR(optimizerT4, step_size=5, gamma=0.1)
T4model, historyT4 = train(T4model, loss, optimizerT4, trainloader64, validloader64, save_file_name='MODELT4', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        historyT4[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model with 2 hidden layers)')
plt.savefig('Task4_model_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * historyT4[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model with 2 hidden layers)')
plt.savefig('Task4_model_error')

T4_Model_best = T4new_model()
T4_Model_best.load_state_dict(torch.load("MODELT4"))
test_loss, test_acc, test_error = test(T4_Model_best, testloader, loss)
plt.clf()
plt.figure(figsize=(8, 6))
plt.hlines(100*test_error, 0, 1, label='test error')
plt.legend()
plt.ylabel('test error %')
plt.title('Test Error (Model with 2 hidden layers)')
plt.savefig('Best Test Error (Model T4)')

# Task 5
# ------------------------------------------------------------------------------------------------------------
def T5new_model():
  input_size = 784
  hidden_sizes = 1000
  output_size = 10
  model = nn.Sequential(nn.Linear(input_size, hidden_sizes),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(hidden_sizes, output_size))
  return model


#Model with dropout
print("model with dropout:")
T5model = T5new_model()
optimizerT5 = optim.SGD(T5model.parameters(), lr=0.05)
schedulerT5 = optim.lr_scheduler.StepLR(optimizerT5, step_size=5, gamma=0.1)
T5model, historyT5 = train(T5model, loss, optimizerT5, trainloader64, validloader64, save_file_name='MODELT5', max_epochs_stop=3, n_epochs=50, print_every=2)

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        historyT5[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average CE Loss')
plt.title('Training and Validation Losses (Model with dropout)')
plt.savefig('Task5_model_loss')

plt.clf()
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 - 100 * historyT5[c], label="1-"+c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Error %')
plt.title('Training and Validation Error (Model with dropout)')
plt.savefig('Task5_model_error')

T5_Model_best = T5new_model()
T5_Model_best.load_state_dict(torch.load("MODELT5"))
test_loss, test_acc, test_error = test(T5_Model_best, testloader, loss)
plt.clf()
plt.figure(figsize=(8, 6))
plt.hlines(100*test_error, 0, 1, label='test error')
plt.legend()
plt.ylabel('test error %')
plt.title('Test Error (Model with dropout)')
plt.savefig('Best Test Error (Model T5)')