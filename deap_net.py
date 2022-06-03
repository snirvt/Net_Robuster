import numpy as np
from copy import deepcopy
import random

import torch
from torch import nn

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier


def get_model_performance(model, dataloader, sample_size = float('inf'), print_values = True):
    correct = 0
    total = 0
    loss = 0
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            if total >= sample_size:
                break
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            loss += cross_entropy_loss(outputs, labels.long())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct // total
    if print_values:
        print(f'Accuracy of the network on {total} images: {accuracy} %')
        print(f'Loss of the network on {total} images: {loss}')
    return loss, accuracy




def network_score(net, train_loader, val_loader, epsilon, sample_size):
    n_epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = random.choice([0.01,0.005,0.001]))# 0.01, 0.005, 0.001

    epochs_train_acc = np.zeros(n_epochs)
    epochs_train_losses = np.zeros(n_epochs)
    epochs_test_acc = np.zeros(n_epochs)
    epochs_test_losses = np.zeros(n_epochs)


    for epoch in range(n_epochs):
        train_batch_loss = 0
        test_batch_loss = 0
        correct = 0
        total = 0
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            # images = images[:10]
            # labels = labels[:10] # save time for debuging
            # if total >= sample_size:
                # break
            images = images.to(device)
            reg_images = images[:int(len(images)/2)]
            reg_labels = labels[:int(len(images)/2)]
            att_images = images[int(len(images)/2):]
            att_labels = labels[int(len(images)/2):]
            optimizer.zero_grad()
            y_pred = net(reg_images)
            train_loss = criterion(y_pred, reg_labels.to(device))

            predictions = torch.argmax(y_pred, dim=1)
            total += reg_labels.shape[0]
            correct += torch.sum((predictions == reg_labels.to(device)).int())

            train_batch_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

            ### attack ###
            classifier = PyTorchClassifier(
                model=net,
                # clip_values=(min_pixel_value, max_pixel_value),
                loss=criterion,
                optimizer=optimizer,
                input_shape=tuple(images.shape[1:]),#(3, 32, 32),
                nb_classes=y_pred.shape[1],
            )
            attack = FastGradientMethod(estimator=classifier, eps=epsilon)
            x_attack = attack.generate(x=att_images.cpu().numpy())
            x_attack = torch.tensor(x_attack).to(device)
            optimizer.zero_grad()
            y_pred = net(x_attack)
            train_loss = criterion(y_pred, att_labels.to(device))
            predictions = torch.argmax(y_pred, dim=1)
            total += att_labels.shape[0]
            correct += torch.sum((predictions == att_labels.to(device)).int())
            train_batch_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()


        epochs_train_losses[epoch] = train_batch_loss / len(train_loader)
        epochs_train_acc[epoch] = correct / total
        print(f'Training loss for epoch #{epoch}: {epochs_train_losses[epoch]:.4f}')
        print(f'Training accuracy for epoch #{epoch}: {epochs_train_acc[epoch]:.4f}')

        test_total = 0
        test_true = 0

        net.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                # images = images[:10]
                # labels = labels[:10] # save time for debuging
                # if test_total >= sample_size:
                    # break
                images = images.to(device)
                y_pred = net(images)

                predictions = torch.argmax(y_pred, dim = 1)
                test_total += labels.shape[0]
                test_true += torch.sum((predictions == labels.to(device)).int())

                test_loss = criterion(y_pred, labels.to(device))
                test_batch_loss += test_loss.item()

                ### attack ###

                with torch.enable_grad():
                    x_attack = attack.generate(x=images.cpu().numpy())
                x_attack = torch.tensor(x_attack).to(device)
                y_pred = net(x_attack)
                test_loss = criterion(y_pred, labels.to(device))
                predictions = torch.argmax(y_pred, dim=1)
                test_total += labels.shape[0]
                test_true += torch.sum((predictions == labels.to(device)).int())
                test_batch_loss += test_loss.item()




            epochs_test_losses[epoch] = test_batch_loss / len(val_loader)
            epochs_test_acc[epoch] = test_true / test_total
            print(f'Test loss for epoch #{epoch}: {epochs_test_losses[epoch]:.4f}')
            print(f'Test accuracy for epoch #{epoch}: {epochs_test_acc[epoch]:.4f}')

    # print_losses_acc_to_file(file_name, epochs_train_losses, epochs_test_losses, epochs_train_acc, epochs_test_acc, mode_flag)

    best_val_idx = np.argmin(epochs_test_losses)
    print('best accuracy: {}'.format(epochs_test_acc[best_val_idx]))
    return epochs_test_losses[best_val_idx], epochs_test_acc[best_val_idx]



def get_test_score(net, data_loader, epsilon):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    test_total = 0
    test_true = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            y_pred = net(images)

            predictions = torch.argmax(y_pred, dim = 1)
            test_total += labels.shape[0]
            test_true += torch.sum((predictions == labels.to(device)).int())

            ### attack ###
            classifier = PyTorchClassifier(
                model=net,
                # clip_values=(min_pixel_value, max_pixel_value),
                loss=criterion,
                optimizer=optimizer,
                input_shape=tuple(images.shape[1:]),#(3, 32, 32),
                nb_classes=y_pred.shape[1],
            )
            
            attack = FastGradientMethod(estimator=classifier, eps=epsilon)

            with torch.enable_grad():
                x_attack = attack.generate(x=images.cpu().numpy())
            x_attack = torch.tensor(x_attack).to(device)
            y_pred = net(x_attack)
            # test_loss = criterion(y_pred, labels.to(device))
            predictions = torch.argmax(y_pred, dim=1)
            test_total += labels.shape[0]
            test_true += torch.sum((predictions == labels.to(device)).int())
            # test_batch_loss += test_loss.item()

        epochs_test_acc = test_true / test_total
        return epochs_test_acc.item()






