import torch


def get_model_accuracy(model, dataloader, sample_size = float('inf')):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            if total >= sample_size:
                break
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct // total
    print(f'Accuracy of the network on {total} images: {accuracy} %')
    return accuracy
