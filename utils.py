import matplotlib.pyplot as plt
import torch


def plot_batch(batch, columns, rows):
    permuted = batch.permute(0, 2, 3, 1)
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows + 1):
        img = permuted[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)


def calculate_loss_and_accuracy(loader, model, criterion, stop_at=1200, print_every=99999):

    correct = 0
    total = 0
    steps = 0
    total_loss = 0

    sz = len(loader)

    for inputs, labels in loader:

        if total % print_every == 0 and total > 0:
            accuracy = 100 * correct / total
            print(accuracy)

        if total >= stop_at:
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Forward pass only to get logits/output
        outputs = model(inputs)

        # Get Loss for validation data
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)
        steps += 1

        correct += (predicted == labels).sum().item()

        del outputs, loss, _, predicted

    accuracy = 100 * correct / total
    return total_loss/steps, accuracy
