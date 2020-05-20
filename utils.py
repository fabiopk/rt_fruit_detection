import matplotlib.pyplot as plt


def plot_batch(batch, columns, rows):
    permuted = batch.permute(1, 2, 0)
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows + 1):
        img = permuted[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
