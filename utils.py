import matplotlib.pyplot as plt
import torch


def plot_batch(batch, columns, rows):
    """Function to plot in a grid {columns} x {rows} a given batch of images

    Arguments:
        batch {tensor} -- The batch of images
        columns {int} -- number of columns
        rows {int} -- number of rows
    """
    permuted = batch.permute(0, 2, 3, 1)
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows + 1):
        img = permuted[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)


def calculate_loss_and_accuracy(loader, model, criterion, stop_at=1200, print_every=99999):
    """Calculates loss and accuracy for a given data_loader, model and criterion

    Arguments:
        loader {DataLoader} -- The torchvision DataLoader instance
        model {nn.Module} -- The model used in the benchmark
        criterion {_WeightedLoss} -- The criterion used (for calculating average Loss)

    Keyword Arguments:
        stop_at {int} -- Breaks after {stop_at} batches (default: {1200})
        print_every {int} -- Print after {print_every} batches (default: {99999})

    Returns:
        [average_loss] -- Average loss by the model
        [accuracy] -- Accuracy in the data
    """
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

# List of classes (ordered)


classes = ['Apple Braeburn',
           'Apple Crimson Snow',
           'Apple Golden 1',
           'Apple Golden 2',
           'Apple Golden 3',
           'Apple Granny Smith',
           'Apple Pink Lady',
           'Apple Red 1',
           'Apple Red 2',
           'Apple Red 3',
           'Apple Red Delicious',
           'Apple Red Yellow 1',
           'Apple Red Yellow 2',
           'Apricot',
           'Avocado',
           'Avocado ripe',
           'Banana',
           'Banana Lady Finger',
           'Banana Red',
           'Beetroot',
           'Blueberry',
           'Cactus fruit',
           'Cantaloupe 1',
           'Cantaloupe 2',
           'Carambula',
           'Cauliflower',
           'Cherry 1',
           'Cherry 2',
           'Cherry Rainier',
           'Cherry Wax Black',
           'Cherry Wax Red',
           'Cherry Wax Yellow',
           'Chestnut',
           'Clementine',
           'Cocos',
           'Corn',
           'Corn Husk',
           'Cucumber Ripe',
           'Cucumber Ripe 2',
           'Dates',
           'Eggplant',
           'Fig',
           'Ginger Root',
           'Granadilla',
           'Grape Blue',
           'Grape Pink',
           'Grape White',
           'Grape White 2',
           'Grape White 3',
           'Grape White 4',
           'Grapefruit Pink',
           'Grapefruit White',
           'Guava',
           'Hazelnut',
           'Huckleberry',
           'Kaki',
           'Kiwi',
           'Kohlrabi',
           'Kumquats',
           'Lemon',
           'Lemon Meyer',
           'Limes',
           'Lychee',
           'Mandarine',
           'Mango',
           'Mango Red',
           'Mangostan',
           'Maracuja',
           'Melon Piel de Sapo',
           'Mulberry',
           'Nectarine',
           'Nectarine Flat',
           'Nut Forest',
           'Nut Pecan',
           'Onion Red',
           'Onion Red Peeled',
           'Onion White',
           'Orange',
           'Papaya',
           'Passion Fruit',
           'Peach',
           'Peach 2',
           'Peach Flat',
           'Pear',
           'Pear 2',
           'Pear Abate',
           'Pear Forelle',
           'Pear Kaiser',
           'Pear Monster',
           'Pear Red',
           'Pear Stone',
           'Pear Williams',
           'Pepino',
           'Pepper Green',
           'Pepper Orange',
           'Pepper Red',
           'Pepper Yellow',
           'Physalis',
           'Physalis with Husk',
           'Pineapple',
           'Pineapple Mini',
           'Pitahaya Red',
           'Plum',
           'Plum 2',
           'Plum 3',
           'Pomegranate',
           'Pomelo Sweetie',
           'Potato Red',
           'Potato Red Washed',
           'Potato Sweet',
           'Potato White',
           'Quince',
           'Rambutan',
           'Raspberry',
           'Redcurrant',
           'Salak',
           'Strawberry',
           'Strawberry Wedge',
           'Tamarillo',
           'Tangelo',
           'Tomato 1',
           'Tomato 2',
           'Tomato 3',
           'Tomato 4',
           'Tomato Cherry Red',
           'Tomato Heart',
           'Tomato Maroon',
           'Tomato Yellow',
           'Tomato not Ripened',
           'Walnut',
           'Watermelon']


classes_to_idx = {'Apple Braeburn': 0,
                  'Apple Crimson Snow': 1,
                  'Apple Golden 1': 2,
                  'Apple Golden 2': 3,
                  'Apple Golden 3': 4,
                  'Apple Granny Smith': 5,
                  'Apple Pink Lady': 6,
                  'Apple Red 1': 7,
                  'Apple Red 2': 8,
                  'Apple Red 3': 9,
                  'Apple Red Delicious': 10,
                  'Apple Red Yellow 1': 11,
                  'Apple Red Yellow 2': 12,
                  'Apricot': 13,
                  'Avocado': 14,
                  'Avocado ripe': 15,
                  'Banana': 16,
                  'Banana Lady Finger': 17,
                  'Banana Red': 18,
                  'Beetroot': 19,
                  'Blueberry': 20,
                  'Cactus fruit': 21,
                  'Cantaloupe 1': 22,
                  'Cantaloupe 2': 23,
                  'Carambula': 24,
                  'Cauliflower': 25,
                  'Cherry 1': 26,
                  'Cherry 2': 27,
                  'Cherry Rainier': 28,
                  'Cherry Wax Black': 29,
                  'Cherry Wax Red': 30,
                  'Cherry Wax Yellow': 31,
                  'Chestnut': 32,
                  'Clementine': 33,
                  'Cocos': 34,
                  'Corn': 35,
                  'Corn Husk': 36,
                  'Cucumber Ripe': 37,
                  'Cucumber Ripe 2': 38,
                  'Dates': 39,
                  'Eggplant': 40,
                  'Fig': 41,
                  'Ginger Root': 42,
                  'Granadilla': 43,
                  'Grape Blue': 44,
                  'Grape Pink': 45,
                  'Grape White': 46,
                  'Grape White 2': 47,
                  'Grape White 3': 48,
                  'Grape White 4': 49,
                  'Grapefruit Pink': 50,
                  'Grapefruit White': 51,
                  'Guava': 52,
                  'Hazelnut': 53,
                  'Huckleberry': 54,
                  'Kaki': 55,
                  'Kiwi': 56,
                  'Kohlrabi': 57,
                  'Kumquats': 58,
                  'Lemon': 59,
                  'Lemon Meyer': 60,
                  'Limes': 61,
                  'Lychee': 62,
                  'Mandarine': 63,
                  'Mango': 64,
                  'Mango Red': 65,
                  'Mangostan': 66,
                  'Maracuja': 67,
                  'Melon Piel de Sapo': 68,
                  'Mulberry': 69,
                  'Nectarine': 70,
                  'Nectarine Flat': 71,
                  'Nut Forest': 72,
                  'Nut Pecan': 73,
                  'Onion Red': 74,
                  'Onion Red Peeled': 75,
                  'Onion White': 76,
                  'Orange': 77,
                  'Papaya': 78,
                  'Passion Fruit': 79,
                  'Peach': 80,
                  'Peach 2': 81,
                  'Peach Flat': 82,
                  'Pear': 83,
                  'Pear 2': 84,
                  'Pear Abate': 85,
                  'Pear Forelle': 86,
                  'Pear Kaiser': 87,
                  'Pear Monster': 88,
                  'Pear Red': 89,
                  'Pear Stone': 90,
                  'Pear Williams': 91,
                  'Pepino': 92,
                  'Pepper Green': 93,
                  'Pepper Orange': 94,
                  'Pepper Red': 95,
                  'Pepper Yellow': 96,
                  'Physalis': 97,
                  'Physalis with Husk': 98,
                  'Pineapple': 99,
                  'Pineapple Mini': 100,
                  'Pitahaya Red': 101,
                  'Plum': 102,
                  'Plum 2': 103,
                  'Plum 3': 104,
                  'Pomegranate': 105,
                  'Pomelo Sweetie': 106,
                  'Potato Red': 107,
                  'Potato Red Washed': 108,
                  'Potato Sweet': 109,
                  'Potato White': 110,
                  'Quince': 111,
                  'Rambutan': 112,
                  'Raspberry': 113,
                  'Redcurrant': 114,
                  'Salak': 115,
                  'Strawberry': 116,
                  'Strawberry Wedge': 117,
                  'Tamarillo': 118,
                  'Tangelo': 119,
                  'Tomato 1': 120,
                  'Tomato 2': 121,
                  'Tomato 3': 122,
                  'Tomato 4': 123,
                  'Tomato Cherry Red': 124,
                  'Tomato Heart': 125,
                  'Tomato Maroon': 126,
                  'Tomato Yellow': 127,
                  'Tomato not Ripened': 128,
                  'Walnut': 129,
                  'Watermelon': 130}
