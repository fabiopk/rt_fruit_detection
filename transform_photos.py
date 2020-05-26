from PIL import Image
import os

# Small script created to downsize the resolution of the photos I took at home
# Goes to path (expects only images there) open and resize the images


path = 'D:/temp'

for i, file in enumerate(os.listdir(path)):
    print(file)
    img = Image.open(os.path.join(path, file))
    img = img.resize((512, 384))
    img.save('bgs/' + str(i) + ".jpg", "JPEG")
