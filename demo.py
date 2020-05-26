import numpy as np
import cv2
import torch
from FruitModel import FruitModel
from torchvision import transforms
from utils import classes
import matplotlib.pyplot as plt

# Setup video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Load model
model = FruitModel()
model.load_state_dict(torch.load('models/fruit_bg_net.pt')['state_dict'])
model = model.cuda()
model = model.eval()

# Center crop the image to be sent to the network
crop_size = 300


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    w, h, _ = frame.shape
    starting_point = ((h-crop_size)//2, (w-crop_size)//2)
    end_point = (starting_point[0] + crop_size, starting_point[1] + crop_size)
    image = cv2.rectangle(frame, starting_point, end_point, (0, 0, 255), 2)
    cv2.imshow('frame', image)

    # Permute color channels
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Transform to torch Tensor
    torch_frame = transforms.ToPILImage()(frame)
    torch_frame = transforms.CenterCrop((crop_size, crop_size))(torch_frame)
    cv2.imshow('subframe', cv2.cvtColor(
        np.array(torch_frame), cv2.COLOR_RGB2BGR))  # Shows the portion croped in a different window
    torch_frame = transforms.Resize((100, 100))(torch_frame)
    torch_frame = transforms.ToTensor()(torch_frame)
    torch_frame = torch_frame.unsqueeze(0)

    # Compute
    output = model(torch_frame.cuda())
    _, idx = torch.topk(output, 3)
    print([classes[x] for x in idx[0]])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
