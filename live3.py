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
model.load_state_dict(torch.load('models/fruit_light_net.pt')['state_dict'])
model = model.cuda()
model = model.eval()

# Skip frames to slow down
frame_number = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Computes every 10 frames
    frame_number += 1
    if frame_number % 10:
        continue

    # Transform to torch Tensor
    torch_frame = transforms.ToPILImage()(frame)
    torch_frame = transforms.CenterCrop((300, 300))(torch_frame)
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
