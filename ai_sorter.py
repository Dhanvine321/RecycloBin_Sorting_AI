# use model of accuracy 93% to predict the class of the garbage in camera 
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch

# load the model
model = torch.load('garbage_classification.pth', map_location=torch.device('cpu'))
model.eval()

# load the labels
class_names = ['e-waste', 'glass', 'metal', 'paper', 'plastic']

# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# use the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert the frame to tensor
    inputs = transform(frame)
    inputs = inputs.unsqueeze(0)

    # get the prediction
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    # get the class name
    class_name = class_names[predicted]

    # put the class name on the frame
    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # convert the frame back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Camera closed')