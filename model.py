
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes_kazu = ["0","1","2","3","4","5","6","7","8","9"]
classes_en = ["zero","one","two","three","four","five","six","seven","eight","nine"]
n_class = len(classes_kazu)
img_size = 28

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(img_size*img_size, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, img_size*img_size)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net()

net.load_state_dict(torch.load(
    "tegaki_model.pth", map_location=torch.device("cpu")))

def predict(img):
  img = img.convert("L")
  img = img.resize((img_size, img_size))
  transform = transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.0),(1.0))])
  img = transform(img)
  x = img.reshape(1, 1, img_size, img_size)

  net.eval()
  y = net(x)

  y_prob = torch.nn.functional.softmax(torch.squeeze(y))
  sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)
  return [(classes_kazu[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
