from torch import nn
from torchvision import transforms

# SignGuard CNN class
class SigCNN(nn.Module):
  def __init__(self, input_channels, num_classes):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same"),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="same"),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same"),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=9216, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=128, out_features=2)
    )
    
  def forward(self, x):
    return self.classifier(self.features(x))

#Custom Invert for our Data Transform
class CustomInvert:
  def __call__(self, image):
    invert_image = transforms.functional.invert(image)
    for i in range(invert_image.shape[1]):
      for j in range(invert_image.shape[2]):
        if (invert_image[0][i][j] <= 0.19607843137):
          invert_image[0][i][j] = 0
        else:
          invert_image[0][i][j] = 1
    return invert_image

# Creating Transform
signature_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.Resize((96, 192)),
                                          transforms.ToTensor(),
                                          CustomInvert()])