import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils


class Auto_encoder(nn.Module):
    """
    A simple auto-encoder using deconvolution as decoder
    This program is designed for MNIST
    """

    def __init__(self):
        super(Auto_encoder, self).__init__()
        ## encoder: downsample the input so its shape is shrinked by 2^3 times
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        ## decoder: upsample the feature maps so the shape is extend back to normal
        self.deconv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=0)
        # It's recommended that padding=(kernel_size-1)/2, output_padding=stride-1
        self.bn_1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1)
        # self.bn_3=nn.BatchNorm(64)

        self.images_count = 0  # for images visualization

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.MSE = nn.MSELoss(reduction='mean')

    def forward(self, input):
        features = self.relu(self.bn1(self.conv1(input)))
        features = self.relu(self.bn2(self.conv2(features)))
        features = self.relu(self.bn3(self.conv3(features)))
        code = self.relu(self.bn4(self.conv4(features)))
        # print(code.shape)

        features = self.relu(self.bn_1(self.deconv1(code)))
        features = self.relu(self.bn_2(self.deconv2(features)))
        output = self.sigmoid(self.deconv3(features))
        return output

    def reconstruction_loss(self, images, de_images):
        if self.images_count % 10 == 0:
            output_images = de_images.cpu()
            utils.save_image(output_images, 'reconstruction_images.png')
        self.images_count += 1
        loss = self.MSE(images, de_images)
        return loss


import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 30
batch_size = 150
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(root='./mnist-torch', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist-torch', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = Auto_encoder().to(device)  # .half()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)

total_p = sum([param.nelement() for param in model.parameters()])
print('Number of params: %.2fM' % (total_p / 1e6))


def train(epoch):
    model.train()
    total_step = len(train_loader)
    current_lr = learning_rate
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # .half()
        # labels_v=to_one_hot(labels,10).to(device)

        outputs = model(images)
        loss = model.reconstruction_loss(images, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch:{}, step[{}/{}] Loss:{:.4f}'
              .format(epoch, i, total_step, loss.item()))


def test(epoch):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            loss = model.reconstruction_loss(images, outputs)

        print('Loss:{:.4f}%'.format(loss.item()))
    return loss


temp = 0
for i in range(num_epochs):
    train(i)
    loss = test(i)
    scheduler.step(loss)

total_p = sum([param.nelement() for param in model.parameters()])
print('Number of params: %.2fM' % (total_p / 1e6))