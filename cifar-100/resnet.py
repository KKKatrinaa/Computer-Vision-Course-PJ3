import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet34

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

# 加载数据集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 定义ResNet模型
class ResNetCIFAR(nn.Module):
    def __init__(self):
        super(ResNetCIFAR, self).__init__()
        self.model = resnet34(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 100)

    def forward(self, x):
        return self.model(x)

net = ResNetCIFAR()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 初始化TensorBoard
writer = SummaryWriter('runs/experiment_resnet34')

# 训练模型
for epoch in range(100):  # 共训练100个epoch
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total

    writer.add_scalar('training loss', train_loss, epoch)
    writer.add_scalar('training accuracy', train_accuracy, epoch)


    scheduler.step()

    # 验证模型
    net.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(testloader)
    val_accuracy = 100 * correct / total

    writer.add_scalar('validation loss', val_loss, epoch)
    writer.add_scalar('validation accuracy', val_accuracy, epoch)

    print(f'Epoch {epoch + 1}: Training Loss: {train_loss:.3f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')


writer.close()
print('Finished Training')

# 保存模型
PATH = './resnet34_cifar.pth'
torch.save(net.state_dict(), PATH)
