import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from timm.models.vision_transformer import vit_small_patch16_224
import numpy as np

# CutMix数据增强
def cutmix_data(x, y, alpha=1.0):
    indices = torch.randperm(x.size(0))
    shuffled_x = x[indices]
    shuffled_y = y[indices]
    lam = np.random.beta(alpha, alpha)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = shuffled_x[:, :, bbx1:bbx2, bby1:bby2]

    return x, y, shuffled_y, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Mixup数据增强
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

# 加载数据集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 定义预训练的ViT Small模型
class ViTCIFAR(nn.Module):
    def __init__(self):
        super(ViTCIFAR, self).__init__()
        self.model = vit_small_patch16_224(pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 100)

    def forward(self, x):
        return self.model(x)

net = ViTCIFAR()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 初始化TensorBoard
writer = SummaryWriter('runs/experiment_vit_small_enhance_pretrained')

# 记录最高准确率
best_acc = 0

# 训练模型
for epoch in range(100):  # 共训练100个epoch
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 随机选择使用CutMix或Mixup
        if np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
        else:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (lam * (predicted == targets_a).sum().item() + (1 - lam) * (predicted == targets_b).sum().item())

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

    # 保存验证集上最高准确率的模型
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(net.state_dict(), 'vit_small_enhance_pretrained_cifar.pth')

writer.close()
print('Finished Training')
print(f'Best Validation Accuracy: {best_acc:.2f}%')
