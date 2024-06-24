import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from models.moco import MoCo
# from models.resnet import resnet18
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import os

def main():
    device = torch.device("cpu")

    # 定义数据增强
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载CIFAR-100数据集
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

    # 定义ResNet-18模型和MoCo算法
    model = models.resnet18
    moco = MoCo(base_encoder=model,projection_dim = 128).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(moco.parameters(), lr=0.001)

    # 使用TensorBoard进行可视化
    writer = SummaryWriter(log_dir='./logs_moco')

    # 自监督学习训练过程
    num_epochs = 200
    for epoch in range(num_epochs):
        total_loss = 0.0
        moco.train()
        for images, _ in train_loader:
            images = images.to(device)
            im_q, im_k = torch.split(images, images.size(0) // 2, dim=0)
            optimizer.zero_grad()
            logits, labels = moco(im_q, im_k)
            loss = criterion(logits, labels)
            # optimizer.zero_grad()
            # loss = moco(images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 保存自监督学习训练好的模型
    torch.save(moco.state_dict(), 'moco.pth')
    writer.close()

    #线性分类
    # 加载CIFAR-100测试集
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 冻结ResNet-18特征提取器
    for param in moco.encoder_q.parameters():
        param.requires_grad = False

    # 线性分类器
    classifier = nn.Linear(512, 100).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # 线性分类器训练过程
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        classifier.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = moco.encoder_q(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # 在测试集上评估性能
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                features = moco.encoder_q(images)
                outputs = classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
        print(f'Test Accuracy: {test_accuracy:.2f}%')

    writer.close()

if __name__ == '__main__':
    main()