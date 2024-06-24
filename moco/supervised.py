import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
# from models.moco import MoCo
# from models.resnet import resnet18
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
def main():
    device = torch.device("cpu")
    # model_supervised = torchvision.models.resnet18(pretrained=True)
    '''
    model_scratch = resnet18()
    optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.001)
    '''
    #可视化 tensorboard --logdir=./logs_moco

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

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    # 加载ImageNet预训练的ResNet-18模型
    # model_supervised = models.resnet18(pretrained=True).to(device)
    model_supervised = models.resnet18(pretrained=True).to(device)

    # 冻结特征提取器
    for param in model_supervised.parameters():
        param.requires_grad = False

    # 线性分类器
    classifier_supervised = nn.Linear(1000, 100).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(classifier_supervised.parameters(), lr=0.1)
    optimizer = optim.SGD(classifier_supervised.parameters(), lr=20)#, momentum=0.9, weight_decay=5e-4

    writer = SummaryWriter(log_dir='./logs_supervised')

    # 线性分类器训练过程
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        classifier_supervised.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = model_supervised(images)
            outputs = classifier_supervised(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/Loss_Supervised', avg_loss, epoch)
        writer.add_scalar('Train/Accuracy_Supervised', train_accuracy, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # 在测试集上评估性能
        classifier_supervised.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                features = model_supervised(images)
                outputs = classifier_supervised(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        writer.add_scalar('Test/Accuracy_Supervised', test_accuracy, epoch)
        print(f'Test Accuracy: {test_accuracy:.2f}%')

    writer.close()
    # torch.save(model_s.state_dict(), 'moco.pth')
    torch.save(classifier_supervised.state_dict(), 'classifier_supervised.pth')
    torch.save(model_supervised.state_dict(), 'model_supervised.pth')

if __name__ == '__main__':
    main()