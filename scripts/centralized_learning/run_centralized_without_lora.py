import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser(description="Centralized training configuration")
parser.add_argument("--cuda_device", type=str, default="2", help="Visible GPU device ID.")
parser.add_argument("--data_dir", type=str, default="./data", help="Dataset storage path.")
parser.add_argument("--model_name", type=str, default="vit-base-patch16-224", help="Pretrained model name.")
parser.add_argument("--num_labels", type=int, help="Number of output labels for the classifier.")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum value.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
parser.add_argument("--num_epochs", type=int, default=2, help="Number of centralized training epochs.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--save_dir", type=str, default="./outputs", help="Directory for saving outputs.")
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "MNIST", "STL10", "SVHN", "tiny-imagenet-200"], help="Dataset name.")
parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "AdamW"], help="Optimizer type.")
args = parser.parse_args()
import glob
for file_path in glob.glob(os.path.join(args.save_dir, '*.pth')):
    try:
        os.remove(file_path)
        print(f"Removed file: {file_path}")
    except Exception as e:
        print(f"Failed to remove file: {file_path} - {str(e)}")
for file_path in glob.glob(os.path.join(args.save_dir, '*.npy')):
    try:
        os.remove(file_path)
        print(f'npy: {file_path}')
    except Exception as e:
        print(f'npy: {file_path} - {str(e)}')
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = args.data_dir
if args.dataset == 'cifar10':
    if args.model_name == 'vit-base-patch16-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif args.model_name == 'swin-base-patch4-window7-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    full_test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    args.num_labels = 10
elif args.dataset == 'cifar100':
    if args.model_name == 'vit-base-patch16-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif args.model_name == 'swin-base-patch4-window7-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    full_test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    args.num_labels = 100
elif args.dataset == 'MNIST':
    if args.model_name == 'vit-base-patch16-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    elif args.model_name == 'swin-base-patch4-window7-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    args.num_labels = 10
elif args.dataset == 'STL10':
    if args.model_name == 'vit-base-patch16-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))])
    elif args.model_name == 'swin-base-patch4-window7-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_train_dataset = datasets.STL10(root=data_dir, split='train', download=True, transform=transform)
    full_test_dataset = datasets.STL10(root=data_dir, split='test', download=True, transform=transform)
    args.num_labels = 10
elif args.dataset == 'SVHN':
    if args.model_name == 'vit-base-patch16-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))])
    elif args.model_name == 'swin-base-patch4-window7-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    full_test_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
    args.num_labels = 10
elif args.dataset == 'tiny-imagenet-200':
    from PIL import Image
    if args.model_name == 'vit-base-patch16-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    elif args.model_name == 'swin-base-patch4-window7-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    class TinyImageNetDataset(torch.utils.data.Dataset):

        def __init__(self, root_dir, split='train', transform=None):
            self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200')
            self.transform = transform
            self.split = split
            self.classes = sorted(os.listdir(os.path.join(self.root_dir, 'train')))
            self.class_to_idx = {cls: i for (i, cls) in enumerate(self.classes)}
            self.images = []
            self.labels = []
            if split == 'train':
                for cls in self.classes:
                    cls_dir = os.path.join(self.root_dir, 'train', cls, 'images')
                    for img_name in os.listdir(cls_dir):
                        if img_name.endswith('.JPEG'):
                            self.images.append(os.path.join(cls_dir, img_name))
                            self.labels.append(self.class_to_idx[cls])
            elif split == 'val':
                with open(os.path.join(self.root_dir, 'val', 'val_annotations.txt'), 'r') as f:
                    for line in f:
                        (img_name, cls, _, _, _, _) = line.strip().split('\t')
                        self.images.append(os.path.join(self.root_dir, 'val', 'images', img_name))
                        self.labels.append(self.class_to_idx[cls])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return (image, label)
    full_train_dataset = TinyImageNetDataset(root_dir=data_dir, split='train', transform=transform)
    full_test_dataset = TinyImageNetDataset(root_dir=data_dir, split='val', transform=transform)
    args.num_labels = 200

def get_model():
    import transformers
    transformers.logging.set_verbosity_error()
    if args.model_name == 'vit-base-patch16-224':
        model = ViTForImageClassification.from_pretrained(args.model_name, num_labels=args.num_labels, ignore_mismatched_sizes=True)
        model.classifier = nn.Linear(model.classifier.in_features, args.num_labels)
    elif args.model_name == 'swin-base-patch4-window7-224':
        from transformers import SwinForImageClassification
        model = SwinForImageClassification.from_pretrained(args.model_name, num_labels=args.num_labels, ignore_mismatched_sizes=True)
        model.classifier = nn.Linear(model.classifier.in_features, args.num_labels)
    for param in model.parameters():
        param.requires_grad = True
    return model.to(device)

def main():
    start_time = time.time()
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)
    model = get_model()
    print('Processing...')
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_accuracy = 0.0
    train_loss_history = []
    test_accuracy_history = []
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        for (inputs, labels) in train_loader:
            (inputs, labels) = (inputs.to(device), labels.to(device))
            outputs = model(inputs).logits
            loss = nn.CrossEntropyLoss()(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        avg_train_loss = train_loss / batch_count
        train_loss_history.append(avg_train_loss)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in test_loader:
                (inputs, labels) = (inputs.to(device), labels.to(device))
                outputs = model(inputs).logits
                (_, predicted) = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        test_accuracy_history.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(args.save_dir, f'best_model_epoch{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
        print(f'Epoch {epoch + 1}/{args.num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Accuracy: {accuracy:.2f}% | ')
    duration = time.time() - start_time
    print(f'\n, {duration:.2f} ')
    print(f': {best_accuracy:.2f}%')
    save_centralized_results(args, best_accuracy, duration, train_loss_history, test_accuracy_history)

def save_centralized_results(args, best_accuracy, duration, train_loss_history, test_accuracy_history):
    os.makedirs(args.save_dir, exist_ok=True)
    result_filename = f'Centralized_FullFineTuning_batch_size{args.batch_size}_epochs{args.num_epochs}_lr={args.lr}_{args.model_name}_{args.dataset}.txt'
    result_path = os.path.join(args.save_dir, result_filename)
    content = ':\n'
    content += '=' * 50 + '\n'
    for arg in vars(args):
        content += f'{arg}: {getattr(args, arg)}\n'
    content += '=' * 50 + '\n\n'
    content += ':\n'
    content += '=' * 50 + '\n'
    content += f': {duration:.2f} \n'
    content += f': {best_accuracy:.2f}%\n\n'
    content += ':\n'
    content += '=' * 50 + '\n'
    for epoch in range(len(train_loss_history)):
        content += f'Epoch {epoch + 1}/{args.num_epochs} | Train Loss: {train_loss_history[epoch]:.4f} | Test Accuracy: {test_accuracy_history[epoch]:.2f}%\n'
    content += '\n'
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'\n: {result_path}')
if __name__ == '__main__':
    print('\n:')
    print('=' * 50)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('=' * 50)
    main()
    for file_path in glob.glob(os.path.join(args.save_dir, '*.pth')):
        try:
            os.remove(file_path)
            print('No additional status message.')
        except Exception as e:
            print(f"Failed to remove file: {file_path} - {str(e)}")
    for file_path in glob.glob(os.path.join(args.save_dir, '*.npy')):
        try:
            os.remove(file_path)
            print('No additional status message.')
        except Exception as e:
            print(f'npy: {file_path} - {str(e)}')
