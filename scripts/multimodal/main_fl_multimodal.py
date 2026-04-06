
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model
import numpy as np
import flwr as fl
from flwr.common import Metrics
from typing import Dict, List, Tuple, Optional
import os
import time
import argparse

parser = argparse.ArgumentParser(description="Federated Learning ViT-LoRA Configuration")

parser.add_argument('--cuda_device', type=str, default='2', help='Visible GPU device ID (default: 3)')
parser.add_argument('--training_mode', type=str, default='Homo', choices=['Centralized', 'Homo'], help='Training mode: Centralized or Homo (federated learning) (default: Homo)')
# Data path
parser.add_argument('--data_dir', type=str, default='./data', help='Dataset storage path (default: "./data")')
# LoRA configuration
parser.add_argument('--lora_r_client', type=int, default=4, help='Rank size of the local client LoRA matrices (default: 4)')
parser.add_argument('--lora_r_server', type=int, default=6, help='Rank size of the server LoRA matrices (default: 4)')
parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA scaling factor (default: 16)')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate (default: 0.1)')
parser.add_argument('--target_modules', type=list, default=["query", "value"], help='Target modules to apply LoRA to (default: ["query", "value"])')
# Model configuration
parser.add_argument('--model_name', type=str, default='swin-base-patch4-window7-224', help='Pretrained ViT model name (default: "vit-base-patch16-224", "swin-base-patch4-window7-224")')
parser.add_argument('--num_labels', type=int, help='Output dimension of the classification head (10 for CIFAR-10, 100 for CIFAR-100)')
# Federated learning strategy
parser.add_argument('--fraction_fit', type=float, default=1.0, help='Fraction of clients participating in each round (default: 1.0)')
parser.add_argument('--min_fit_clients', type=int, default=3, help='Minimum number of clients for training per round (default: 3)')
parser.add_argument('--min_evaluate_clients', type=int, default=3, help='Minimum number of clients for evaluation per round (default: 3)')
parser.add_argument('--num_rounds', type=int, default=2, help='Total number of federated learning rounds (default: 10)')
# Training hyperparameters
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
parser.add_argument('--local_epochs', type=int, default=1, help='Number of local training epochs per client (default: 1)')
parser.add_argument('--batch_size', type=int, default=128, help='Training/testing batch size (default: 128)')
# Client settings
parser.add_argument('--num_clients', type=int, default=3, help='Total number of clients (default: 3)')
parser.add_argument('--algorithm', type=str, default='FlexLoRA', choices=['FEDIT', 'FLORA', 'ILORA','LoRA_FAIR','FFA-LORA','FlexLoRA'], help='Select algorithm type (default: FEDIT)')
parser.add_argument('--save_dir', type=str, default='./result', help='Model save directory path (default: ./results/checkpoints)')
parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10', 'cifar100','MNIST','STL10','SVHN','tiny-imagenet-200'],help='Select dataset type (default: cifar10)')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'],help='Select optimizer type: SGD or AdamW (default: SGD)')
parser.add_argument('--distribution', type=str, default='NON-IID', choices=['IID', 'NON-IID'],
                    help='IID and Dirichlet-based NON-IID')
parser.add_argument('--alpha', type=float, default=0.5, help='Controls the degree of Non-IID; smaller values mean stronger Non-IID')
parser.add_argument('--heterogeneous_rank', type=str, default='True', choices=['True', 'False'],
                    help='Whether to enable heterogeneous-rank mode: True or False')
parser.add_argument('--heterogeneous_rank_clients', type=str, default='2,4,6',
                    help='Rank size of the local client LoRA matrices (default: 4)')
parser.add_argument('--use_control', type=str, default='False', choices=['True', 'False'],
                    help='Whether to enable the control algorithm to prevent client drift (default: False)')
parser.add_argument('--lambda_reg', type=float, default=0.01,
                    help='Regularization coefficient in LoRA_FAIR')
parser.add_argument('--lora_scale_factor', type=float, default=0.5,
                   help='Scaling factor s for QR initialization (default: 1.0)')
# NEW: TensorBoardX
from tensorboardX import SummaryWriter

# Add a log directory argument after argparse
parser.add_argument('--log_dir', type=str, default='./runs',
                    help='TensorBoard log directory (default: ./runs)')


args = parser.parse_args()
if args.heterogeneous_rank == 'True':
    #print("Set heterogeneous ranks:")
    args.heterogeneous_rank_clients = [int(r) for r in args.heterogeneous_rank_clients.split(',')]
else:
    #print("Set homogeneous ranks: all clients use the same rank")
    args.heterogeneous_rank_clients = [args.lora_r_client] * args.num_clients
import glob
# Delete all .pth files
for file_path in glob.glob(os.path.join(args.save_dir, "*.pth")):
    try:
        os.remove(file_path)
        print(f"Removed old file: {file_path}")
    except Exception as e:
        print(f"Failed to clean file: {file_path} - {str(e)}")
# Delete all .npy files (newly added part)
for file_path in glob.glob(os.path.join(args.save_dir, "*.npy")):
    try:
        os.remove(file_path)
        print(f"Removed old npy file: {file_path}")
    except Exception as e:
        print(f"Failed to clean npy file: {file_path} - {str(e)}")
# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full dataset
data_dir = args.data_dir
if args.dataset == 'cifar10':
    if args.model_name == "vit-base-patch16-224":
        # Data preprocessing
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif args.model_name == "swin-base-patch4-window7-224":
        transform = transforms.Compose([
            transforms.Resize(224),  # or adjust to a smaller resolution (e.g. 64x64)
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
        ])
    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    full_test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    args.num_labels = 10
elif args.dataset == 'cifar100':
    if args.model_name == "vit-base-patch16-224":
        # Data preprocessing
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif args.model_name == "swin-base-patch4-window7-224":
        transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
        ])
    full_train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    full_test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    #full_train_dataset = Subset(full_train_dataset, indices=list(range(20)))
    #full_test_dataset = Subset(full_test_dataset, indices=list(range(20)))
    args.num_labels = 100
elif args.dataset == 'MNIST':
    if args.model_name == "vit-base-patch16-224":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),  # MNIST-specific
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # MNIST-specific
        ])
    elif args.model_name == "swin-base-patch4-window7-224":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
        ])
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    #full_train_dataset = Subset(full_train_dataset, indices=list(range(20)))
    #full_test_dataset = Subset(full_test_dataset, indices=list(range(20)))
    args.num_labels = 10  # MNIST has 10 classes
elif args.dataset == 'STL10':
    # STL-10 preprocessing (image size 96x96, needs resizing to 224x224)
    if args.model_name == "vit-base-patch16-224":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
    elif args.model_name == "swin-base-patch4-window7-224":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
        ])
    # Load the STL-10 training set (labeled data)
    full_train_dataset = datasets.STL10(root=data_dir,split='train',download=True,transform=transform)
    # Load the STL-10 test set
    full_test_dataset = datasets.STL10(root=data_dir,split='test',download=True,transform=transform)
    # full_train_dataset = Subset(full_train_dataset, indices=list(range(20)))
    # full_test_dataset = Subset(full_test_dataset, indices=list(range(20)))
    args.num_labels = 10  # STL-10 has 10 classes
elif args.dataset == 'SVHN':
    if args.model_name == "vit-base-patch16-224":
        # SVHN preprocessing
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
    elif args.model_name == "swin-base-patch4-window7-224":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
        ])
    # Load the SVHN training set
    full_train_dataset = datasets.SVHN(root=data_dir,split='train',download=True,transform=transform)
    # Load the SVHN test set
    full_test_dataset = datasets.SVHN(root=data_dir,split='test',download=True,transform=transform)
    #full_train_dataset = Subset(full_train_dataset, indices=list(range(20)))
    #full_test_dataset = Subset(full_test_dataset, indices=list(range(20)))
    args.num_labels = 10  # SVHN has 10 digit classes
elif args.dataset == 'tiny-imagenet-200':
    from PIL import Image  # Add this line
    if args.model_name == "vit-base-patch16-224":
        # Tiny-ImageNet preprocessing
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Use standard ImageNet normalization
        ])
    elif args.model_name == "swin-base-patch4-window7-224":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
        ])
    # Custom Tiny-ImageNet dataset class
    class TinyImageNetDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, split='train', transform=None):
            self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200')
            self.transform = transform
            self.split = split
            # Load class information
            self.classes = sorted(os.listdir(os.path.join(self.root_dir, 'train')))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            # Load image paths and labels
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
                        img_name, cls, _, _, _, _ = line.strip().split('\t')
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
            return image, label
    # Load the dataset (make sure tiny-imagenet-200.zip has been downloaded and extracted under data_dir)
    full_train_dataset = TinyImageNetDataset(root_dir=data_dir,split='train',transform=transform)
    full_test_dataset = TinyImageNetDataset(root_dir=data_dir,split='val', transform=transform)
    #full_train_dataset = Subset(full_train_dataset, indices=list(range(20)))
    #full_test_dataset = Subset(full_test_dataset, indices=list(range(20)))
    args.num_labels = 200  # Tiny-ImageNet has 200 classes
def split_data(dataset, num_clients=args.num_clients, distribution=args.distribution):
    if distribution == 'IID':
        # IID split - random uniform allocation
        indices = np.random.permutation(len(dataset))
        return np.array_split(indices, num_clients)
    elif distribution == 'NON-IID':
        # Non-IID split - based on the Dirichlet distribution
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100
        elif args.dataset == 'tiny-imagenet-200':
            num_classes = 200

        # Get dataset labels
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

        # Group indices by class
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

        # Generate a Non-IID split using a Dirichlet distribution
        alpha = args.alpha  # Controls the degree of Non-IID; smaller values mean stronger Non-IID
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            # Skip empty classes
            if len(class_indices[c]) == 0:
                continue

            # Generate a Dirichlet distribution for the current class
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

            # Ensure each client receives at least one sample
            proportions = np.maximum(proportions, 1e-3)
            proportions = proportions / proportions.sum()

            # Allocate samples according to the proportions
            proportions = (proportions * len(class_indices[c])).astype(int)
            proportions[-1] = len(class_indices[c]) - np.sum(proportions[:-1])

            # Randomly shuffle the indices of the current class
            np.random.shuffle(class_indices[c])

            # Assign samples
            start = 0
            for client_id in range(num_clients):
                end = start + proportions[client_id]
                client_indices[client_id].extend(class_indices[c][start:end])
                start = end

        return [np.array(indices) for indices in client_indices]
    else:
        raise ValueError(f"Unknown data distribution type: {distribution}")

# train_indices = split_data(full_train_dataset)
# test_indices = split_data(full_test_dataset)
train_indices = split_data(full_train_dataset, distribution=args.distribution)
test_indices = split_data(full_test_dataset, distribution=args.distribution)
# Define the ViT model (with LoRA)
def get_model(r):
    import transformers
    transformers.logging.set_verbosity_error()
    if args.model_name=="vit-base-patch16-224":
        model = ViTForImageClassification.from_pretrained(
            args.model_name,
            num_labels=args.num_labels,
            ignore_mismatched_sizes=True,
        )
        model.classifier = nn.Linear(model.classifier.in_features, args.num_labels)
    elif args.model_name== "swin-base-patch4-window7-224":
        from transformers import SwinForImageClassification
        model = SwinForImageClassification.from_pretrained(
            args.model_name,
            num_labels=args.num_labels,
            ignore_mismatched_sizes=True
        )
        model.classifier = nn.Linear(model.classifier.in_features, args.num_labels)
    merged_model_path = os.path.join(args.save_dir, "merged_model.pth")
    if os.path.exists(merged_model_path) and args.algorithm != "ILORA":
        #print("Load the merged model as the base...")
        model.load_state_dict(torch.load(merged_model_path))
    # else:
    #     print("Do not load the merged model")
    # LoRA configuration
    if args.model_name == "vit-base-patch16-224":
        lora_config = LoraConfig(
            r=r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
            modules_to_save=["classifier"]
        )
    elif args.model_name == "swin-base-patch4-window7-224":
        lora_config = LoraConfig(
            r=r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["attention.self.query", "attention.self.value"],  # Module names for Swin
            bias="none",
            modules_to_save=["classifier"]
        )
    model = get_peft_model(model, lora_config)
    if args.algorithm == 'ILORA':
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and isinstance(module.base_layer, nn.Linear):
                #print(f"\nProcessing layer: {name}")
                W = module.base_layer.weight.data.clone()
                d, k = W.shape
                # 1. QR decomposition
                Q, R = torch.linalg.qr(W, mode='reduced')
                Q_r = Q[:, :r]
                R_r = R[:r, :]
                # 2. Adaptive scaling
                target_norm = 1e-4 * (d * k) ** 0.5  # Target norm
                current_norm = torch.norm(Q_r @ R_r)# Use the Frobenius norm (torch.norm) to compute the norm of Q_r @ R_r correctly # 1. First normalize
                scale = target_norm / current_norm
                # 2. Symmetric scaling
                scale_factor = torch.sqrt(scale)
                R_r = R_r * scale_factor
                Q_r = Q_r * scale_factor
                # 3. Apply global intensity control

                delta_W = Q_r @ R_r*args.lora_scale_factor
                module.base_layer.weight.data = W - delta_W
                # 4. Initialize LoRA parameters
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A = module.lora_A['default'].weight
                    lora_B = module.lora_B['default'].weight

                    if lora_A.shape == (r, k):
                        lora_A.data = R_r
                    elif lora_A.shape == (k, r):
                        lora_A.data = R_r.T

                    if lora_B.shape == (d, r):
                        lora_B.data = Q_r
                    elif lora_B.shape == (r, d):
                        lora_B.data = Q_r.T

                    # Validation metrics
                    delta_W = Q_r @ R_r

    return model.to(device)
# Custom strategy (for handling LoRA parameters)
class FedLoRAStrategy(fl.server.strategy.FedAvg):
    def __init__(self,lora_r_server ,use_control,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = get_model(lora_r_server)
        self.use_control = use_control
        # Initialize the global control variable c
        #Control step 1: initialize the global server control variable
        if use_control == 'True':
            # Get the shapes of all trainable parameters (LoRA parameters + classifier parameters)
            trainable_params = [
                p.detach().cpu().numpy()
                for n, p in self.global_model.named_parameters()
                if p.requires_grad and ('lora' in n or 'classifier' in n)
            ]
            self.c = [np.zeros_like(p) for p in trainable_params]

        else:
            self.c = None
    def initialize_parameters(self, client_manager):
        # Export the trainable parameters of the global model (r=6) as the initial global parameters
        arrs = [
            p.detach().cpu().numpy()
            for n, p in self.global_model.named_parameters()
            if p.requires_grad
        ]
        from flwr.common import ndarrays_to_parameters
        return ndarrays_to_parameters(arrs)
    def aggregate_fit(self, server_round, results, failures):
        # Print information based on the algorithm type
        if args.algorithm == 'FEDIT':
            #print("Currently using the FedIT (Federated Incremental Training) algorithm")
            #print("FedIT aggregation weight verification:")
            total_examples = sum([metrics.num_examples for _, metrics in results])
            # Create a mapping from client IDs to weights
            if args.heterogeneous_rank == 'True':
                #print("FEDIT uses heterogeneous ranks, enabling zero padding:")
                max_rank = max(args.heterogeneous_rank_clients)
                #print("Max rank for zero padding in FEDIT:", max_rank)
                client_weights = {
                    fit_res.metrics["client_id"]: fit_res.num_examples / total_examples
                    for _, fit_res in results
                }

                #print(f"client_weights: {client_weights}")
                aggregated_params = None
                for client_idx, (client_proxy, fit_res) in enumerate(results):
                    client_idx = fit_res.metrics["client_id"]
                    p_k = client_weights[client_idx]  # Get the current client weight
                    #print(f"\nClient {client_idx} parameter analysis:")
                    #print(f"-Weight: {p_k:.4f}")
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    #print(f"- LoRA rank: {client_rank}")
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    # Print the parameter types and shapes of the current client
                    # print(f"\nDetailed LoRA parameter analysis:")
                    # for param in client_params:
                    #     print("param.shape=",param.shape)
                    # print(f"\nStart padding:")
                    for i, param in enumerate(client_params):
                        if len(param.shape) == 2:
                            if param.shape[0] == client_rank and param.shape[0] < max_rank:  # Need to pad rows
                                padding = ((0, max_rank - param.shape[0]), (0, 0))
                                padded_param = np.pad(param, padding, 'constant')
                                client_params[i] = padded_param  # Replace the original parameter
                            elif param.shape[1] == client_rank and param.shape[1] < max_rank:  # Need to pad columns
                                padding = ((0, 0), (0, max_rank - param.shape[1]))
                                padded_param = np.pad(param, padding, 'constant')
                                client_params[i] = padded_param  # Replace the original parameter
                    # print(f"\nLoRA parameter analysis after padding:")
                    # for param in client_params:
                    #     print("param.shape=", param.shape)
                    if aggregated_params is None:
                        # First iteration, initialize aggregated parameters
                        aggregated_params = []
                        for param in client_params:
                            weighted_param = p_k * param  # Apply weighting to each parameter
                            aggregated_params.append(weighted_param)
                    else:
                        for i, param in enumerate(client_params):
                            aggregated_params[i] += p_k * param
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                if aggregated_parameters is not None:
                    # Update the global model
                    params_dict = zip([n for n, p in self.global_model.named_parameters() if p.requires_grad],
                                      fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for k, v in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    # Save the model (LoRA part)
                    trainable_state = {
                        k: v for k, v in self.global_model.state_dict().items()
                        if any(n in k for n in ['lora', 'classifier'])  # Match LoRA and classifier parameters
                    }
                    torch.save(trainable_state, os.path.join(args.save_dir, "update_LORA.pth"))
                    # Merge the model and save the merged model
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, "merged_model.pth"))
                    aggregated_metrics = {}
                return aggregated_parameters, aggregated_metrics
        if args.algorithm == 'FLORA':
            #print("Currently using the FLoRA (Federated Low-Rank Adaptation) algorithm")
            if args.heterogeneous_rank == 'True':
                #print("FLORA uses heterogeneous ranks:")
                # 1. Get the parameters and number of samples for all clients
                total_examples = sum([metrics.num_examples for _, metrics in results])
                #print(f"total_examples: {total_examples}")
                client_weights = {
                    fit_res.metrics["client_id"]: fit_res.num_examples / total_examples
                    for _, fit_res in results
                }
                #print(f"client_weights: {client_weights}")
                # 2Parameter aggregation
                aggregated_params = None  # Initialize as None
                for client_idx, (client_proxy, fit_res) in enumerate(results):
                    client_idx = fit_res.metrics["client_id"]
                    p_k = client_weights[client_idx]  # Get the current client weight
                    #print(f"\nClient {client_idx} parameter analysis:")
                    #print(f"-Weight: {p_k:.4f}")
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    #print(f"- LoRA rank: {client_rank}")
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    if aggregated_params is None:
                        # Initialize the aggregated parameters directly from the first client
                        aggregated_params = []
                        for i, param in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:  # LoRA_A
                                    aggregated_params.append(p_k * param)  # Apply weights
                                elif param.shape[1] == client_rank:  # LoRA_B
                                    aggregated_params.append(param)  # Do not apply weights
                                else:  # Classifier
                                    aggregated_params.append(p_k * param)
                            else:
                                aggregated_params.append(p_k * param)
                    else:
                        # Stack/accumulate parameters from subsequent clients
                        for i, param in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:  # LoRA_A
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], p_k * param], axis=0)
                                elif param.shape[1] == client_rank:  # LoRA_B
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], param], axis=1)
                                elif param.shape[0] == args.num_labels:  # Classifier
                                    aggregated_params[i] += p_k * param
                            elif len(param.shape) == 1 and param.shape[0] == args.num_labels:  # Classifier.bias
                                aggregated_params[i] += p_k * param
                # 3Convert ndarrays to parameters
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                if aggregated_parameters is not None:
                    # Update the global model
                    params_dict = zip([n for n, p in self.global_model.named_parameters() if p.requires_grad],
                                      fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for k, v in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    # Save the model (LoRA part)
                    trainable_state = {
                        k: v for k, v in self.global_model.state_dict().items()
                        if any(n in k for n in ['lora', 'classifier'])  # Match LoRA and classifier parameters
                    }
                    torch.save(trainable_state, os.path.join(args.save_dir, "update_LORA.pth"))
                    # Merge the model and save the merged model
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, "merged_model.pth"))
                return aggregated_parameters, aggregated_metrics
        if args.algorithm == 'FFA-LORA':
            #print("Currently using the FFA-LORA algorithm")
            if args.heterogeneous_rank == 'True':
               # print("FFA-LORA uses heterogeneous ranks:")
                # 1. Get the parameters and number of samples for all clients
                total_examples = sum([metrics.num_examples for _, metrics in results])
               # print(f"total_examples: {total_examples}")
                client_weights = {
                    fit_res.metrics["client_id"]: fit_res.num_examples / total_examples
                    for _, fit_res in results
                }
              #  print(f"client_weights: {client_weights}")

                # 2. Determine the maximum rank
                max_rank = max(args.heterogeneous_rank_clients)
                #print(f"Maximum rank: {max_rank}")

                # 3. Parameter aggregation - only aggregate matrix B, keep matrix A unchanged
                aggregated_params = None
                for client_idx, (client_proxy, fit_res) in enumerate(results):
                    client_id = fit_res.metrics["client_id"]
                    p_k = client_weights[client_id]
                  # print(f"\nclient {client_id} parameter analysis:")
                 #   print(f"-Weight: {p_k:.4f}")
                    client_rank = args.heterogeneous_rank_clients[int(client_id)]
                 #   print(f"- LoRA rank: {client_rank}")

                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)

                    # Zero-pad client parameters to the maximum rank
                    filled_params = []
                    for param in client_params:
                        if len(param.shape) == 2:
                            # Matrix B (d, r) -> pad to (d, max_rank)
                            if param.shape[1] == client_rank:
                                padding = ((0, 0), (0, max_rank - client_rank))
                                filled_param = np.pad(param, padding, 'constant')
                                filled_params.append(filled_param)
                            # Matrix A (r, k) -> pad to (max_rank, k)
                            elif param.shape[0] == client_rank:
                                padding = ((0, max_rank - client_rank), (0, 0))
                                filled_param = np.pad(param, padding, 'constant')
                                filled_params.append(filled_param)
                            else:
                                filled_params.append(param)  # Classifier parameters remain unchanged
                        else:
                            filled_params.append(param)  # 1D parameters remain unchanged

                    if aggregated_params is None:
                        # Initialize aggregated parameters
                        aggregated_params = [np.zeros_like(p) for p in filled_params]

                    # Perform weighted average aggregation only on matrix B
                    for i, param in enumerate(filled_params):
                        if len(param.shape) == 2:
                            # Matrix B (d, max_rank)
                            if param.shape[1] == max_rank:
                                aggregated_params[i] += p_k * param
                            # Matrix A (max_rank, k) - keep unchanged, do not aggregate
                            elif param.shape[0] == max_rank:
                                if np.all(aggregated_params[i] == 0):  # If this is the first time, set it to the current value
                                    aggregated_params[i] = param
                            # Aggregate classifier parameters normally
                            else:
                                aggregated_params[i] += p_k * param
                        else:
                            # Aggregate 1D parameters normally
                            aggregated_params[i] += p_k * param

                # 4. Convert ndarrays to parameters
                aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}

                # 5. Update the global model
                if aggregated_parameters is not None:
                    params_dict = zip([n for n, p in self.global_model.named_parameters() if p.requires_grad],
                                      fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for k, v in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)

                    # Save the model (LoRA part)
                    trainable_state = {
                        k: v for k, v in self.global_model.state_dict().items()
                        if any(n in k for n in ['lora', 'classifier'])  # Match LoRA and classifier parameters
                    }
                    torch.save(trainable_state, os.path.join(args.save_dir, "update_LORA.pth"))

                    # Merge the model and save the merged model
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, "merged_model.pth"))

                return aggregated_parameters, aggregated_metrics
        if args.algorithm == 'LoRA_FAIR':
            if args.heterogeneous_rank == 'True':
               # print("Currently using the LoRA_FAIR algorithm (heterogeneous ranks)")
                # 1. Get the parameters and number of samples for all clients
                total_examples = sum([metrics.num_examples for _, metrics in results])
               # print(f"total_examples: {total_examples}")

                # 2. Compute each client's weight (based on sample count)
                client_weights = {
                    fit_res.metrics["client_id"]: fit_res.num_examples / total_examples
                    for _, fit_res in results
                }

                # 3. Determine the maximum rank (the largest among all client ranks)
                max_rank = max(args.heterogeneous_rank_clients)
               # print(f"Maximum rank: {max_rank}")

                # Get parameters from the first client to determine the number of layers
                first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
                num_layers = (len(first_client_params) - 2) // 4
               # print(f"num_layers: {num_layers}")

                # Get the actual dimension of each layer
                layer_dims = []
                for layer_idx in range(num_layers):
                    base_idx = layer_idx * 4
                    q_A = first_client_params[base_idx]
                    layer_dims.append(q_A.shape[1])  # Get the actual dimension of this layer
                #print(f"Layer dimensions: {layer_dims}")

                # Initialize storage structures using dynamic dimensions
                avg_A_query = [np.zeros((max_rank, dim)) for dim in layer_dims]
                avg_B_query = [np.zeros((dim, max_rank)) for dim in layer_dims]
                avg_A_value = [np.zeros((max_rank, dim)) for dim in layer_dims]
                avg_B_value = [np.zeros((dim, max_rank)) for dim in layer_dims]
                global_delta_w_query = [np.zeros((dim, dim)) for dim in layer_dims]
                global_delta_w_value = [np.zeros((dim, dim)) for dim in layer_dims]
                # 5. Iterate over each client, collect parameters, and pad them
                for _, fit_res in results:
                    client_id = fit_res.metrics["client_id"]
                    p_k = client_weights[client_id]
                    client_rank = args.heterogeneous_rank_clients[int(client_id)]

                   # print(f"\nclient {client_id} (rank={client_rank}) parameter analysis:")
                   # print(f"-Weight: {p_k:.4f}")

                    # Get client parameters (NumPy arrays)
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)

                    # Iterate over each layer
                    for layer_idx in range(num_layers):
                        # Compute the index
                        base_idx = layer_idx * 4
                        dim = layer_dims[layer_idx]  # Get the actual dimension of the current layer
                        # Get the parameters of the current layer
                        q_A = client_params[base_idx]  # shape=(client_rank, 768)
                        q_B = client_params[base_idx + 1]  # shape=(768, client_rank)
                        v_A = client_params[base_idx + 2]  # shape=(client_rank, 768)
                        v_B = client_params[base_idx + 3]  # shape=(768, client_rank)

                        # Zero-pad to the maximum rank
                        # Zero-pad to the maximum rank
                        q_A_padded = np.zeros((max_rank, dim))
                        q_A_padded[:client_rank, :] = q_A

                        q_B_padded = np.zeros((dim, max_rank))
                        q_B_padded[:, :client_rank] = q_B

                        v_A_padded = np.zeros((max_rank, dim))
                        v_A_padded[:client_rank, :] = v_A

                        v_B_padded = np.zeros((dim, max_rank))
                        v_B_padded[:, :client_rank] = v_B

                        # Accumulate A and B (weighted)
                        avg_A_query[layer_idx] += p_k * q_A_padded
                        avg_B_query[layer_idx] += p_k * q_B_padded
                        avg_A_value[layer_idx] += p_k * v_A_padded
                        avg_B_value[layer_idx] += p_k * v_B_padded

                        # Compute delta_W and accumulate it (weighted)
                        delta_w_query = q_B_padded @ q_A_padded
                        delta_w_value = v_B_padded @ v_A_padded
                        global_delta_w_query[layer_idx] += p_k * delta_w_query
                        global_delta_w_value[layer_idx] += p_k * delta_w_value

                # 6. Compute the correction term delta_B (Equation 8 in the paper)
                lambda_reg = args.lambda_reg  # Regularization coefficient
                updated_B_query = []
                updated_B_value = []

                for layer_idx in range(num_layers):
                    # Query part
                    R_query = global_delta_w_query[layer_idx] - (avg_B_query[layer_idx] @ avg_A_query[layer_idx])
                    # Compute the ridge regression solution
                    I = np.eye(max_rank)
                    A = avg_A_query[layer_idx]
                    inv_part = np.linalg.inv(A @ A.T + lambda_reg * I)
                    delta_B_query = R_query @ A.T @ inv_part

                    # Value part
                    R_value = global_delta_w_value[layer_idx] - (avg_B_value[layer_idx] @ avg_A_value[layer_idx])
                    A = avg_A_value[layer_idx]
                    inv_part = np.linalg.inv(A @ A.T + lambda_reg * I)
                    delta_B_value = R_value @ A.T @ inv_part

                    # Update matrix B
                    updated_B_query.append(avg_B_query[layer_idx] + delta_B_query)
                    updated_B_value.append(avg_B_value[layer_idx] + delta_B_value)

                # 7. Prepare the parameters to return to clients (padded low-rank matrices)
                aggregated_params = []
                for layer_idx in range(num_layers):
                    # Query part
                    aggregated_params.append(avg_A_query[layer_idx])  # Matrix A (max_rank, 768)
                    aggregated_params.append(updated_B_query[layer_idx])  # Updated matrix B (768, max_rank)

                    # Value part
                    aggregated_params.append(avg_A_value[layer_idx])  # Matrix A (max_rank, 768)
                    aggregated_params.append(updated_B_value[layer_idx])  # Updated matrix B (768, max_rank)

                # 8. Add classifier parameters (weighted average)
                classifier_weight = np.zeros_like(first_client_params[-2])
                classifier_bias = np.zeros_like(first_client_params[-1])

                for _, fit_res in results:
                    p_k = client_weights[fit_res.metrics["client_id"]]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    classifier_weight += p_k * client_params[-2]
                    classifier_bias += p_k * client_params[-1]

                aggregated_params.append(classifier_weight)
                aggregated_params.append(classifier_bias)

                # 9. Convert to Flower parameter format and return
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)

                # 10. Update the global model using padded parameters
                params_dict = zip(
                    [name for name, param in self.global_model.named_parameters() if param.requires_grad],
                    aggregated_params
                )
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                self.global_model.load_state_dict(state_dict, strict=False)

                # 11. Save the model
                trainable_state = {
                    k: v for k, v in self.global_model.state_dict().items()
                    if any(n in k for n in ['lora', 'classifier'])
                }
                torch.save(trainable_state, os.path.join(args.save_dir, "update_LORA.pth"))
                # Merge and save the model
                import copy
                model_copy = copy.deepcopy(self.global_model)
                merged_model = model_copy.merge_and_unload()
                torch.save(merged_model.state_dict(), os.path.join(args.save_dir, "merged_model.pth"))
                # Return the aggregation result
                aggregated_metrics = {}
                return aggregated_parameters, aggregated_metrics

        if args.algorithm == 'ILORA':

            if args.heterogeneous_rank == 'True':
                #print("ILORA heterogeneous case:")
                if args.use_control == 'True':
                    #print("ILORA heterogeneous case + control:")
                    # Load global c from the local file
                    c_save_path = os.path.join(args.save_dir, "global_c.npy")
                    try:
                        self.c = np.load(c_save_path, allow_pickle=True).tolist()
                        #print("Successfully loaded c")
                    except:
                        print(f"Initialize c")
                max_rank = max(args.heterogeneous_rank_clients)
                # 1. Get the parameters and number of samples for all clients
                total_examples = sum([metrics.num_examples for _, metrics in results])
                #print(f"total_examples: {total_examples}")
                client_weights = {
                    fit_res.metrics["client_id"]: fit_res.num_examples / total_examples
                    for _, fit_res in results
                }
                #print(f"client_weights: {client_weights}")
                if args.use_control == 'True':
                    #print("\n-> Control: start aggregating client delta_ci and updating global c (padding required for different ranks)")
                    delta_c_aggregated = [np.zeros_like(p) for p in self.c]
                    for client_idx, (_, fit_res) in enumerate(results):
                        client_idx = fit_res.metrics["client_id"]
                        delta_ci_path = os.path.join(args.save_dir, f"client_{client_idx}_delta_ci.npy")
                        try:
                            delta_ci = np.load(delta_ci_path, allow_pickle=True).tolist()
                           # print(f"\n-> Checking delta_ci values for client {client_idx}:")
                            client_rank = args.heterogeneous_rank_clients[client_idx]
                            # for i, ci_val in enumerate(delta_ci[:3]):  # Only inspect the first 3
                            for i, ci_val in enumerate(delta_ci):  # Only inspect the first 3
                                #################Zero-pad delta_ci##################
                                # print(f"Original ci[{i}].shape: {ci_val.shape}")  # Based on this inspection
                                # print(f"  - Contains NaN: {np.isnan(ci_val).any()}")
                                # print(f"  - Contains Inf: {np.isinf(ci_val).any()}")
                                # print(f"  - Value range: min={np.nanmin(ci_val):.6f}, max={np.nanmax(ci_val):.6f}")
                                if len(ci_val.shape) == 2:
                                    if ci_val.shape[0] == client_rank and ci_val.shape[0] < max_rank:  # Need to pad rows
                                        padding = ((0, max_rank - ci_val.shape[0]), (0, 0))
                                        padded_param = np.pad(ci_val, padding, 'constant')
                                        delta_ci[i] = padded_param  # Replace the original parameter
                                        #print(f"Row padding: {ci_val.shape} -> {padded_param.shape}")
                                    elif ci_val.shape[1] == client_rank and ci_val.shape[1] < max_rank:  # Need to pad columns
                                        padding = ((0, 0), (0, max_rank - ci_val.shape[1]))
                                        padded_param = np.pad(ci_val, padding, 'constant')
                                        delta_ci[i] = padded_param  # Replace the original parameter
                                        #print(f"Column padding: {ci_val.shape} -> {delta_ci[i].shape}")
                            # print(f"\nParameter analysis after padding:")
                            # for i, ci_val in enumerate(delta_ci):
                            #     print("ci_val.shape=", delta_ci[i].shape)

                            # weight =client_weights[client_idx]
                            # Weighted aggregation of delta_ci
                            for i in range(len(delta_c_aggregated)):
                                delta_c_aggregated[i] += (1 / args.num_clients) * delta_ci[i]  # This needs to be adjusted
                        except Exception as e:
                            print(f"Unable to load the delta_ci file for client {client_idx}: {str(e)}")
                            continue

                    for i in range(len(self.c)):
                        self.c[i] += delta_c_aggregated[i]  # Aggregate correctly
                    if args.use_control == 'True':
                        # #print("\nSome values of global c after the update (control variable):")
                        # for i, c in enumerate(self.c[:3]):  # Print the first 3
                        # print(f"  c[{i}].shape: {c.shape}, mean: {np.mean(c):.4f}")
                        # New: save global c to a local file
                        c_save_path = os.path.join(args.save_dir, "global_c.npy")
                        np.save(c_save_path, np.array(self.c, dtype=object))  # Save as a NumPy object array
                        #print(f"Saved global c to {c_save_path}")
                # 2Parameter aggregation
                aggregated_params = None  # Initialize as None
                for client_idx, (client_proxy, fit_res) in enumerate(results):
                    client_idx = fit_res.metrics["client_id"]
                    p_k = client_weights[client_idx]  # Get the current client weight
                    #print(f"\nClient {client_idx} parameter analysis:")
                   # print(f"-Weight: {p_k:.4f}")
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    #print(f"- LoRA rank: {client_rank}")
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    if aggregated_params is None:
                        # Initialize the aggregated parameters directly from the first client
                        aggregated_params = []
                        for i, param in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:  # LoRA_A
                                    aggregated_params.append(p_k * param)  # Apply weights
                                elif param.shape[1] == client_rank:  # LoRA_B
                                    aggregated_params.append(param)  # Do not apply weights
                                else:  # Classifier
                                    aggregated_params.append(p_k * param)
                            else:
                                aggregated_params.append(p_k * param)
                    else:
                        # Stack/accumulate parameters from subsequent clients
                        for i, param in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:  # LoRA_A
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], p_k * param], axis=0)
                                elif param.shape[1] == client_rank:  # LoRA_B
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], param], axis=1)
                                elif param.shape[0] == args.num_labels:  # Classifier
                                    aggregated_params[i] += p_k * param
                            elif len(param.shape) == 1 and param.shape[0] == args.num_labels:  # Classifier.bias
                                aggregated_params[i] += p_k * param

                num_layers = (len(aggregated_params) - 2) // 4  # Compute the total number of layers (excluding the final classifier)
                # print(f"num_layers: {num_layers}")  # 12 layers
                # Create save paths for three ranks
                Decomposition_save_paths = [
                    os.path.join(args.save_dir, f"update_LORA_Decomposition_client{i}.pth")
                    for i in range(len(args.heterogeneous_rank_clients))
                ]
                # Create corresponding model parameters for each rank
                for rank_idx, target_rank in enumerate(args.heterogeneous_rank_clients):

                    #print(f"\nProcess QR decomposition for rank {target_rank}...")
                    # Copy aggregated parameters
                    rank_specific_params = [param.copy() for param in aggregated_params]
                    for layer_idx in range(num_layers):
                        base_idx = layer_idx * 4
                        # Query part
                        q_A = rank_specific_params[base_idx]  # shape=(12,768)
                        q_B = rank_specific_params[base_idx + 1]  # shape=(768,12)
                        Q_full = q_B @ q_A  # (768,768)
                        # QR decomposition
                        Q_q, R_q = np.linalg.qr(Q_full, mode='reduced')
                        q_A_prime = R_q[:target_rank, :]
                        q_B_prime = Q_q[:, :target_rank]
                        rank_specific_params[base_idx] = q_A_prime
                        rank_specific_params[base_idx + 1] = q_B_prime
                        # Value part
                        v_A = rank_specific_params[base_idx + 2]  # shape=(12,768)
                        v_B = rank_specific_params[base_idx + 3]  # shape=(768,12)
                        V_full = v_B @ v_A  # (768,768)
                        # QR decomposition
                        Q_v, R_v = np.linalg.qr(V_full, mode='reduced')
                        v_A_prime = R_v[:target_rank, :]
                        v_B_prime = Q_v[:, :target_rank]
                        rank_specific_params[base_idx + 2] = v_A_prime
                        rank_specific_params[base_idx + 3] = v_B_prime
                    if target_rank == max_rank:
                        max_rank_params = rank_specific_params
                    # Convert to PyTorch model parameters and save
                    params_dict = zip(
                        [n for n, p in self.global_model.named_parameters() if p.requires_grad],
                        rank_specific_params
                    )
                    state_dict = {k: torch.tensor(v) for k, v in params_dict}
                    # Save only LoRA and classifier parameters
                    trainable_state = {
                        k: v for k, v in state_dict.items()
                        if any(n in k for n in ['lora', 'classifier'])
                    }
                    # Save parameters for the specific rank
                    torch.save(trainable_state, Decomposition_save_paths[rank_idx])
                    # print(f"Saved parameters for rank {target_rank} to {Decomposition_save_paths[rank_idx]}")
                # In the final round, use the maximum-rank parameters to update the global model
                if server_round == args.num_rounds - 1 and max_rank_params is not None:
                    #print("\nFinal round: update the global model using the maximum-rank parameters...")
                    from flwr.common import ndarrays_to_parameters
                    max_rank_parameters = ndarrays_to_parameters(max_rank_params)
                    # Update the global model
                    params_dict = zip(
                        [n for n, p in self.global_model.named_parameters() if p.requires_grad],
                        fl.common.parameters_to_ndarrays(max_rank_parameters)
                    )
                    state_dict = {k: torch.tensor(v) for k, v in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    # Save the final model
                    final_model_path = os.path.join(args.save_dir, "final_global_model.pth")
                    torch.save(self.global_model.state_dict(), final_model_path)
                    #print(f"Saved the final global model to {final_model_path}")
                # Return default parameters (any rank can be returned here because each client will load its own corresponding parameters)
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                return aggregated_parameters, aggregated_metrics

        if args.algorithm == 'FlexLoRA':
            # ---------- 1. Collect client parameters and sample counts ----------
            total_examples = sum([metrics.num_examples for _, metrics in results])
            client_weights = {
                fit_res.metrics["client_id"]: fit_res.num_examples / total_examples
                for _, fit_res in results
            }

            # Get the first client's parameters to determine the structure and dimensions of each layer
            first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
            num_layers = (len(first_client_params) - 2) // 4   # 4 LoRA parameters per layer, with the final 2 being the classifier

            # Dynamically obtain the hidden dimension of each layer (from the second dimension of matrix A)
            layer_dims = []
            for layer_idx in range(num_layers):
                base = layer_idx * 4
                q_A = first_client_params[base]          # (r, hidden_dim)
                layer_dims.append(q_A.shape[1])          # Hidden dimension of the current layer

            # Initialize accumulators: each layer corresponds to a zero matrix of shape (dim, dim)
            W_query_sum = [np.zeros((dim, dim)) for dim in layer_dims]
            W_value_sum = [np.zeros((dim, dim)) for dim in layer_dims]
            classifier_weight_sum = np.zeros_like(first_client_params[-2])
            classifier_bias_sum = np.zeros_like(first_client_params[-1])

            # ---------- 2. Iterate over clients and accumulate weighted delta_W ----------
            for _, fit_res in results:
                client_id = fit_res.metrics["client_id"]
                p_k = client_weights[client_id]
                client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)

                for layer_idx in range(num_layers):
                    base = layer_idx * 4
                    dim = layer_dims[layer_idx]
                    # Query
                    q_A = client_params[base]            # (r, dim)
                    q_B = client_params[base+1]          # (dim, r)
                    W_q = q_B @ q_A                       # (dim, dim)
                    W_query_sum[layer_idx] += p_k * W_q

                    # Value
                    v_A = client_params[base+2]          # (r, dim)
                    v_B = client_params[base+3]          # (dim, r)
                    W_v = v_B @ v_A
                    W_value_sum[layer_idx] += p_k * W_v

                # Classifier
                classifier_weight_sum += p_k * client_params[-2]
                classifier_bias_sum += p_k * client_params[-1]

            # ---------- 3. Perform SVD on W for each layer and save U, S, Vt ----------
            U_q_list, S_q_list, Vt_q_list = [], [], []
            U_v_list, S_v_list, Vt_v_list = [], [], []
            for layer_idx in range(num_layers):
                U_q, S_q, Vt_q = np.linalg.svd(W_query_sum[layer_idx], full_matrices=False)
                U_q_list.append(U_q)
                S_q_list.append(S_q)
                Vt_q_list.append(Vt_q)

                U_v, S_v, Vt_v = np.linalg.svd(W_value_sum[layer_idx], full_matrices=False)
                U_v_list.append(U_v)
                S_v_list.append(S_v)
                Vt_v_list.append(Vt_v)

            # ---------- 4. Build server model parameters (rank = args.lora_r_server) ----------
            server_params = []
            for layer_idx in range(num_layers):
                dim = layer_dims[layer_idx]
                r_server = args.lora_r_server
                # Query
                U_q = U_q_list[layer_idx][:, :r_server]
                S_q = S_q_list[layer_idx][:r_server]
                Vt_q = Vt_q_list[layer_idx][:r_server, :]
                A_q_server = np.diag(np.sqrt(S_q)) @ Vt_q   # (r_server, dim)
                B_q_server = U_q @ np.diag(np.sqrt(S_q))    # (dim, r_server)
                server_params.append(A_q_server)
                server_params.append(B_q_server)

                # Value
                U_v = U_v_list[layer_idx][:, :r_server]
                S_v = S_v_list[layer_idx][:r_server]
                Vt_v = Vt_v_list[layer_idx][:r_server, :]
                A_v_server = np.diag(np.sqrt(S_v)) @ Vt_v   # (r_server, dim)
                B_v_server = U_v @ np.diag(np.sqrt(S_v))    # (dim, r_server)
                server_params.append(A_v_server)
                server_params.append(B_v_server)

            server_params.append(classifier_weight_sum)
            server_params.append(classifier_bias_sum)

            # Convert to Flower parameter format
            from flwr.common import ndarrays_to_parameters
            aggregated_parameters = ndarrays_to_parameters(server_params)

            # Update the global model
            trainable_names = [n for n, p in self.global_model.named_parameters() if p.requires_grad]
            params_dict = zip(trainable_names, server_params)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.global_model.load_state_dict(state_dict, strict=False)

            # Save the model (LoRA part)
            trainable_state = {
                k: v for k, v in self.global_model.state_dict().items()
                if any(n in k for n in ['lora', 'classifier'])
            }
            torch.save(trainable_state, os.path.join(args.save_dir, "update_LORA.pth"))

            # Merge and save the model
            import copy
            model_copy = copy.deepcopy(self.global_model)
            merged_model = model_copy.merge_and_unload()
            torch.save(merged_model.state_dict(), os.path.join(args.save_dir, "merged_model.pth"))

            # ---------- 5. Save client initialization files ----------
            if args.heterogeneous_rank == 'True':
                # Heterogeneous: save the file corresponding to each client's rank separately
                for client_idx, rank in enumerate(args.heterogeneous_rank_clients):
                    client_params = []
                    for layer_idx in range(num_layers):
                        dim = layer_dims[layer_idx]
                        # Query
                        U_q = U_q_list[layer_idx][:, :rank]
                        S_q = S_q_list[layer_idx][:rank]
                        Vt_q = Vt_q_list[layer_idx][:rank, :]
                        A_q_client = np.diag(np.sqrt(S_q)) @ Vt_q   # (rank, dim)
                        B_q_client = U_q @ np.diag(np.sqrt(S_q))    # (dim, rank)
                        client_params.append(A_q_client)
                        client_params.append(B_q_client)

                        # Value
                        U_v = U_v_list[layer_idx][:, :rank]
                        S_v = S_v_list[layer_idx][:rank]
                        Vt_v = Vt_v_list[layer_idx][:rank, :]
                        A_v_client = np.diag(np.sqrt(S_v)) @ Vt_v   # (rank, dim)
                        B_v_client = U_v @ np.diag(np.sqrt(S_v))    # (dim, rank)
                        client_params.append(A_v_client)
                        client_params.append(B_v_client)

                    client_params.append(classifier_weight_sum)
                    client_params.append(classifier_bias_sum)

                    # Build the state_dict
                    client_state_dict = {}
                    param_names = [n for n, p in self.global_model.named_parameters() if p.requires_grad]
                    for name, param in zip(param_names, client_params):
                        client_state_dict[name] = torch.tensor(param)

                    save_path = os.path.join(args.save_dir, f"FlexLoRA_client_{client_idx}_rank_{rank}.pth")
                    torch.save(client_state_dict, save_path)

            aggregated_metrics = {}
            return aggregated_parameters, aggregated_metrics


# Client class
from flwr.client import Client, NumPyClient
class CIFAR10Client(NumPyClient):  # Keep this as NumPyClient
    def __init__(self, cid, train_indices, test_indices):
        self.cid = cid
        #print("cid=",cid)
        client_rank = args.heterogeneous_rank_clients[int(cid)]
        # print(f"Client {cid} initialized successfully!", f"rank={client_rank}")  # Print cid here
        self.model = get_model(client_rank)  # Use the client-specific rank
        # print(f"client {cid} Completed get_model execution")
        #Control step 2: initialize the local client control variable
        if args.use_control == 'True':
            trainable_params = [p.detach().cpu().numpy() for p in self.model.parameters() if p.requires_grad]
            #self.ci = [np.zeros_like(p) for p in trainable_params]
            self.ci = [torch.zeros_like(p).cpu().numpy()
                       for p in self.model.parameters() if p.requires_grad]
        self.train_loader = DataLoader(
            Subset(full_train_dataset, train_indices[cid]),
            batch_size=args.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            Subset(full_test_dataset, test_indices[cid]),
            batch_size=args.batch_size
        )


    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for name, val in self.model.named_parameters() if val.requires_grad]
    def set_parameters(self, parameters):
        params_dict = zip([name for name, param in self.model.named_parameters() if param.requires_grad], parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        #print(f"Client {self.cid}: received parameter shapes = {[p.shape for p in parameters]}")

        if args.algorithm == 'FFA-LORA' and args.heterogeneous_rank == 'True':
            #print(f"FFA-LORA Heterogeneous-rank mode - parameter truncation for client {self.cid}")
            params_dict = zip([name for name, param in self.model.named_parameters() if param.requires_grad],
                              parameters)
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            #print(f"Rank of client {self.cid}: {client_rank}")
            # Get the maximum rank (used to check whether truncation is needed)
            max_rank = max(args.heterogeneous_rank_clients)
            truncated_state_dict = {}
            for (name, _), param in zip(params_dict, parameters):
                if len(param.shape) == 2:
                    # Check whether this is LoRA_A (shape [max_rank, hidden_dim])
                    if param.shape[0] == max_rank and 'lora_A' in name:
                        if param.shape[0] > client_rank:  # Rows need to be truncated
                            truncated_param = param[:client_rank, :]
                            #print(f"LoRA_A Row truncation: {param.shape} -> {truncated_param.shape}")
                        else:
                            truncated_param = param
                        truncated_state_dict[name] = torch.tensor(truncated_param)

                    # Check whether this is LoRA_B (shape [hidden_dim, max_rank])
                    elif param.shape[1] == max_rank and 'lora_B' in name:
                        if param.shape[1] > client_rank:  # Columns need to be truncated
                            truncated_param = param[:, :client_rank]
                            #print(f"LoRA_B Column truncation: {param.shape} -> {truncated_param.shape}")
                        else:
                            truncated_param = param
                        truncated_state_dict[name] = torch.tensor(truncated_param)

                    # Other cases (such as the classifier)
                #     else:
                # #print(f"Non-LoRA parameter: {name} (shape={param.shape})")
                #         truncated_state_dict[name] = torch.tensor(param)
                # else:
                #     # Handle 1D parameters (such as bias)
                # #print(f"Non-matrix parameter: {name} (shape={param.shape})")
                #     truncated_state_dict[name] = torch.tensor(param)

            # Load the truncated parameters
            self.model.load_state_dict(truncated_state_dict, strict=False)

        if args.algorithm == 'LoRA_FAIR' and args.heterogeneous_rank == 'True':
           # print(f"LoRA_FAIR Heterogeneous-rank mode - parameter truncation for client {self.cid}")
            params_dict = zip([name for name, param in self.model.named_parameters() if param.requires_grad],
                              parameters)
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            #print(f"Rank of client {self.cid}: {client_rank}")
            # Get the maximum rank (used to check whether truncation is needed)
            max_rank = max(args.heterogeneous_rank_clients)
            truncated_state_dict = {}
            for (name, _), param in zip(params_dict, parameters):
                if len(param.shape) == 2:
                    # print("debug")
                    # Check whether this is LoRA_A (shape [rank, hidden_dim])
                    if param.shape[0] == max_rank :  # Assume hidden_dim=768
                        # print(f"Detected a LoRA_A parameter: {name} (shape={param.shape})")
                        if param.shape[0] > client_rank:  # Rows need to be truncated
                            truncated_param = param[:client_rank, :]
                           # print(f"Row truncation: {param.shape} -> {truncated_param.shape}")
                        else:
                            truncated_param = param
                        truncated_state_dict[name] = torch.tensor(truncated_param)

                    # Check whether this is LoRA_B (shape [hidden_dim, rank])
                    elif param.shape[1] == max_rank:
                        # print(f"Detected a LoRA_B parameter: {name} (shape={param.shape})")
                        if param.shape[1] > client_rank:  # Columns need to be truncated
                            truncated_param = param[:, :client_rank]
                            #print(f"Column truncation: {param.shape} -> {truncated_param.shape}")
                        else:
                            truncated_param = param
                        truncated_state_dict[name] = torch.tensor(truncated_param)

                    # Other cases (such as the classifier)
                #     else:
                # #print(f"Non-LoRA parameter: {name} (shape={param.shape})")
                #         truncated_state_dict[name] = torch.tensor(param)
                # else:
                #     # Handle 1D parameters (such as bias)
                # #print(f"Non-matrix parameter: {name} (shape={param.shape})")
                #     truncated_state_dict[name] = torch.tensor(param)

            # Load the truncated parameters
            self.model.load_state_dict(truncated_state_dict, strict=False)
        if args.algorithm == 'ILORA':
            if args.heterogeneous_rank == 'True':  # Heterogeneous case
                # Get the current client rank
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                # Build the path of the corresponding parameter file
                param_path = os.path.join(args.save_dir, f"update_LORA_Decomposition_client{self.cid}.pth")
                if os.path.exists(param_path):
                    #print(f"Client {self.cid} is loading decomposition parameters for rank {client_rank}...")
                    trainable_state = torch.load(param_path)
                    self.model.load_state_dict(trainable_state, strict=False)
                # else:
                #     print(f"Warning: parameter file for rank {client_rank} was not found {param_path}")
        #Newly modified
        if args.algorithm == 'FlexLoRA' and args.heterogeneous_rank == 'True':
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            file_path = os.path.join(args.save_dir, f"FlexLoRA_client_{self.cid}_rank_{client_rank}.pth")
            if os.path.exists(file_path):
                # Load the decomposed parameters saved for this client
                state_dict = torch.load(file_path)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                # First round: truncate the server parameters to obtain local low-rank parameters
                params_dict = zip([name for name, param in self.model.named_parameters() if param.requires_grad], parameters)
                truncated_state_dict = {}
                for (name, _), param in zip(params_dict, parameters):
                    if len(param.shape) == 2:
                        if 'lora_A' in name:          # shape (server_rank, hidden_dim)
                            if param.shape[0] > client_rank:
                                param = param[:client_rank, :]
                        elif 'lora_B' in name:        # shape (hidden_dim, server_rank)
                            if param.shape[1] > client_rank:
                                param = param[:, :client_rank]
                    truncated_state_dict[name] = torch.tensor(param)
                self.model.load_state_dict(truncated_state_dict, strict=False)


    def fit(self, parameters, config):
        #print("Start executing the fit function")
        self.set_parameters(parameters)
        if args.algorithm == 'ILORA':
            print("ILORANo reinitialization is required!")
        elif args.algorithm == 'LoRA_FAIR':
            print("LoRA_FAIRNo reinitialization is required!")
        elif args.algorithm == 'FFA-LORA':
            print("FFA-LORANo reinitialization is required!")
        elif args.algorithm == 'FEDIT':
            print("FEDITNo reinitialization is required!")
        elif args.algorithm == 'FlexLoRA':
            print("FlexLoRANo reinitialization is required!")
        else:
            print("Reinitialization is required!")
            #self.model=get_model(args.lora_r_client)
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            #print("client_rank value in fit:", client_rank)
            self.model = get_model(client_rank)
        initial_params = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
        # Control step 3.2: get the global control variable c (passed from the server)
        if args.use_control == 'True':
            ci_path = os.path.join(args.save_dir, f"client_{self.cid}_ci.npy")
            if os.path.exists(ci_path):
                self.ci = np.load(ci_path, allow_pickle=True).tolist()
               # print(f"Client {self.cid} successfully loaded historical ci values")
            # Load global c from the local file
            c_save_path = os.path.join(args.save_dir, "global_c.npy")
            try:
                c = np.load(c_save_path, allow_pickle=True).tolist()
                #print(f"Client {self.cid} successfully loaded global c locally")
                # Truncate c values based on the client rank
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                max_rank = max(args.heterogeneous_rank_clients)
                truncated_c = []
                for param_idx, param in enumerate(c):
                    if len(param.shape) == 2:
                        # Handle LoRA_A parameters (rank, hidden_dim)
                        if param.shape[0] == max_rank and param.shape[0] > client_rank:  # Rows need to be truncated
                            truncated_param = param[:client_rank, :]
                            #print(f"Truncated parameter {param_idx}: {param.shape} -> {truncated_param.shape} (Row truncation)")
                        # Handle LoRA_B parameters (hidden_dim, rank)
                        elif param.shape[1] == max_rank and param.shape[1] > client_rank:  # Columns need to be truncated
                            truncated_param = param[:, :client_rank]
                            #print(f"Truncated parameter {param_idx}: {param.shape} -> {truncated_param.shape} (Column truncation)")
                        else:
                            truncated_param = param
                        truncated_c.append(truncated_param)
                    else:
                        truncated_c.append(param)  # Non-matrix parameters remain unchanged
                c = truncated_c
                #print(f"Client {self.cid} (rank={client_rank}) truncated global c values")
            except:
                c = self.ci
                #print(f"Client {self.cid} failed to load global c and will use the initial ci")
            # Ensure that c and ci have matching shapes
            if len(c) != len(self.ci):
                #print("Warning: loaded c and ci have mismatched lengths, using the initial ci")
                c = self.ci
        import copy
        if args.use_control == 'True':
            original_ci = copy.deepcopy(self.ci)
            # ========== OPTION I implementation ==========
            control_option = 1
            if control_option == 1:
                #print("Use control Option I: compute gradients over the entire dataset")
                # Save the training state
                original_mode = self.model.training
                self.model.eval()  # Use eval mode to ensure consistent results
                # Compute gradients over the entire local dataset
                self.model.zero_grad()
                total_samples = 0
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.model(inputs).logits
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    # Compute gradients without updating parameters
                    loss.backward()
                    total_samples += labels.size(0)
                # Average gradients (Option I)
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad /= total_samples

                # Update ci to the current gradients
                new_ci = [
                    param.grad.clone().detach().cpu().numpy()
                    for param in self.model.parameters()
                    if param.requires_grad
                ]
                # Restore the original training state
                self.model.train(original_mode)

                #print(f"Client {self.cid} completed Option I gradient computation")
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.get("lr", args.lr),
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.get("lr", args.lr),
                weight_decay=args.weight_decay
            )


        self.model.train()
        K = 0
        for epoch in range(config.get("epochs", args.local_epochs)):
            # NEW: Track local training loss
            running_loss = 0.0
            batch_count = 0
            for inputs, labels in self.train_loader:
                K += 1
                inputs, labels = inputs.to(device), labels.to(device)


                optimizer.zero_grad()
                outputs = self.model(inputs).logits
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                #Control step 4: apply control correction
                if args.use_control == 'True':
                    with torch.no_grad():
                        for param, ci_val, c_val in zip(
                                [p for p in self.model.parameters() if p.requires_grad],
                                self.ci,
                                c
                        ):
                            param.grad.add_(torch.tensor(c_val - ci_val).to(device))
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
            # NEW: Compute local validation accuracy (a quick full evaluation)
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs).logits
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            local_acc = correct / total if total > 0 else 0.0
            self.model.train()


        # Control step 5: update ci:
            if args.use_control == 'True':
                # Save ci to a local file
                self.ci = new_ci
                ci_save_path = os.path.join(args.save_dir, f"client_{self.cid}_ci.npy")
                np.save(ci_save_path, np.array(self.ci, dtype=object))
               # print(f"Client {self.cid} saved ci to {ci_save_path}")
                delta_ci = [new_ci - old_ci for new_ci, old_ci in zip(self.ci, original_ci)]
                # Save delta_ci to a local file
                delta_ci_path = os.path.join(args.save_dir, f"client_{self.cid}_delta_ci.npy")
                np.save(delta_ci_path, np.array(delta_ci, dtype=object))
                #print(f"Client {self.cid} saved delta_ci to {delta_ci_path}")

        return self.get_parameters({}), len(self.train_loader.dataset), {"client_id": int(self.cid)}



    def evaluate(self, parameters, config):
       # print("Start executing evaluate")
        self.set_parameters(parameters)
        loss, accuracy = 0.0, 0.0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs).logits
                loss += nn.CrossEntropyLoss()(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        accuracy /= total
        return loss, total, {"accuracy": accuracy}
# Weighted average aggregation metric
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}
# Server-side evaluation function
def get_evaluate_fn(test_loader):
    # Ensure the log directory exists
    os.makedirs(os.path.join(args.log_dir, "server"), exist_ok=True)
    server_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "server"))

    # NEW: Use a mutable object to persist batch counts (accumulated across server_rounds)
    server_batch_step = {"step": 0}
    def evaluate(server_round, parameters, config):
        model = get_model(args.lora_r_server)
        # Only set trainable parameters
        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        params_dict = zip(trainable_params, parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        if server_round!=0:
            if args.algorithm == 'ILORA' and args.heterogeneous_rank == 'True':
                max_rank = max(args.heterogeneous_rank_clients)
                rank_idx = args.heterogeneous_rank_clients.index(max_rank)
                merged_model_path = os.path.join(args.save_dir, f"update_LORA_Decomposition_client{rank_idx}.pth")
                model.load_state_dict(torch.load(merged_model_path),strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
        # Evaluation
        # ====== Evaluate and write logs by batch ======
        model.eval()
        total_loss_sum = 0.0
        total_correct_sum = 0
        total_seen = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                batch_loss = nn.CrossEntropyLoss()(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels).sum().item()
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0

                # Accumulate for the whole round
                total_loss_sum += batch_loss
                total_correct_sum += batch_correct
                total_seen += batch_total

                # NEW: --Log by batch--
                step = server_batch_step["step"]
                #server_writer.add_scalar("server/batch_loss", batch_loss, step)
                #server_writer.add_scalar("server/batch_accuracy", batch_acc, step)
                server_writer.flush()
                server_batch_step["step"] += 1  # Increment the global batch step
        # Compute the aggregated metrics for this round (full-round evaluation) and log once
        round_loss = total_loss_sum  # Note that this is the sum of batch losses (consistent with your original implementation)
        round_acc = (total_correct_sum / total_seen) if total_seen > 0 else 0.0
        #server_writer.add_scalar("server/round_loss_sum", round_loss, server_round)
        #server_writer.add_scalar("server/round_accuracy", round_acc, server_round)
        #server_writer.flush()

        #print(f"[Server Eval] round={server_round}  acc={round_acc:.4f}  loss_sum={round_loss:.4f}")
        # Flower still returns full-round metrics
        return round_loss, {"accuracy": round_acc}

    return evaluate
def main():
    # Create the test-set data loader
    start_time = time.time()
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)

    if args.training_mode == 'Homo':
        test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)
        # Define the federated learning strategy
        strategy = FedLoRAStrategy(
            args.lora_r_server,
            use_control=args.use_control,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=0.0,# Set to 0 to disable client evaluation
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=0,# Set to 0 to disable client evaluation
            min_available_clients=args.min_evaluate_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=get_evaluate_fn(test_loader),
        )
        def client_fn(cid: str) -> Client:#Modification 2
            # Create a NumPyClient instance
            numpy_client = CIFAR10Client(int(cid), train_indices, test_indices)
            # Convert to a Client instance
            return fl.client.NumPyClient.to_client(numpy_client)
        # Start the simulation
        print("Starting Federated Learning with LoRA...")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 1}
        )
        # Output results
        duration = time.time() - start_time
        print(f"\nFederated Learning completed in {duration:.2f} seconds")

        if history.metrics_distributed and 'accuracy' in history.metrics_distributed:
            final_acc = history.metrics_distributed['accuracy'][-1][1]
            print(f"Final accuracy: {final_acc:.4f}")
        # Save results to a file
        save_results_to_file(args, history, duration)

def save_results_to_file(args, history, duration):#Save function dedicated to federated learning
    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    # Create the result filename (according to the specified format)
    result_filename = (
        f"{args.algorithm}_clients{args.num_clients}_"
        f"batch_size{args.batch_size}_epochs{args.local_epochs}_"
        f"num_rounds={args.num_rounds}_lr={args.lr}_"
        f"r_client={args.lora_r_client}_r_server={args.lora_r_server}_"
        f"optimizer={args.optimizer}_"
        f"distribution={args.distribution}_alpha={args.alpha}_"
        f"heterogeneous_rank={args.heterogeneous_rank}_heterogeneous_rank_clients={args.heterogeneous_rank_clients}_"
        f"{args.dataset}.txt"
    )
    result_path = os.path.join(args.save_dir, result_filename)
    # Prepare the content to save
    content = "Current runtime configuration:\n"
    content += "=" * 50 + "\n"
    for arg in vars(args):
        content += f"{arg}: {getattr(args, arg)}\n"
    content += "=" * 50 + "\n\n"

    # Add training results
    content += "Training results:\n"
    content += "=" * 50 + "\n"
    content += f"Federated learning completion time: {duration:.2f} seconds\n\n"

    # Add loss history
    content += "Loss history (centralized):\n"
    for i, loss in enumerate(history.losses_centralized):
        content += f"round {i}: {loss}\n"

    # Add accuracy history
    content += "\nAccuracy history (centralized):\n"
    if history.metrics_centralized and 'accuracy' in history.metrics_centralized:
        for round_num, acc in history.metrics_centralized['accuracy']:
            content += f"round {round_num}: {acc:.4f}\n"

    # Write to file
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\nResults saved to file: {result_path}")
if __name__ == "__main__":
    # Print all parameters
    print("\nCurrent runtime configuration:")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*50)
    main()
    for file_path in glob.glob(os.path.join(args.save_dir, "*.pth")):
        try:
            os.remove(file_path)
            print("Cleanup completed")
        except Exception as e:
            print(f"Failed to clean file: {file_path} - {str(e)}")
    # Delete all .npy files (newly added part)
    for file_path in glob.glob(os.path.join(args.save_dir, "*.npy")):
        try:
            os.remove(file_path)
            print("Cleanup completed")
        except Exception as e:
            print(f"Failed to clean npy file: {file_path} - {str(e)}")
