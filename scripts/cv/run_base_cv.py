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
parser = argparse.ArgumentParser(description='ViT-LoRA')
parser.add_argument('--cuda_device', type=str, default='2', help='Visible GPU device IDs.')
parser.add_argument('--training_mode', type=str, default='Homo', choices=['Centralized', 'Homo'], help='Training mode: Centralized or Homo.')
parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory.')
parser.add_argument('--lora_r_client', type=int, default=4, help='Client-side LoRA rank.')
parser.add_argument('--lora_r_server', type=int, default=4, help='Server-side LoRA rank.')
parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA scaling factor.')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate.')
parser.add_argument('--target_modules', type=list, default=['query', 'value'], help='Target modules for LoRA.')
parser.add_argument('--model_name', type=str, default='vit-base-patch16-224', help='Backbone model name.')
parser.add_argument('--num_labels', type=int, help='Number of output labels.')
parser.add_argument('--fraction_fit', type=float, default=1.0, help='Fraction of clients used for training in each round.')
parser.add_argument('--min_fit_clients', type=int, default=3, help='Minimum number of training clients per round.')
parser.add_argument('--min_evaluate_clients', type=int, default=3, help='Minimum number of evaluation clients per round.')
parser.add_argument('--num_rounds', type=int, default=2, help='Total number of training rounds.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
parser.add_argument('--local_epochs', type=int, default=1, help='Number of local training epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--num_clients', type=int, default=3, help='Number of clients.')
parser.add_argument('--algorithm', type=str, default='FEDIT', choices=['FEDIT', 'FLORA', 'ILORA', 'LoRA_FAIR', 'FFA-LORA'], help='Federated LoRA algorithm.')
parser.add_argument('--save_dir', type=str, default='/path/to/save_dir', help='Directory used to save outputs.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'MNIST', 'STL10', 'SVHN', 'tiny-imagenet-200'], help='Dataset name.')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'], help='Optimizer type.')
parser.add_argument('--distribution', type=str, default='NON-IID', choices=['IID', 'NON-IID'], help='Data distribution type: IID or NON-IID.')
parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha controlling Non-IID severity.')
parser.add_argument('--heterogeneous_rank', type=str, default='False', choices=['True', 'False'], help='Enable heterogeneous LoRA ranks.')
parser.add_argument('--heterogeneous_rank_clients', type=str, default='2,4,6', help='Per-client LoRA ranks.')
parser.add_argument('--use_control', type=str, default='False', choices=['True', 'False'], help='Enable control variates.')
parser.add_argument('--lambda_reg', type=float, default=0.01, help='Regularization coefficient for LoRA_FAIR.')
parser.add_argument('--lora_scale_factor', type=float, default=0.5, help='Scale factor used by QR initialization.')
from tensorboardX import SummaryWriter
parser.add_argument('--log_dir', type=str, default='./runs', help='TensorBoard log directory.')
from sam_optimizer_ours import SAM
args = parser.parse_args()
if args.heterogeneous_rank == 'True':
    args.heterogeneous_rank_clients = [int(r) for r in args.heterogeneous_rank_clients.split(',')]
else:
    args.heterogeneous_rank_clients = [args.lora_r_client] * args.num_clients
import glob
for file_path in glob.glob(os.path.join(args.save_dir, '*.pth')):
    try:
        os.remove(file_path)
        print(f'Cleaned old file: {file_path}')
    except Exception as e:
        print(f'Failed to remove file: {file_path} - {str(e)}')
for file_path in glob.glob(os.path.join(args.save_dir, '*.npy')):
    try:
        os.remove(file_path)
        print(f'Cleaned old NumPy file: {file_path}')
    except Exception as e:
        print(f'Failed to remove NumPy file: {file_path} - {str(e)}')
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

def split_data(dataset, num_clients=args.num_clients, distribution=args.distribution):
    if distribution == 'IID':
        indices = np.random.permutation(len(dataset))
        return np.array_split(indices, num_clients)
    elif distribution == 'NON-IID':
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100
        elif args.dataset == 'MNIST':
            num_classes = 10
        elif args.dataset == 'STL10':
            num_classes = 10
        elif args.dataset == 'SVHN':
            num_classes = 10
        elif args.dataset == 'tiny-imagenet-200':
            num_classes = 200
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
        alpha = args.alpha
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            if len(class_indices[c]) == 0:
                continue
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.maximum(proportions, 0.001)
            proportions = proportions / proportions.sum()
            proportions = (proportions * len(class_indices[c])).astype(int)
            proportions[-1] = len(class_indices[c]) - np.sum(proportions[:-1])
            np.random.shuffle(class_indices[c])
            start = 0
            for client_id in range(num_clients):
                end = start + proportions[client_id]
                client_indices[client_id].extend(class_indices[c][start:end])
                start = end
        return [np.array(indices) for indices in client_indices]
    else:
        raise ValueError(f'Unknown data distribution type: {distribution}')
train_indices = split_data(full_train_dataset, distribution=args.distribution)
test_indices = split_data(full_test_dataset, distribution=args.distribution)

def get_model(r):
    import transformers
    transformers.logging.set_verbosity_error()
    if args.model_name == 'vit-base-patch16-224':
        model = ViTForImageClassification.from_pretrained(args.model_name, num_labels=args.num_labels, ignore_mismatched_sizes=True)
        model.classifier = nn.Linear(model.classifier.in_features, args.num_labels)
    elif args.model_name == 'swin-base-patch4-window7-224':
        from transformers import SwinForImageClassification
        model = SwinForImageClassification.from_pretrained(args.model_name, num_labels=args.num_labels, ignore_mismatched_sizes=True)
        model.classifier = nn.Linear(model.classifier.in_features, args.num_labels)
    merged_model_path = os.path.join(args.save_dir, 'merged_model.pth')
    if os.path.exists(merged_model_path) and args.algorithm != 'ILORA':
        model.load_state_dict(torch.load(merged_model_path))
    if args.model_name == 'vit-base-patch16-224':
        lora_config = LoraConfig(r=r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=args.target_modules, bias='none', modules_to_save=['classifier'])
    elif args.model_name == 'swin-base-patch4-window7-224':
        lora_config = LoraConfig(r=r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=['attention.self.query', 'attention.self.value'], bias='none', modules_to_save=['classifier'])
    model = get_peft_model(model, lora_config)
    if args.algorithm == 'ILORA':
        from peft.tuners.lora import LoraLayer
        for (name, module) in model.named_modules():
            if isinstance(module, LoraLayer) and isinstance(module.base_layer, nn.Linear):
                W = module.base_layer.weight.data.clone()
                (d, k) = W.shape
                (Q, R) = torch.linalg.qr(W, mode='reduced')
                Q_r = Q[:, :r]
                R_r = R[:r, :]
                target_norm = 0.0001 * (d * k) ** 0.5
                current_norm = torch.norm(Q_r @ R_r)
                scale = target_norm / current_norm
                scale_factor = torch.sqrt(scale)
                R_r = R_r * scale_factor
                Q_r = Q_r * scale_factor
                delta_W = Q_r @ R_r * args.lora_scale_factor
                module.base_layer.weight.data = W - delta_W
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
                    delta_W = Q_r @ R_r
    return model.to(device)

class FedLoRAStrategy(fl.server.strategy.FedAvg):

    def __init__(self, lora_r_server, use_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = get_model(lora_r_server)
        self.use_control = use_control
        if use_control == 'True':
            trainable_params = [p.detach().cpu().numpy() for (n, p) in self.global_model.named_parameters() if p.requires_grad and ('lora' in n or 'classifier' in n)]
            self.c = [np.zeros_like(p) for p in trainable_params]
        else:
            self.c = None

    def initialize_parameters(self, client_manager):
        arrs = [p.detach().cpu().numpy() for (n, p) in self.global_model.named_parameters() if p.requires_grad]
        from flwr.common import ndarrays_to_parameters
        return ndarrays_to_parameters(arrs)

    def aggregate_fit(self, server_round, results, failures):
        if args.algorithm == 'FEDIT':
            total_examples = sum([metrics.num_examples for (_, metrics) in results])
            if args.heterogeneous_rank == 'True':
                max_rank = max(args.heterogeneous_rank_clients)
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    for (i, param) in enumerate(client_params):
                        if len(param.shape) == 2:
                            if param.shape[0] == client_rank and param.shape[0] < max_rank:
                                padding = ((0, max_rank - param.shape[0]), (0, 0))
                                padded_param = np.pad(param, padding, 'constant')
                                client_params[i] = padded_param
                            elif param.shape[1] == client_rank and param.shape[1] < max_rank:
                                padding = ((0, 0), (0, max_rank - param.shape[1]))
                                padded_param = np.pad(param, padding, 'constant')
                                client_params[i] = padded_param
                    if aggregated_params is None:
                        aggregated_params = []
                        for param in client_params:
                            weighted_param = p_k * param
                            aggregated_params.append(weighted_param)
                    else:
                        for (i, param) in enumerate(client_params):
                            aggregated_params[i] += p_k * param
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                if aggregated_parameters is not None:
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                    torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, 'merged_model.pth'))
                    aggregated_metrics = {}
                return (aggregated_parameters, aggregated_metrics)
            elif args.heterogeneous_rank == 'False':
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    if aggregated_params is None:
                        aggregated_params = []
                        for param in client_params:
                            weighted_param = p_k * param
                            aggregated_params.append(weighted_param)
                    else:
                        for i in range(len(aggregated_params)):
                            aggregated_params[i] += p_k * client_params[i]
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                if aggregated_parameters is not None:
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                    torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, 'merged_model.pth'))
                aggregated_metrics = {}
                return (aggregated_parameters, aggregated_metrics)
        if args.algorithm == 'FLORA':
            if args.heterogeneous_rank == 'True':
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    if aggregated_params is None:
                        aggregated_params = []
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:
                                    aggregated_params.append(p_k * param)
                                elif param.shape[1] == client_rank:
                                    aggregated_params.append(param)
                                else:
                                    aggregated_params.append(p_k * param)
                            else:
                                aggregated_params.append(p_k * param)
                    else:
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], p_k * param], axis=0)
                                elif param.shape[1] == client_rank:
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], param], axis=1)
                                elif param.shape[0] == args.num_labels:
                                    aggregated_params[i] += p_k * param
                            elif len(param.shape) == 1 and param.shape[0] == args.num_labels:
                                aggregated_params[i] += p_k * param
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                if aggregated_parameters is not None:
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                    torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, 'merged_model.pth'))
                return (aggregated_parameters, aggregated_metrics)
            elif args.heterogeneous_rank == 'False':
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    if aggregated_params is None:
                        aggregated_params = []
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == args.lora_r_client:
                                    aggregated_params.append(p_k * param)
                                elif param.shape[1] == args.lora_r_client:
                                    aggregated_params.append(param)
                                else:
                                    aggregated_params.append(p_k * param)
                            else:
                                aggregated_params.append(p_k * param)
                    else:
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == args.lora_r_client:
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], p_k * param], axis=0)
                                elif param.shape[1] == args.lora_r_client:
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], param], axis=1)
                                elif param.shape[0] == args.num_labels:
                                    aggregated_params[i] += p_k * param
                            elif len(param.shape) == 1 and param.shape[0] == args.num_labels:
                                aggregated_params[i] += p_k * param
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                if aggregated_parameters is not None:
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                    torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, 'merged_model.pth'))
                return (aggregated_parameters, aggregated_metrics)
        if args.algorithm == 'FFA-LORA':
            if args.heterogeneous_rank == 'True':
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                max_rank = max(args.heterogeneous_rank_clients)
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_id = fit_res.metrics['client_id']
                    p_k = client_weights[client_id]
                    client_rank = args.heterogeneous_rank_clients[int(client_id)]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    filled_params = []
                    for param in client_params:
                        if len(param.shape) == 2:
                            if param.shape[1] == client_rank:
                                padding = ((0, 0), (0, max_rank - client_rank))
                                filled_param = np.pad(param, padding, 'constant')
                                filled_params.append(filled_param)
                            elif param.shape[0] == client_rank:
                                padding = ((0, max_rank - client_rank), (0, 0))
                                filled_param = np.pad(param, padding, 'constant')
                                filled_params.append(filled_param)
                            else:
                                filled_params.append(param)
                        else:
                            filled_params.append(param)
                    if aggregated_params is None:
                        aggregated_params = [np.zeros_like(p) for p in filled_params]
                    for (i, param) in enumerate(filled_params):
                        if len(param.shape) == 2:
                            if param.shape[1] == max_rank:
                                aggregated_params[i] += p_k * param
                            elif param.shape[0] == max_rank:
                                if np.all(aggregated_params[i] == 0):
                                    aggregated_params[i] = param
                            else:
                                aggregated_params[i] += p_k * param
                        else:
                            aggregated_params[i] += p_k * param
                aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                if aggregated_parameters is not None:
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                    torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, 'merged_model.pth'))
                return (aggregated_parameters, aggregated_metrics)
            elif args.heterogeneous_rank == 'False':
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_id = fit_res.metrics['client_id']
                    p_k = client_weights[client_id]
                    client_rank = args.heterogeneous_rank_clients[int(client_id)]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    if aggregated_params is None:
                        aggregated_params = [np.zeros_like(param) for param in client_params]
                    for (i, param) in enumerate(client_params):
                        if len(param.shape) == 2:
                            if param.shape[1] == client_rank:
                                aggregated_params[i] += p_k * param
                            elif param.shape[0] == client_rank:
                                if np.all(aggregated_params[i] == 0):
                                    aggregated_params[i] = param
                            else:
                                aggregated_params[i] += p_k * param
                        else:
                            aggregated_params[i] += p_k * param
                aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                if aggregated_parameters is not None:
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                    torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                    import copy
                    model_copy = copy.deepcopy(self.global_model)
                    merged_model = model_copy.merge_and_unload()
                    torch.save(merged_model.state_dict(), os.path.join(args.save_dir, 'merged_model.pth'))
                return (aggregated_parameters, aggregated_metrics)
        if args.algorithm == 'LoRA_FAIR':
            if args.heterogeneous_rank == 'True':
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                max_rank = max(args.heterogeneous_rank_clients)
                first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
                num_layers = (len(first_client_params) - 2) // 4
                layer_dims = []
                for layer_idx in range(num_layers):
                    base_idx = layer_idx * 4
                    q_A = first_client_params[base_idx]
                    layer_dims.append(q_A.shape[1])
                avg_A_query = [np.zeros((max_rank, dim)) for dim in layer_dims]
                avg_B_query = [np.zeros((dim, max_rank)) for dim in layer_dims]
                avg_A_value = [np.zeros((max_rank, dim)) for dim in layer_dims]
                avg_B_value = [np.zeros((dim, max_rank)) for dim in layer_dims]
                global_delta_w_query = [np.zeros((dim, dim)) for dim in layer_dims]
                global_delta_w_value = [np.zeros((dim, dim)) for dim in layer_dims]
                for (_, fit_res) in results:
                    client_id = fit_res.metrics['client_id']
                    p_k = client_weights[client_id]
                    client_rank = args.heterogeneous_rank_clients[int(client_id)]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    for layer_idx in range(num_layers):
                        base_idx = layer_idx * 4
                        dim = layer_dims[layer_idx]
                        q_A = client_params[base_idx]
                        q_B = client_params[base_idx + 1]
                        v_A = client_params[base_idx + 2]
                        v_B = client_params[base_idx + 3]
                        q_A_padded = np.zeros((max_rank, dim))
                        q_A_padded[:client_rank, :] = q_A
                        q_B_padded = np.zeros((dim, max_rank))
                        q_B_padded[:, :client_rank] = q_B
                        v_A_padded = np.zeros((max_rank, dim))
                        v_A_padded[:client_rank, :] = v_A
                        v_B_padded = np.zeros((dim, max_rank))
                        v_B_padded[:, :client_rank] = v_B
                        avg_A_query[layer_idx] += p_k * q_A_padded
                        avg_B_query[layer_idx] += p_k * q_B_padded
                        avg_A_value[layer_idx] += p_k * v_A_padded
                        avg_B_value[layer_idx] += p_k * v_B_padded
                        delta_w_query = q_B_padded @ q_A_padded
                        delta_w_value = v_B_padded @ v_A_padded
                        global_delta_w_query[layer_idx] += p_k * delta_w_query
                        global_delta_w_value[layer_idx] += p_k * delta_w_value
                lambda_reg = args.lambda_reg
                updated_B_query = []
                updated_B_value = []
                for layer_idx in range(num_layers):
                    R_query = global_delta_w_query[layer_idx] - avg_B_query[layer_idx] @ avg_A_query[layer_idx]
                    I = np.eye(max_rank)
                    A = avg_A_query[layer_idx]
                    inv_part = np.linalg.inv(A @ A.T + lambda_reg * I)
                    delta_B_query = R_query @ A.T @ inv_part
                    R_value = global_delta_w_value[layer_idx] - avg_B_value[layer_idx] @ avg_A_value[layer_idx]
                    A = avg_A_value[layer_idx]
                    inv_part = np.linalg.inv(A @ A.T + lambda_reg * I)
                    delta_B_value = R_value @ A.T @ inv_part
                    updated_B_query.append(avg_B_query[layer_idx] + delta_B_query)
                    updated_B_value.append(avg_B_value[layer_idx] + delta_B_value)
                aggregated_params = []
                for layer_idx in range(num_layers):
                    aggregated_params.append(avg_A_query[layer_idx])
                    aggregated_params.append(updated_B_query[layer_idx])
                    aggregated_params.append(avg_A_value[layer_idx])
                    aggregated_params.append(updated_B_value[layer_idx])
                classifier_weight = np.zeros_like(first_client_params[-2])
                classifier_bias = np.zeros_like(first_client_params[-1])
                for (_, fit_res) in results:
                    p_k = client_weights[fit_res.metrics['client_id']]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    classifier_weight += p_k * client_params[-2]
                    classifier_bias += p_k * client_params[-1]
                aggregated_params.append(classifier_weight)
                aggregated_params.append(classifier_bias)
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                params_dict = zip([name for (name, param) in self.global_model.named_parameters() if param.requires_grad], aggregated_params)
                state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                self.global_model.load_state_dict(state_dict, strict=False)
                trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                import copy
                model_copy = copy.deepcopy(self.global_model)
                merged_model = model_copy.merge_and_unload()
                torch.save(merged_model.state_dict(), os.path.join(args.save_dir, 'merged_model.pth'))
                aggregated_metrics = {}
                return (aggregated_parameters, aggregated_metrics)
            if args.heterogeneous_rank == 'False':
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                for (client_idx, (_, fit_res)) in enumerate(results):
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
                num_layers = (len(first_client_params) - 2) // 4
                layer_dims = []
                for layer_idx in range(num_layers):
                    base_idx = layer_idx * 4
                    q_A = first_client_params[base_idx]
                    layer_dims.append(q_A.shape[1])
                avg_A_query = [np.zeros((args.lora_r_client, dim)) for dim in layer_dims]
                avg_B_query = [np.zeros((dim, args.lora_r_client)) for dim in layer_dims]
                avg_A_value = [np.zeros((args.lora_r_client, dim)) for dim in layer_dims]
                avg_B_value = [np.zeros((dim, args.lora_r_client)) for dim in layer_dims]
                global_delta_w_query = [np.zeros((dim, dim)) for dim in layer_dims]
                global_delta_w_value = [np.zeros((dim, dim)) for dim in layer_dims]
                for (client_idx, (_, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    for layer_idx in range(num_layers):
                        base_idx = layer_idx * 4
                        q_A = client_params[base_idx]
                        q_B = client_params[base_idx + 1]
                        v_A = client_params[base_idx + 2]
                        v_B = client_params[base_idx + 3]
                        avg_A_query[layer_idx] += p_k * q_A
                        avg_B_query[layer_idx] += p_k * q_B
                        avg_A_value[layer_idx] += p_k * v_A
                        avg_B_value[layer_idx] += p_k * v_B
                        delta_w_query = q_B @ q_A
                        delta_w_value = v_B @ v_A
                        global_delta_w_query[layer_idx] += p_k * delta_w_query
                        global_delta_w_value[layer_idx] += p_k * delta_w_value
                lambda_reg = args.lambda_reg
                updated_B_query = []
                updated_B_value = []
                for layer_idx in range(num_layers):
                    R_query = global_delta_w_query[layer_idx] - avg_B_query[layer_idx] @ avg_A_query[layer_idx]
                    I = np.eye(args.lora_r_client)
                    A = avg_A_query[layer_idx]
                    inv_part = np.linalg.inv(A @ A.T + lambda_reg * I)
                    delta_B_query = R_query @ A.T @ inv_part
                    R_value = global_delta_w_value[layer_idx] - avg_B_value[layer_idx] @ avg_A_value[layer_idx]
                    A = avg_A_value[layer_idx]
                    inv_part = np.linalg.inv(A @ A.T + lambda_reg * I)
                    delta_B_value = R_value @ A.T @ inv_part
                    updated_B_query.append(avg_B_query[layer_idx] + delta_B_query)
                    updated_B_value.append(avg_B_value[layer_idx] + delta_B_value)
                aggregated_params = []
                for layer_idx in range(num_layers):
                    aggregated_params.append(avg_A_query[layer_idx])
                    aggregated_params.append(updated_B_query[layer_idx])
                    aggregated_params.append(avg_A_value[layer_idx])
                    aggregated_params.append(updated_B_value[layer_idx])
                classifier_weight = np.zeros_like(client_params[-2])
                classifier_bias = np.zeros_like(client_params[-1])
                total_weight = 0.0
                for (client_idx, (_, fit_res)) in enumerate(results):
                    p_k = client_weights[fit_res.metrics['client_id']]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    classifier_weight += p_k * client_params[-2]
                    classifier_bias += p_k * client_params[-1]
                aggregated_params.append(classifier_weight)
                aggregated_params.append(classifier_bias)
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                params_dict = zip([name for (name, param) in self.global_model.named_parameters() if param.requires_grad], aggregated_params)
                state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                self.global_model.load_state_dict(state_dict, strict=False)
                trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                import copy
                model_copy = copy.deepcopy(self.global_model)
                merged_model = model_copy.merge_and_unload()
                torch.save(merged_model.state_dict(), os.path.join(args.save_dir, 'merged_model.pth'))
                aggregated_metrics = {}
                return (aggregated_parameters, aggregated_metrics)
        if args.algorithm == 'ILORA':
            if args.heterogeneous_rank == 'False':
                c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                try:
                    self.c = np.load(c_save_path, allow_pickle=True).tolist()
                except:
                    print(f'c')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                if args.use_control == 'True':
                    delta_c_aggregated = [np.zeros_like(p) for p in self.c]
                    for (client_idx, (_, fit_res)) in enumerate(results):
                        client_idx = fit_res.metrics['client_id']
                        delta_ci_path = os.path.join(args.save_dir, f'client_{client_idx}_delta_ci.npy')
                        try:
                            delta_ci = np.load(delta_ci_path, allow_pickle=True).tolist()
                            for i in range(len(delta_c_aggregated)):
                                delta_c_aggregated[i] += 1 / args.num_clients * delta_ci[i]
                        except Exception as e:
                            print(f'Failed to load client {client_idx} delta_ci file: {str(e)}')
                            continue
                    for i in range(len(self.c)):
                        self.c[i] += delta_c_aggregated[i]
                    if args.use_control == 'True':
                        c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                        np.save(c_save_path, np.array(self.c, dtype=object))
                        print(f'Saved global control variate c to {c_save_path}')
                aggregated_params = None
                for (client_idx, (_, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    if aggregated_params is None:
                        aggregated_params = []
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == args.lora_r_client:
                                    aggregated_params.append(p_k * param)
                                elif param.shape[1] == args.lora_r_client:
                                    aggregated_params.append(param)
                                else:
                                    aggregated_params.append(p_k * param)
                            else:
                                aggregated_params.append(p_k * param)
                    else:
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == args.lora_r_client:
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], p_k * param], axis=0)
                                elif param.shape[1] == args.lora_r_client:
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], param], axis=1)
                                elif param.shape[0] == args.num_labels:
                                    aggregated_params[i] += p_k * param
                            elif len(param.shape) == 1 and param.shape[0] == args.num_labels:
                                aggregated_params[i] += p_k * param
                num_layers = (len(aggregated_params) - 2) // 4
                for layer_idx in range(num_layers):
                    base_idx = layer_idx * 4
                    q_A = aggregated_params[base_idx]
                    q_B = aggregated_params[base_idx + 1]
                    Q_full = q_B @ q_A
                    (Q_q, R_q) = np.linalg.qr(Q_full, mode='reduced')
                    q_A_prime = R_q[:args.lora_r_client, :]
                    q_B_prime = Q_q[:, :args.lora_r_client]
                    aggregated_params[base_idx] = q_A_prime
                    aggregated_params[base_idx + 1] = q_B_prime
                    v_A = aggregated_params[base_idx + 2]
                    v_B = aggregated_params[base_idx + 3]
                    V_full = v_B @ v_A
                    (Q_v, R_v) = np.linalg.qr(V_full, mode='reduced')
                    v_A_prime = R_v[:args.lora_r_client, :]
                    v_B_prime = Q_v[:, :args.lora_r_client]
                    aggregated_params[base_idx + 2] = v_A_prime
                    aggregated_params[base_idx + 3] = v_B_prime
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                if aggregated_parameters is not None:
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(aggregated_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    trainable_state = {k: v for (k, v) in self.global_model.state_dict().items() if any((n in k for n in ['lora', 'classifier']))}
                    torch.save(trainable_state, os.path.join(args.save_dir, 'update_LORA.pth'))
                aggregated_metrics = {}
                return (aggregated_parameters, aggregated_metrics)
            if args.heterogeneous_rank == 'True':
                if args.use_control == 'True':
                    c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                    try:
                        self.c = np.load(c_save_path, allow_pickle=True).tolist()
                    except:
                        print(f'c')
                max_rank = max(args.heterogeneous_rank_clients)
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                if args.use_control == 'True':
                    delta_c_aggregated = [np.zeros_like(p) for p in self.c]
                    for (client_idx, (_, fit_res)) in enumerate(results):
                        client_idx = fit_res.metrics['client_id']
                        delta_ci_path = os.path.join(args.save_dir, f'client_{client_idx}_delta_ci.npy')
                        try:
                            delta_ci = np.load(delta_ci_path, allow_pickle=True).tolist()
                            client_rank = args.heterogeneous_rank_clients[client_idx]
                            for (i, ci_val) in enumerate(delta_ci):
                                if len(ci_val.shape) == 2:
                                    if ci_val.shape[0] == client_rank and ci_val.shape[0] < max_rank:
                                        padding = ((0, max_rank - ci_val.shape[0]), (0, 0))
                                        padded_param = np.pad(ci_val, padding, 'constant')
                                        delta_ci[i] = padded_param
                                    elif ci_val.shape[1] == client_rank and ci_val.shape[1] < max_rank:
                                        padding = ((0, 0), (0, max_rank - ci_val.shape[1]))
                                        padded_param = np.pad(ci_val, padding, 'constant')
                                        delta_ci[i] = padded_param
                            for i in range(len(delta_c_aggregated)):
                                delta_c_aggregated[i] += 1 / args.num_clients * delta_ci[i]
                        except Exception as e:
                            print(f'Failed to load client {client_idx} delta_ci file: {str(e)}')
                            continue
                    for i in range(len(self.c)):
                        self.c[i] += delta_c_aggregated[i]
                    if args.use_control == 'True':
                        c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                        np.save(c_save_path, np.array(self.c, dtype=object))
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    if aggregated_params is None:
                        aggregated_params = []
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:
                                    aggregated_params.append(p_k * param)
                                elif param.shape[1] == client_rank:
                                    aggregated_params.append(param)
                                else:
                                    aggregated_params.append(p_k * param)
                            else:
                                aggregated_params.append(p_k * param)
                    else:
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], p_k * param], axis=0)
                                elif param.shape[1] == client_rank:
                                    aggregated_params[i] = np.concatenate([aggregated_params[i], param], axis=1)
                                elif param.shape[0] == args.num_labels:
                                    aggregated_params[i] += p_k * param
                            elif len(param.shape) == 1 and param.shape[0] == args.num_labels:
                                aggregated_params[i] += p_k * param
                num_layers = (len(aggregated_params) - 2) // 4
                Decomposition_save_paths = [os.path.join(args.save_dir, f'update_LORA_Decomposition_client{i}.pth') for i in range(len(args.heterogeneous_rank_clients))]
                for (rank_idx, target_rank) in enumerate(args.heterogeneous_rank_clients):
                    rank_specific_params = [param.copy() for param in aggregated_params]
                    for layer_idx in range(num_layers):
                        base_idx = layer_idx * 4
                        q_A = rank_specific_params[base_idx]
                        q_B = rank_specific_params[base_idx + 1]
                        Q_full = q_B @ q_A
                        (Q_q, R_q) = np.linalg.qr(Q_full, mode='reduced')
                        q_A_prime = R_q[:target_rank, :]
                        q_B_prime = Q_q[:, :target_rank]
                        rank_specific_params[base_idx] = q_A_prime
                        rank_specific_params[base_idx + 1] = q_B_prime
                        v_A = rank_specific_params[base_idx + 2]
                        v_B = rank_specific_params[base_idx + 3]
                        V_full = v_B @ v_A
                        (Q_v, R_v) = np.linalg.qr(V_full, mode='reduced')
                        v_A_prime = R_v[:target_rank, :]
                        v_B_prime = Q_v[:, :target_rank]
                        rank_specific_params[base_idx + 2] = v_A_prime
                        rank_specific_params[base_idx + 3] = v_B_prime
                    if target_rank == max_rank:
                        max_rank_params = rank_specific_params
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], rank_specific_params)
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    trainable_state = {k: v for (k, v) in state_dict.items() if any((n in k for n in ['lora', 'classifier']))}
                    torch.save(trainable_state, Decomposition_save_paths[rank_idx])
                if server_round == args.num_rounds - 1 and max_rank_params is not None:
                    from flwr.common import ndarrays_to_parameters
                    max_rank_parameters = ndarrays_to_parameters(max_rank_params)
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(max_rank_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    final_model_path = os.path.join(args.save_dir, 'final_global_model.pth')
                    torch.save(self.global_model.state_dict(), final_model_path)
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                return (aggregated_parameters, aggregated_metrics)
from flwr.client import Client, NumPyClient

class CIFAR10Client(NumPyClient):

    def __init__(self, cid, train_indices, test_indices):
        self.cid = cid
        client_rank = args.heterogeneous_rank_clients[int(cid)]
        self.model = get_model(client_rank)
        if args.use_control == 'True':
            trainable_params = [p.detach().cpu().numpy() for p in self.model.parameters() if p.requires_grad]
            self.ci = [torch.zeros_like(p).cpu().numpy() for p in self.model.parameters() if p.requires_grad]
        self.train_loader = DataLoader(Subset(full_train_dataset, train_indices[cid]), batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(Subset(full_test_dataset, test_indices[cid]), batch_size=args.batch_size)

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for (name, val) in self.model.named_parameters() if val.requires_grad]

    def set_parameters(self, parameters):
        params_dict = zip([name for (name, param) in self.model.named_parameters() if param.requires_grad], parameters)
        state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
        if args.algorithm == 'FFA-LORA' and args.heterogeneous_rank == 'False':
            self.model.load_state_dict(state_dict, strict=False)
        if args.algorithm == 'FFA-LORA' and args.heterogeneous_rank == 'True':
            params_dict = zip([name for (name, param) in self.model.named_parameters() if param.requires_grad], parameters)
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            max_rank = max(args.heterogeneous_rank_clients)
            truncated_state_dict = {}
            for ((name, _), param) in zip(params_dict, parameters):
                if len(param.shape) == 2:
                    if param.shape[0] == max_rank and 'lora_A' in name:
                        if param.shape[0] > client_rank:
                            truncated_param = param[:client_rank, :]
                        else:
                            truncated_param = param
                        truncated_state_dict[name] = torch.tensor(truncated_param)
                    elif param.shape[1] == max_rank and 'lora_B' in name:
                        if param.shape[1] > client_rank:
                            truncated_param = param[:, :client_rank]
                        else:
                            truncated_param = param
                        truncated_state_dict[name] = torch.tensor(truncated_param)
            self.model.load_state_dict(truncated_state_dict, strict=False)
        if args.algorithm == 'LoRA_FAIR' and args.heterogeneous_rank == 'False':
            self.model.load_state_dict(state_dict, strict=False)
        if args.algorithm == 'LoRA_FAIR' and args.heterogeneous_rank == 'True':
            params_dict = zip([name for (name, param) in self.model.named_parameters() if param.requires_grad], parameters)
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            max_rank = max(args.heterogeneous_rank_clients)
            truncated_state_dict = {}
            for ((name, _), param) in zip(params_dict, parameters):
                if len(param.shape) == 2:
                    if param.shape[0] == max_rank:
                        if param.shape[0] > client_rank:
                            truncated_param = param[:client_rank, :]
                        else:
                            truncated_param = param
                        truncated_state_dict[name] = torch.tensor(truncated_param)
                    elif param.shape[1] == max_rank:
                        if param.shape[1] > client_rank:
                            truncated_param = param[:, :client_rank]
                        else:
                            truncated_param = param
                        truncated_state_dict[name] = torch.tensor(truncated_param)
            self.model.load_state_dict(truncated_state_dict, strict=False)
        if args.algorithm == 'ILORA':
            if args.heterogeneous_rank == 'False':
                self.model.load_state_dict(state_dict, strict=False)
            elif args.heterogeneous_rank == 'True':
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                param_path = os.path.join(args.save_dir, f'update_LORA_Decomposition_client{self.cid}.pth')
                if os.path.exists(param_path):
                    trainable_state = torch.load(param_path)
                    self.model.load_state_dict(trainable_state, strict=False)
        if args.algorithm == 'FEDIT':
            params_dict = zip([name for (name, param) in self.model.named_parameters() if param.requires_grad], parameters)
            if args.heterogeneous_rank == 'False':
                self.model.load_state_dict(state_dict, strict=False)
            elif args.heterogeneous_rank == 'True':
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                max_rank = max(args.heterogeneous_rank_clients)
                print(f' {self.cid} FEDIT')
                print(f': {client_rank}, : {max_rank}')
                print(f':')
                truncated_state_dict = {}
                for (i, ((name, _), param)) in enumerate(zip(params_dict, parameters)):
                    if len(param.shape) == 2:
                        is_lora_A = param.shape[0] == max_rank and 'lora_A' in name
                        is_lora_B = param.shape[1] == max_rank and 'lora_B' in name
                        if is_lora_A:
                            truncated_param = param[:client_rank, :] if param.shape[0] > client_rank else param
                            truncated_state_dict[name] = torch.tensor(truncated_param)
                        elif is_lora_B:
                            truncated_param = param[:, :client_rank] if param.shape[1] > client_rank else param
                            truncated_state_dict[name] = torch.tensor(truncated_param)
                self.model.load_state_dict(truncated_state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if args.algorithm == 'ILORA':
            print('ILORA: reinitialization is not required.')
        elif args.algorithm == 'LoRA_FAIR':
            print('LoRA_FAIR: reinitialization is not required.')
        elif args.algorithm == 'FFA-LORA':
            print('FFA-LORA: reinitialization is not required.')
        elif args.algorithm == 'FEDIT':
            print('FEDIT: reinitialization is not required.')
        else:
            print('Reinitialization is required.')
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            self.model = get_model(client_rank)
        initial_params = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
        if args.use_control == 'True':
            ci_path = os.path.join(args.save_dir, f'client_{self.cid}_ci.npy')
            if os.path.exists(ci_path):
                self.ci = np.load(ci_path, allow_pickle=True).tolist()
            c_save_path = os.path.join(args.save_dir, 'global_c.npy')
            try:
                c = np.load(c_save_path, allow_pickle=True).tolist()
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                max_rank = max(args.heterogeneous_rank_clients)
                truncated_c = []
                for (param_idx, param) in enumerate(c):
                    if len(param.shape) == 2:
                        if param.shape[0] == max_rank and param.shape[0] > client_rank:
                            truncated_param = param[:client_rank, :]
                        elif param.shape[1] == max_rank and param.shape[1] > client_rank:
                            truncated_param = param[:, :client_rank]
                        else:
                            truncated_param = param
                        truncated_c.append(truncated_param)
                    else:
                        truncated_c.append(param)
                c = truncated_c
            except:
                c = self.ci
            if len(c) != len(self.ci):
                c = self.ci
        import copy
        if args.use_control == 'True':
            original_ci = copy.deepcopy(self.ci)
            control_option = 1
            if control_option == 1:
                original_mode = self.model.training
                self.model.eval()
                self.model.zero_grad()
                total_samples = 0
                for (inputs, labels) in self.train_loader:
                    (inputs, labels) = (inputs.to(device), labels.to(device))
                    outputs = self.model(inputs).logits
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    loss.backward()
                    total_samples += labels.size(0)
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad /= total_samples
                new_ci = [param.grad.clone().detach().cpu().numpy() for param in self.model.parameters() if param.requires_grad]
                self.model.train(original_mode)
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.get('lr', args.lr), momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.get('lr', args.lr), weight_decay=args.weight_decay)
        self.model.train()
        K = 0
        for epoch in range(config.get('epochs', args.local_epochs)):
            running_loss = 0.0
            batch_count = 0
            for (inputs, labels) in self.train_loader:
                K += 1
                (inputs, labels) = (inputs.to(device), labels.to(device))
                optimizer.zero_grad()
                outputs = self.model(inputs).logits
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                if args.use_control == 'True':
                    with torch.no_grad():
                        for (param, ci_val, c_val) in zip([p for p in self.model.parameters() if p.requires_grad], self.ci, c):
                            param.grad.add_(torch.tensor(c_val - ci_val).to(device))
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
            self.model.eval()
            (correct, total) = (0, 0)
            with torch.no_grad():
                for (inputs, labels) in self.test_loader:
                    (inputs, labels) = (inputs.to(device), labels.to(device))
                    outputs = self.model(inputs).logits
                    (_, predicted) = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            local_acc = correct / total if total > 0 else 0.0
            self.model.train()
            if args.use_control == 'True':
                self.ci = new_ci
                ci_save_path = os.path.join(args.save_dir, f'client_{self.cid}_ci.npy')
                np.save(ci_save_path, np.array(self.ci, dtype=object))
                delta_ci = [new_ci - old_ci for (new_ci, old_ci) in zip(self.ci, original_ci)]
                delta_ci_path = os.path.join(args.save_dir, f'client_{self.cid}_delta_ci.npy')
                np.save(delta_ci_path, np.array(delta_ci, dtype=object))
        return (self.get_parameters({}), len(self.train_loader.dataset), {'client_id': int(self.cid)})

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        (loss, accuracy) = (0.0, 0.0)
        total = 0
        self.model.eval()
        with torch.no_grad():
            for (inputs, labels) in self.test_loader:
                (inputs, labels) = (inputs.to(device), labels.to(device))
                outputs = self.model(inputs).logits
                loss += nn.CrossEntropyLoss()(outputs, labels).item()
                (_, predicted) = torch.max(outputs, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        accuracy /= total
        return (loss, total, {'accuracy': accuracy})

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['accuracy'] for (num_examples, m) in metrics]
    examples = [num_examples for (num_examples, _) in metrics]
    return {'accuracy': sum(accuracies) / sum(examples)}

def get_evaluate_fn(test_loader):
    os.makedirs(os.path.join(args.log_dir, 'server'), exist_ok=True)
    server_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'server'))
    server_batch_step = {'step': 0}

    def evaluate(server_round, parameters, config):
        model = get_model(args.lora_r_server)
        trainable_params = [name for (name, param) in model.named_parameters() if param.requires_grad]
        params_dict = zip(trainable_params, parameters)
        state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
        if server_round != 0:
            if args.algorithm == 'ILORA' and args.heterogeneous_rank == 'True':
                max_rank = max(args.heterogeneous_rank_clients)
                rank_idx = args.heterogeneous_rank_clients.index(max_rank)
                merged_model_path = os.path.join(args.save_dir, f'update_LORA_Decomposition_client{rank_idx}.pth')
                model.load_state_dict(torch.load(merged_model_path), strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
        model.eval()
        total_loss_sum = 0.0
        total_correct_sum = 0
        total_seen = 0
        with torch.no_grad():
            for (batch_idx, (inputs, labels)) in enumerate(test_loader):
                (inputs, labels) = (inputs.to(device), labels.to(device))
                outputs = model(inputs).logits
                batch_loss = nn.CrossEntropyLoss()(outputs, labels).item()
                (_, predicted) = torch.max(outputs, 1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels).sum().item()
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
                total_loss_sum += batch_loss
                total_correct_sum += batch_correct
                total_seen += batch_total
                step = server_batch_step['step']
                server_writer.flush()
                server_batch_step['step'] += 1
        round_loss = total_loss_sum
        round_acc = total_correct_sum / total_seen if total_seen > 0 else 0.0
        return (round_loss, {'accuracy': round_acc})
    return evaluate

def main():
    start_time = time.time()
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)
    if args.training_mode == 'Centralized':
        model = get_model(args.lora_r_client)
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        best_accuracy = 0.0
        train_loss_history = []
        test_accuracy_history = []
        for epoch in range(args.num_rounds):
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
            print(f'Epoch {epoch + 1}/{args.num_rounds} | Train Loss: {avg_train_loss:.4f} | Test Accuracy: {accuracy:.2f}% | ')
        duration = time.time() - start_time
        print(f'\nCentralized Learning completed in {duration:.2f} seconds')
        print(f'Best test accuracy: {best_accuracy:.2f}%')
        save_centralized_results(args, best_accuracy, duration, train_loss_history, test_accuracy_history)
    elif args.training_mode == 'Homo':
        test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)
        strategy = FedLoRAStrategy(args.lora_r_server, use_control=args.use_control, fraction_fit=args.fraction_fit, fraction_evaluate=0.0, min_fit_clients=args.min_fit_clients, min_evaluate_clients=0, min_available_clients=args.min_evaluate_clients, evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(test_loader))

        def client_fn(cid: str) -> Client:
            numpy_client = CIFAR10Client(int(cid), train_indices, test_indices)
            return fl.client.NumPyClient.to_client(numpy_client)
        print('Starting Federated Learning with LoRA...')
        history = fl.simulation.start_simulation(client_fn=client_fn, num_clients=args.num_clients, config=fl.server.ServerConfig(num_rounds=args.num_rounds), strategy=strategy, client_resources={'num_cpus': 1, 'num_gpus': 1})
        duration = time.time() - start_time
        print(f'\nFederated Learning completed in {duration:.2f} seconds')
        if history.metrics_distributed and 'accuracy' in history.metrics_distributed:
            final_acc = history.metrics_distributed['accuracy'][-1][1]
            print(f'Final accuracy: {final_acc:.4f}')
        save_results_to_file(args, history, duration)

def save_results_to_file(args, history, duration):
    os.makedirs(args.save_dir, exist_ok=True)
    result_filename = f'{args.algorithm}_clients{args.num_clients}_batch_size{args.batch_size}_epochs{args.local_epochs}_num_rounds={args.num_rounds}_lr={args.lr}_r_client={args.lora_r_client}_r_server={args.lora_r_server}_optimizer={args.optimizer}_distribution={args.distribution}_alpha={args.alpha}_heterogeneous_rank={args.heterogeneous_rank}_heterogeneous_rank_clients={args.heterogeneous_rank_clients}_{args.dataset}.txt'
    result_path = os.path.join(args.save_dir, result_filename)
    content = ':\n'
    content += '=' * 50 + '\n'
    for arg in vars(args):
        content += f'{arg}: {getattr(args, arg)}\n'
    content += '=' * 50 + '\n\n'
    content += ':\n'
    content += '=' * 50 + '\n'
    content += f': {duration:.2f} \n\n'
    content += ' (centralized):\n'
    for (i, loss) in enumerate(history.losses_centralized):
        content += f'round {i}: {loss}\n'
    content += '\n (centralized):\n'
    if history.metrics_centralized and 'accuracy' in history.metrics_centralized:
        for (round_num, acc) in history.metrics_centralized['accuracy']:
            content += f'round {round_num}: {acc:.4f}\n'
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'\n: {result_path}')

def save_centralized_results(args, best_accuracy, duration, train_loss_history, test_accuracy_history):
    os.makedirs(args.save_dir, exist_ok=True)
    result_filename = f'Centralized_batch_size{args.batch_size}_epochs{args.num_rounds}_lr={args.lr}_{args.model_name}_r={args.lora_r_client}_{args.dataset}.txt'
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
        content += f'Epoch {epoch + 1}/{args.num_rounds} | Train Loss: {train_loss_history[epoch]:.4f} | Test Accuracy: {test_accuracy_history[epoch]:.2f}%\n'
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
            print('')
        except Exception as e:
            print(f'Failed to remove file: {file_path} - {str(e)}')
    for file_path in glob.glob(os.path.join(args.save_dir, '*.npy')):
        try:
            os.remove(file_path)
            print('')
        except Exception as e:
            print(f'Failed to remove NumPy file: {file_path} - {str(e)}')
