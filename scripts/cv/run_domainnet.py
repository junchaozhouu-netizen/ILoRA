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
import os
import random
import glob
parser = argparse.ArgumentParser(description='ViT-LoRA')
parser.add_argument('--cuda_device', type=str, default='2', help='Visible GPU device IDs.')
parser.add_argument('--training_mode', type=str, default='Homo', choices=['Centralized', 'Homo'], help='Training mode: Centralized or Homo.')
parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory.')
parser.add_argument('--lora_r_client', type=int, default=4, help='Client-side LoRA rank.')
parser.add_argument('--lora_r_server', type=int, default=6, help='Server-side LoRA rank.')
parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA scaling factor.')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate.')
parser.add_argument('--target_modules', type=list, default=['query', 'value'], help='Target modules for LoRA.')
parser.add_argument('--model_name', type=str, default='vit-base-patch16-224', help='Backbone model name.')
parser.add_argument('--num_labels', type=int, help='Number of output labels.')
parser.add_argument('--fraction_fit', type=float, default=1.0, help='Fraction of clients used for training in each round.')
parser.add_argument('--min_fit_clients', type=int, default=3, help='Minimum number of training clients per round.')
parser.add_argument('--min_evaluate_clients', type=int, default=3, help='Minimum number of evaluation clients per round.')
parser.add_argument('--num_rounds', type=int, default=2, help='Total number of training rounds.')
parser.add_argument('--scheduler', type=str, default='CosineAnnealing', choices=['None', 'StepLR', 'CosineAnnealing', 'Exponential', 'ReduceLROnPlateau'], help='Learning-rate scheduler type.')
parser.add_argument('--step_size', type=int, default=30, help='Step size for StepLR.')
parser.add_argument('--gamma', type=float, default=0.1, help='Decay factor for the scheduler.')
parser.add_argument('--t_max', type=int, default=50, help='T_max value for CosineAnnealingLR.')
parser.add_argument('--min_lr', type=float, default=0.0001, help='Minimum learning rate.')
parser.add_argument('--patience', type=int, default=10, help='Patience for ReduceLROnPlateau.')
parser.add_argument('--data_subset_ratio', type=float, default=0.8, help='Fraction of the dataset to keep.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
parser.add_argument('--local_epochs', type=int, default=1, help='Number of local training epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--num_clients', type=int, default=3, help='Number of clients.')
parser.add_argument('--algorithm', type=str, default='LoRA_FAIR', choices=['FEDIT', 'FLORA', 'ILORA', 'LoRA_FAIR', 'FFA-LORA'], help='Federated LoRA algorithm.')
parser.add_argument('--save_dir', type=str, default='/path/to/save_dir', help='Directory used to save outputs.')
parser.add_argument('--dataset', type=str, default='domainnet_quickdraw', choices=['cifar10', 'cifar100', 'MNIST', 'STL10', 'SVHN', 'tiny-imagenet-200', 'domainnet_clipart', 'domainnet_infograph', 'domainnet_painting', 'domainnet_quickdraw', 'domainnet_real', 'domainnet_sketch'], help='Dataset name.')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'], help='Optimizer type.')
parser.add_argument('--distribution', type=str, default='NON-IID', choices=['IID', 'NON-IID'], help='Data distribution type: IID or NON-IID.')
parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha controlling Non-IID severity.')
parser.add_argument('--heterogeneous_rank', type=str, default='True', choices=['True', 'False'], help='Enable heterogeneous LoRA ranks.')
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
if args.dataset.startswith('domainnet'):
    from PIL import Image
    if args.model_name == 'vit-base-patch16-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    elif args.model_name == 'swin-base-patch4-window7-224':
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    class DomainNetDataset(torch.utils.data.Dataset):

        def __init__(self, data_dir, domain, split='train', transform=None, num_classes=100):
            self.data_dir = data_dir
            self.domain = domain
            self.split = split
            self.transform = transform
            self.num_classes = num_classes
            txt_file = os.path.join(data_dir, f'{domain}_{split}.txt')
            self.samples = []
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            img_path = parts[0]
                            label = int(parts[1])
                            if label < num_classes:
                                full_img_path = os.path.join(data_dir, img_path)
                                self.samples.append((full_img_path, label))
            print(f'Loaded {len(self.samples)} samples from {txt_file} (classes: 0-{num_classes - 1})')

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            (img_path, label) = self.samples[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f'Error loading image {img_path}: {e}')
                image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                image = self.transform(image)
            return (image, label)
    domain_name = args.dataset.split('_')[1]
    full_train_dataset = DomainNetDataset(data_dir=data_dir, domain=domain_name, split='train', transform=transform, num_classes=100)
    full_test_dataset = DomainNetDataset(data_dir=data_dir, domain=domain_name, split='test', transform=transform, num_classes=100)
    args.num_labels = 100

def create_data_subset(dataset, subset_ratio):
    if subset_ratio >= 1.0:
        return dataset
    total_size = len(dataset)
    subset_size = int(total_size * subset_ratio)
    indices = np.random.permutation(total_size)[:subset_size]
    return Subset(dataset, indices)
full_train_dataset = create_data_subset(full_train_dataset, args.data_subset_ratio)
full_test_dataset = create_data_subset(full_test_dataset, args.data_subset_ratio)
print(f'Dataset subset sizes - train: {len(full_train_dataset)}, test: {len(full_test_dataset)}')

def split_data(dataset, num_clients=args.num_clients, distribution=args.distribution):
    if distribution == 'IID':
        indices = np.random.permutation(len(dataset))
        return np.array_split(indices, num_clients)
    elif distribution == 'NON-IID':
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        if args.dataset.startswith('domainnet'):
            num_classes = 100
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
            class_indices_c = class_indices[c].copy()
            np.random.shuffle(class_indices_c)
            start = 0
            for client_id in range(num_clients):
                end = start + proportions[client_id]
                client_indices[client_id].extend(class_indices_c[start:end])
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
    from peft import LoraConfig, get_peft_model, TaskType
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
        if args.algorithm == 'ILORA':
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
        print(f'Initializing client {cid}')
        client_rank = args.heterogeneous_rank_clients[int(cid)]
        print(f'Client {cid} initialized with rank={client_rank}')
        self.model = get_model(client_rank)
        if args.use_control == 'True':
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
                print(f'Client {self.cid}: applying FEDIT parameter truncation')
                print(f'Client rank: {client_rank}, max rank: {max_rank}')
                print('Applying parameter truncation...')
                truncated_state_dict = {}
                for ((name, _), param) in zip(params_dict, parameters):
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
        if args.scheduler != 'None':
            if args.scheduler == 'StepLR':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            elif args.scheduler == 'CosineAnnealing':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.min_lr)
            elif args.scheduler == 'Exponential':
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
            elif args.scheduler == 'ReduceLROnPlateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=args.gamma, min_lr=args.min_lr, verbose=True)
        self.model.train()
        K = 0
        total_samples = len(self.train_loader.dataset)
        subset_size = max(1, int(total_samples * 0.1))
        import random
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)
        subset_indices = all_indices[:subset_size]
        from torch.utils.data import Subset
        subset_dataset = Subset(self.train_loader.dataset, subset_indices)
        subset_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True)
        actual_num_examples = 0
        for epoch in range(config.get('epochs', args.local_epochs)):
            running_loss = 0.0
            batch_count = 0
            for (inputs, labels) in subset_loader:
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
                actual_num_examples += labels.size(0)
            if scheduler is not None:
                if args.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(running_loss / batch_count)
                else:
                    scheduler.step()
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
        return (self.get_parameters({}), actual_num_examples, {'client_id': int(self.cid)})

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
    if args.training_mode == 'Homo':
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
