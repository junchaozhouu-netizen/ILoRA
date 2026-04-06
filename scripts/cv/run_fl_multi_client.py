import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DataCollatorWithPadding
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from peft import LoraConfig, get_peft_model
import numpy as np
import flwr as fl
from flwr.common import Metrics
from typing import Dict, List, Tuple, Optional
import os
import time
import argparse
parser = argparse.ArgumentParser(description='Federated LoRA')
parser.add_argument('--cuda_device', type=str, default='0', help='Visible GPU device IDs.')
parser.add_argument('--training_mode', type=str, default='Homo', choices=['Centralized', 'Homo'], help='Training mode: Centralized or Homo.')
parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory.')
parser.add_argument('--lora_r_client', type=int, default=4, help='Client-side LoRA rank.')
parser.add_argument('--lora_r_server', type=int, default=6, help='Server-side LoRA rank.')
parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA scaling factor.')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate.')
parser.add_argument('--target_modules', type=list, default=['query', 'value'], help='Target modules for LoRA.')
parser.add_argument('--model_name', type=str, default='roberta-base', help='Backbone model name.')
parser.add_argument('--num_labels', type=int, help='Number of output labels.')
parser.add_argument('--fraction_fit', type=float, default=1.0, help='Fraction of clients used for training in each round.')
parser.add_argument('--min_fit_clients', type=int, default=3, help='Minimum number of training clients per round.')
parser.add_argument('--min_evaluate_clients', type=int, default=3, help='Minimum number of evaluation clients per round.')
parser.add_argument('--num_rounds', type=int, default=5, help='Total number of training rounds.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
parser.add_argument('--local_epochs', type=int, default=1, help='Number of local training epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--num_clients', type=int, default=3, help='Number of clients.')
parser.add_argument('--algorithm', type=str, default='ILORA', choices=['FFA-LORA', 'FEDIT', 'FLORA', 'ILORA', 'LoRA_FAIR'], help='Federated LoRA algorithm.')
parser.add_argument('--save_dir', type=str, default='/path/to/save_dir', help='Directory used to save outputs.')
parser.add_argument('--dataset', type=str, default='QQP', choices=['QQP', 'MNLI', 'STS-B', 'WNLI', 'RTE', 'MRPC', 'qnli', 'cola', 'sst2', 'SNLI', 'IMDB', 'AG_NEWS', 'DBPedia', 'ANLI', 'PAWS', 'AMAZON_POL', 'DBPEDIA14', 'YELP_POLARITY', 'YAHOO_ANS', 'TREC', 'SICK', 'YELP_REVIEW_FULL', '20NEWSGROUPS', 'EMOTION', 'sst5', 'MR', 'HateXplain'], help='Dataset name.')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'], help='Optimizer type.')
parser.add_argument('--distribution', type=str, default='NON-IID', choices=['IID', 'NON-IID'], help='Data distribution type: IID or NON-IID.')
parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha controlling Non-IID severity.')
parser.add_argument('--heterogeneous_rank', type=str, default='True', choices=['True', 'False'], help='Enable heterogeneous LoRA ranks.')
parser.add_argument('--heterogeneous_rank_clients', type=str, default='2,4,6', help='Per-client LoRA ranks.')
parser.add_argument('--use_control', type=str, default='False', choices=['True', 'False'], help='Enable control variates.')
parser.add_argument('--lambda_reg', type=float, default=0.01, help='Regularization coefficient for LoRA_FAIR.')
parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length.')
parser.add_argument('--lora_scale_factor', type=float, default=0.5, help='Scale factor used by QR initialization.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--QR_init', type=str, default='Auto', choices=['Auto', 'True', 'False'], help='QR initialization mode.')
args = parser.parse_args()
import random

def seed_local_for_client(base_seed: int, cid: int):
    s = int(base_seed) + int(cid)
    import random, numpy as np, torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_everything(seed: int=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        pass
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except Exception:
        pass
seed_everything(args.seed)
if args.heterogeneous_rank == 'True':
    print('Using heterogeneous client ranks.')
    rank_list = [int(r) for r in args.heterogeneous_rank_clients.split(',')]
    args.heterogeneous_rank_clients = []
    for i in range(args.num_clients):
        if i < 5:
            args.heterogeneous_rank_clients.append(rank_list[0])
        elif i < 10:
            args.heterogeneous_rank_clients.append(rank_list[1])
        else:
            args.heterogeneous_rank_clients.append(rank_list[2])
    print(f'Heterogeneous rank assignment: {args.heterogeneous_rank_clients}')
else:
    print('Using heterogeneous client ranks.')
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
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
data_dir = args.data_dir
if args.model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
if args.dataset == 'QQP':
    dataset_path = f'{args.data_dir}/glue/qqp'
    dataset = load_dataset(dataset_path)

    def preprocess_function(example):
        return tokenizer(example['question1'], example['question2'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    args.num_labels = 2
if args.dataset == 'YAHOO_ANS':
    from datasets import load_dataset
    import os
    local_dir = os.path.join(args.data_dir, 'yahoo_answers_topics')
    required = ['train-00000-of-00002.parquet', 'train-00001-of-00002.parquet', 'test-00000-of-00001.parquet']
    if not all((os.path.isfile(os.path.join(local_dir, f)) for f in required)):
        raise FileNotFoundError('Yahoo! Answers parquet \n' + local_dir + '\n' + 'Message' + ', '.join(required))
    dataset = load_dataset(local_dir)
    SUBSET_SIZE_TRAIN = 100000
    SUBSET_SIZE_TEST = 10000
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(min(SUBSET_SIZE_TRAIN, len(dataset['train']))))
    dataset['test'] = dataset['test'].shuffle(seed=42).select(range(min(SUBSET_SIZE_TEST, len(dataset['test']))))
    print(f"YAHOO_ANS {len(dataset['train'])} {len(dataset['test'])}")
    labels = [int(ex['topic']) for ex in dataset['train']]
    min_label = min(labels)
    max_label = max(labels)
    args.num_labels = max_label - min_label + 1
    print(f'Label range: {min_label} to {max_label}, num_labels = {args.num_labels}')

    def fix_labels(example):
        example['label'] = int(example['topic']) - min_label
        return example
    dataset = dataset.map(fix_labels)

    def concat_text(example):
        title = example.get('question_title', '')
        content = example.get('question_content', '')
        answer = example.get('best_answer', '')
        example['text'] = f'{title} {content} {answer}'.strip()
        return example
    dataset = dataset.map(concat_text)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
elif args.dataset == 'DBPEDIA14':
    from datasets import load_dataset
    import numpy as np
    dataset = load_dataset('dbpedia_14')
    target_train_size = 50000
    target_test_size = 10000
    print(f'DBPedia14...')
    print(f": {len(dataset['train'])}")
    print(f": {len(dataset['test'])}")
    from sklearn.model_selection import train_test_split
    train_labels = dataset['train']['label']
    test_labels = dataset['test']['label']
    train_indices = np.arange(len(dataset['train']))
    test_indices = np.arange(len(dataset['test']))
    if len(dataset['train']) > target_train_size:
        (sampled_train_indices, _) = train_test_split(train_indices, train_size=target_train_size, random_state=42, stratify=train_labels)
        sampled_train_dataset = dataset['train'].select(sampled_train_indices)
    else:
        sampled_train_dataset = dataset['train']
    if len(dataset['test']) > target_test_size:
        (sampled_test_indices, _) = train_test_split(test_indices, train_size=target_test_size, random_state=42, stratify=test_labels)
        sampled_test_dataset = dataset['test'].select(sampled_test_indices)
    else:
        sampled_test_dataset = dataset['test']
    print(f': {len(sampled_train_dataset)}')
    print(f': {len(sampled_test_dataset)}')
    sampled_dataset = {'train': sampled_train_dataset, 'test': sampled_test_dataset}

    def preprocess_function(examples):
        titles = examples.get('title', None)
        contents = examples.get('content', None)
        if isinstance(titles, list) and isinstance(contents, list):
            texts = [((t or '') + ' ' + (c or '')).strip() for (t, c) in zip(titles, contents)]
        else:
            t = examples.get('title') or ''
            c = examples.get('content') or ''
            texts = (t + ' ' + c).strip()
        return tokenizer(texts, truncation=True, padding='max_length', max_length=64)
    cols = sampled_dataset['train'].column_names
    cols_to_remove = [c for c in cols if c != 'label']
    tokenized = {'train': sampled_dataset['train'].map(preprocess_function, batched=True, remove_columns=cols_to_remove), 'test': sampled_dataset['test'].map(preprocess_function, batched=True, remove_columns=cols_to_remove)}
    for split in ['train', 'test']:
        tokenized[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized['train']
    full_test_dataset = tokenized['test']
    args.num_labels = 14
    print(f'DBPedia14 ')
    print(f': {len(full_train_dataset)}')
    print(f': {len(full_test_dataset)}')
if args.dataset == 'IMDB':
    try:
        dataset_path = f'{args.data_dir}/imdb'
        dataset = load_dataset(dataset_path)
    except:
        try:
            print('Hugging Face HubIMDB...')
            dataset = load_dataset('imdb', cache_dir=args.data_dir)
        except ConnectionError as e:
            print(f': {e}')
            print('')
            exit(1)
    args.num_labels = 2

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
if args.dataset == 'AG_NEWS':
    try:
        dataset_path = f'{args.data_dir}/ag_news'
        dataset = load_dataset(dataset_path)
    except:
        try:
            print('Hugging Face HubAG_NEWS...')
            dataset = load_dataset('ag_news', cache_dir=args.data_dir)
        except ConnectionError as e:
            print(f': {e}')
            print('')
            exit(1)
    args.num_labels = 4

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
if args.dataset == 'qnli':
    dataset_path = f'{args.data_dir}/glue/qnli'
    dataset = load_dataset(dataset_path)

    def preprocess_function(examples):
        inputs = tokenizer(examples['question'], examples['sentence'], truncation=True, max_length=64, padding='max_length')
        labels = [1 if label == 'entailment' else 0 for label in examples['label']]
        inputs['labels'] = labels
        return inputs
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    args.num_labels = 2
if args.dataset == 'sst2':

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', return_tensors='pt', max_length=64)
    dataset_path = f'{args.data_dir}/glue/sst2'
    dataset = load_dataset(dataset_path)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    args.num_labels = 2

def seed_worker(worker_id):
    worker_seed = (args.seed + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def split_data(dataset, num_clients=args.num_clients, distribution=args.distribution):
    if args.dataset == 'STS-B' or args.num_labels == 1:
        y = np.array(dataset['label'], dtype=np.float32)
        K = 5
        bins = np.linspace(y.min(), y.max(), K + 1)
        pseudo_cls = np.digitize(y, bins[:-1], right=False) - 1
        pseudo_cls = np.clip(pseudo_cls, 0, K - 1)
        num_classes = K
        targets = pseudo_cls
    else:
        targets = np.array(dataset['label'])
        num_classes = args.num_labels
    if distribution == 'IID':
        indices = np.random.permutation(len(targets))
        return np.array_split(indices, num_clients)
    elif distribution == 'NON-IID':
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
        for i in range(num_clients):
            if len(client_indices[i]) == 0:
                extra = np.random.choice(np.arange(len(targets)), size=max(1, len(targets) // (20 * num_clients)), replace=False)
                client_indices[i].extend(extra.tolist())
        return [np.array(idx) for idx in client_indices]
    else:
        raise ValueError(f'Unknown data distribution type: {distribution}')
train_indices = split_data(full_train_dataset, distribution=args.distribution)
test_indices = split_data(full_test_dataset, distribution=args.distribution)

def apply_qr_init(model: nn.Module, r: int, scale_factor: float):
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
            s = torch.sqrt(scale)
            Q_r = Q_r * s
            R_r = R_r * s
            delta_W = Q_r @ R_r * scale_factor
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

def get_model(r):
    import transformers
    transformers.logging.set_verbosity_error()
    if args.model_name == 'roberta-base':
        print('Loading RoBERTa model...')
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    merged_model_path = os.path.join(args.save_dir, 'merged_model.pth')
    if os.path.exists(merged_model_path) and args.algorithm == 'FLORA':
        print('Loading resources...')
        model.load_state_dict(torch.load(merged_model_path))
    else:
        print('')
    from peft import LoraConfig, get_peft_model, TaskType
    if args.model_name == 'roberta-base':
        lora_config = LoraConfig(r=r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=['query', 'value'], bias='none', task_type=TaskType.SEQ_CLS, modules_to_save=['classifier'])
    model = get_peft_model(model, lora_config)
    enable_qr = args.QR_init == 'True' or (args.QR_init == 'Auto' and args.algorithm == 'ILORA')
    if enable_qr:
        print(f'[QR_init] Applying QR initialization: r={r}, scale_factor={args.lora_scale_factor}')
        apply_qr_init(model, r=r, scale_factor=args.lora_scale_factor)
    else:
        print('[QR_init] Disabled.')
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
        results = sorted(results, key=lambda x: x[1].metrics['client_id'])
        if args.algorithm == 'FEDIT':
            print('Using FedIT (Federated Incremental Training).')
            print('FedIT aggregation weight summary:')
            total_examples = sum([metrics.num_examples for (_, metrics) in results])
            if args.heterogeneous_rank == 'True':
                print('FEDIT with heterogeneous ranks:')
                max_rank = max(args.heterogeneous_rank_clients)
                print('FEDIT max rank:', max_rank)
                num_classifier_params = 4
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'Client weights: {client_weights}')
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'- weight: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    print(f'- LoRA rank: {client_rank}')
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    lora_params = client_params[:-num_classifier_params]
                    classifier_params = client_params[-num_classifier_params:]
                    padded_lora_params = []
                    for param in lora_params:
                        if len(param.shape) == 2:
                            if param.shape[0] == client_rank and param.shape[0] < max_rank:
                                padding = ((0, max_rank - param.shape[0]), (0, 0))
                                padded_param = np.pad(param, padding, 'constant')
                                padded_lora_params.append(padded_param)
                            elif param.shape[1] == client_rank and param.shape[1] < max_rank:
                                padding = ((0, 0), (0, max_rank - param.shape[1]))
                                padded_param = np.pad(param, padding, 'constant')
                                padded_lora_params.append(padded_param)
                            else:
                                padded_lora_params.append(param)
                        else:
                            padded_lora_params.append(param)
                    padded_client_params = padded_lora_params + classifier_params
                    if aggregated_params is None:
                        aggregated_params = []
                        for param in padded_client_params:
                            weighted_param = p_k * param
                            aggregated_params.append(weighted_param)
                    else:
                        for (i, param) in enumerate(padded_client_params):
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
        if args.algorithm == 'ILORA':
            if args.heterogeneous_rank == 'True':
                print('ILORA with heterogeneous ranks:')
                num_classifier_params = 4
                if args.use_control == 'True':
                    print('ILORA with heterogeneous ranks and control variates:')
                    c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                    try:
                        self.c = np.load(c_save_path, allow_pickle=True).tolist()
                        print('Initialized control variate c.')
                    except:
                        print(f'c')
                max_rank = max(args.heterogeneous_rank_clients)
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'Total examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'Client weights: {client_weights}')
                if args.use_control == 'True':
                    print('\n control: cic()')
                    delta_c_aggregated = [np.zeros_like(p) for p in self.c]
                    for (client_idx, (_, fit_res)) in enumerate(results):
                        client_idx = fit_res.metrics['client_id']
                        delta_ci_path = os.path.join(args.save_dir, f'client_{client_idx}_delta_ci.npy')
                        try:
                            delta_ci = np.load(delta_ci_path, allow_pickle=True).tolist()
                            print(f'\n {client_idx} ci :')
                            client_rank = args.heterogeneous_rank_clients[client_idx]
                            lora_delta_ci = delta_ci[:-num_classifier_params]
                            classifier_delta_ci = delta_ci[-num_classifier_params:]
                            padded_lora_delta_ci = []
                            for ci_val in lora_delta_ci:
                                if len(ci_val.shape) == 2:
                                    if ci_val.shape[0] == client_rank and ci_val.shape[0] < max_rank:
                                        padding = ((0, max_rank - ci_val.shape[0]), (0, 0))
                                        padded_param = np.pad(ci_val, padding, 'constant')
                                        padded_lora_delta_ci.append(padded_param)
                                    elif ci_val.shape[1] == client_rank and ci_val.shape[1] < max_rank:
                                        padding = ((0, 0), (0, max_rank - ci_val.shape[1]))
                                        padded_param = np.pad(ci_val, padding, 'constant')
                                        padded_lora_delta_ci.append(padded_param)
                                    else:
                                        padded_lora_delta_ci.append(ci_val)
                                else:
                                    padded_lora_delta_ci.append(ci_val)
                            padded_delta_ci = padded_lora_delta_ci + classifier_delta_ci
                            for i in range(len(delta_c_aggregated)):
                                delta_c_aggregated[i] += 1 / args.num_clients * padded_delta_ci[i]
                        except Exception as e:
                            print(f'Failed to load client {client_idx} delta_ci file: {str(e)}')
                            continue
                    for i in range(len(self.c)):
                        self.c[i] += delta_c_aggregated[i]
                    if args.use_control == 'True':
                        print('\n c control :')
                        for (i, c) in enumerate(self.c[:3]):
                            print(f'  c[{i}].shape: {c.shape}, mean: {np.mean(c):.4f}')
                        c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                        np.save(c_save_path, np.array(self.c, dtype=object))
                        print(f'Saved global control variate c to {c_save_path}')
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'- weight: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    print(f'- LoRA rank: {client_rank}')
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    lora_params = client_params[:-num_classifier_params]
                    classifier_params = client_params[-num_classifier_params:]
                    padded_lora_params = []
                    for param in lora_params:
                        if len(param.shape) == 2:
                            if param.shape[0] == client_rank and param.shape[0] < max_rank:
                                padding = ((0, max_rank - param.shape[0]), (0, 0))
                                padded_param = np.pad(param, padding, 'constant')
                                padded_lora_params.append(padded_param)
                            elif param.shape[1] == client_rank and param.shape[1] < max_rank:
                                padding = ((0, 0), (0, max_rank - param.shape[1]))
                                padded_param = np.pad(param, padding, 'constant')
                                padded_lora_params.append(padded_param)
                            else:
                                padded_lora_params.append(param)
                        else:
                            padded_lora_params.append(param)
                    padded_client_params = padded_lora_params + classifier_params
                    if aggregated_params is None:
                        aggregated_params = []
                        param_idx = 0
                        for param in padded_lora_params:
                            if len(param.shape) == 2:
                                if param.shape[0] == max_rank:
                                    aggregated_params.append(p_k * param)
                                    param_idx += 1
                                elif param.shape[1] == max_rank:
                                    aggregated_params.append(param)
                                    param_idx += 1
                                else:
                                    aggregated_params.append(p_k * param)
                                    param_idx += 1
                            else:
                                aggregated_params.append(p_k * param)
                                param_idx += 1
                        for param in classifier_params:
                            aggregated_params.append(p_k * param)
                    else:
                        param_idx = 0
                        for param in padded_lora_params:
                            if len(param.shape) == 2:
                                if param.shape[0] == max_rank:
                                    aggregated_params[param_idx] = np.concatenate([aggregated_params[param_idx], p_k * param], axis=0)
                                    param_idx += 1
                                elif param.shape[1] == max_rank:
                                    aggregated_params[param_idx] = np.concatenate([aggregated_params[param_idx], param], axis=1)
                                    param_idx += 1
                                else:
                                    aggregated_params[param_idx] += p_k * param
                                    param_idx += 1
                            else:
                                aggregated_params[param_idx] += p_k * param
                                param_idx += 1
                        for (i, param) in enumerate(classifier_params):
                            aggregated_params[param_idx + i] += p_k * param
                num_layers = (len(aggregated_params) - 2) // 4
                print(f'Number of layers: {num_layers}')
                Decomposition_save_paths = [os.path.join(args.save_dir, f'update_LORA_Decomposition_client{i}.pth') for i in range(len(args.heterogeneous_rank_clients))]
                for (rank_idx, target_rank) in enumerate(args.heterogeneous_rank_clients):
                    print(f'\n {target_rank} QR...')
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
                    print(f'Saved rank-{target_rank} parameters to {Decomposition_save_paths[rank_idx]}')
                if server_round == args.num_rounds - 1 and max_rank_params is not None:
                    print('\n...')
                    from flwr.common import ndarrays_to_parameters
                    max_rank_parameters = ndarrays_to_parameters(max_rank_params)
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(max_rank_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    final_model_path = os.path.join(args.save_dir, 'final_global_model.pth')
                    torch.save(self.global_model.state_dict(), final_model_path)
                    print(f'Saved final global model to {final_model_path}')
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                return (aggregated_parameters, aggregated_metrics)
from flwr.client import Client, NumPyClient

class CIFAR10Client(NumPyClient):

    def __init__(self, cid, train_indices, test_indices):
        seed_local_for_client(args.seed, cid)
        self.cid = cid
        print(f'Initializing client {cid}')
        client_rank = args.heterogeneous_rank_clients[int(cid)]
        print(f'Client {cid} initialized with rank={client_rank}')
        self.model = get_model(client_rank)
        if args.use_control == 'True':
            self.ci = [torch.zeros_like(p).cpu().numpy() for p in self.model.parameters() if p.requires_grad]
        client_gen = torch.Generator()
        client_gen.manual_seed(args.seed + int(cid))
        self.train_loader = DataLoader(Subset(full_train_dataset, train_indices[cid]), batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=client_gen)
        self.test_loader = DataLoader(Subset(full_test_dataset, test_indices[cid]), batch_size=args.batch_size, worker_init_fn=seed_worker, generator=client_gen)

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for (name, val) in self.model.named_parameters() if val.requires_grad]

    def set_parameters(self, parameters):
        params_dict = zip([name for (name, param) in self.model.named_parameters() if param.requires_grad], parameters)
        state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
        if args.algorithm == 'ILORA':
            if args.heterogeneous_rank == 'False':
                self.model.load_state_dict(state_dict, strict=False)
            elif args.heterogeneous_rank == 'True':
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                param_path = os.path.join(args.save_dir, f'update_LORA_Decomposition_client{self.cid}.pth')
                if os.path.exists(param_path):
                    print(f'Client {self.cid}: loading decomposition parameters for rank {client_rank}...')
                    trainable_state = torch.load(param_path)
                    self.model.load_state_dict(trainable_state, strict=False)
                else:
                    print(f'Warning: decomposition parameters for rank {client_rank} were not found: {param_path}')
        if args.algorithm == 'FEDIT':
            params_dict = zip([name for (name, param) in self.model.named_parameters() if param.requires_grad], parameters)
            if args.heterogeneous_rank == 'False':
                self.model.load_state_dict(state_dict, strict=False)
            elif args.heterogeneous_rank == 'True':
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                max_rank = max(args.heterogeneous_rank_clients)
                print(f' {self.cid} FEDIT')
                print(f': {client_rank}, : {max_rank}')
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
        print('Starting local fit.')
        seed_local_for_client(args.seed, int(self.cid))
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
            print('Client rank during fit:', client_rank)
            self.model = get_model(client_rank)
        initial_params = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
        if args.use_control == 'True':
            ci_path = os.path.join(args.save_dir, f'client_{self.cid}_ci.npy')
            if os.path.exists(ci_path):
                self.ci = np.load(ci_path, allow_pickle=True).tolist()
                print(f'Client {self.cid}: loaded historical ci values.')
            c_save_path = os.path.join(args.save_dir, 'global_c.npy')
            try:
                c = np.load(c_save_path, allow_pickle=True).tolist()
                print(f'Client {self.cid}: loaded global control variate c.')
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                max_rank = max(args.heterogeneous_rank_clients)
                truncated_c = []
                for (param_idx, param) in enumerate(c):
                    if len(param.shape) == 2:
                        if param.shape[0] == max_rank and param.shape[0] > client_rank:
                            truncated_param = param[:client_rank, :]
                            print(f'Truncated parameter {param_idx}: {param.shape} -> {truncated_param.shape} (row truncation)')
                        elif param.shape[1] == max_rank and param.shape[1] > client_rank:
                            truncated_param = param[:, :client_rank]
                            print(f'Truncated parameter {param_idx}: {param.shape} -> {truncated_param.shape} (row truncation)')
                        else:
                            truncated_param = param
                        truncated_c.append(truncated_param)
                    else:
                        truncated_c.append(param)
                c = truncated_c
                print(f'Client {self.cid} (rank={client_rank}) truncated global control variate c.')
            except:
                c = self.ci
                print(f'Client {self.cid}: failed to load global c, using ci instead.')
            if len(c) != len(self.ci):
                print('Warning: loaded c and ci have different lengths; using ci instead.')
                c = self.ci
        import copy
        if args.use_control == 'True':
            original_ci = copy.deepcopy(self.ci)
            control_option = 1
            if control_option == 1:
                print('Using control Option I: compute gradients on the full local dataset.')
                original_mode = self.model.training
                self.model.eval()
                self.model.zero_grad()
                total_samples = 0
                for batch in self.train_loader:
                    inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
                    labels = batch['label'].to(device)
                    outputs = self.model(**inputs).logits
                    if args.dataset == 'STS-B':
                        labels = labels.float()
                    else:
                        labels = labels.long()
                    if args.dataset == 'STS-B':
                        loss_fn = nn.MSELoss()
                        loss = loss_fn(outputs.squeeze(), labels)
                    else:
                        loss_fn = nn.CrossEntropyLoss()
                        loss = loss_fn(outputs, labels)
                    loss.backward()
                    total_samples += labels.size(0)
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad /= total_samples
                new_ci = [param.grad.clone().detach().cpu().numpy() for param in self.model.parameters() if param.requires_grad]
                self.model.train(original_mode)
                print(f' {self.cid} Option I')
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.get('lr', args.lr), momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.get('lr', args.lr), weight_decay=args.weight_decay)
        self.model.train()
        K = 0
        for epoch in range(config.get('epochs', args.local_epochs)):
            for batch in self.train_loader:
                K += 1
                inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
                labels = batch['label'].to(device)
                if args.dataset == 'STS-B':
                    labels = labels.float()
                else:
                    labels = labels.long()
                optimizer.zero_grad()
                outputs = self.model(**inputs).logits
                if args.dataset == 'STS-B':
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(outputs.squeeze(), labels)
                else:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(outputs, labels)
                loss.backward()
                if args.use_control == 'True':
                    with torch.no_grad():
                        for (param, ci_val, c_val) in zip([p for p in self.model.parameters() if p.requires_grad], self.ci, c):
                            param.grad.add_(torch.tensor(c_val - ci_val).to(device))
                optimizer.step()
        if args.use_control == 'True':
            self.ci = new_ci
            ci_save_path = os.path.join(args.save_dir, f'client_{self.cid}_ci.npy')
            np.save(ci_save_path, np.array(self.ci, dtype=object))
            print(f' {self.cid} ci {ci_save_path}')
            delta_ci = [ni - oi for (ni, oi) in zip(self.ci, original_ci)]
            delta_ci_path = os.path.join(args.save_dir, f'client_{self.cid}_delta_ci.npy')
            np.save(delta_ci_path, np.array(delta_ci, dtype=object))
            print(f'{self.cid} ci {delta_ci_path}')
        return (self.get_parameters({}), len(self.train_loader.dataset), {'client_id': int(self.cid)})

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['accuracy'] for (num_examples, m) in metrics]
    examples = [num_examples for (num_examples, _) in metrics]
    return {'accuracy': sum(accuracies) / sum(examples)}

def get_evaluate_fn(test_loader):

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
        (loss, accuracy) = (0.0, 0.0)
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
                labels = batch['label'].to(device)
                outputs = model(**inputs).logits
                from scipy.stats import spearmanr
                if args.dataset == 'STS-B' or (args.dataset == 'SICK' and args.num_labels == 1):
                    loss_fn = nn.MSELoss()
                    loss += loss_fn(outputs.squeeze(), labels.float()).item()
                    preds = outputs.squeeze().cpu().numpy()
                    targets = labels.cpu().numpy()
                    accuracy += spearmanr(preds, targets).correlation
                else:
                    loss_fn = nn.CrossEntropyLoss()
                    loss += loss_fn(outputs, labels.long()).item()
                    (_, predicted) = torch.max(outputs, 1)
                    accuracy += (predicted == labels).sum().item()
                total += labels.size(0)
        if args.dataset == 'STS-B' or (args.dataset == 'SICK' and args.num_labels == 1):
            accuracy /= len(test_loader)
        else:
            accuracy /= total
        print(f'Server-side evaluation loss: {loss:.4f}, accuracy: {accuracy:.4f}')
        return (loss, {'accuracy': accuracy})
    return evaluate

def main():
    start_time = time.time()
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)
    if args.training_mode == 'Homo':
        test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size)
        strategy = FedLoRAStrategy(args.lora_r_server, use_control=args.use_control, fraction_fit=0.8, fraction_evaluate=0.0, min_fit_clients=int(args.num_clients * 0.8), min_evaluate_clients=0, min_available_clients=args.num_clients, evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(test_loader))

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
