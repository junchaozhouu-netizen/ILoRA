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
parser = argparse.ArgumentParser(description='Federated LoRA for NLP')
parser.add_argument('--cuda_device', type=str, default='0', help='GPU device ID (default: 3)')
parser.add_argument('--training_mode', type=str, default='Homo', choices=['Centralized', 'Homo'], help='Training mode: Centralized or Homo (default: Homo)')
parser.add_argument('--data_dir', type=str, default='./data', help='Data directory (default: "./data")')
parser.add_argument('--lora_r_client', type=int, default=4, help='Client LoRA rank (default: 4)')
parser.add_argument('--lora_r_server', type=int, default=6, help='Server LoRA rank (default: 4)')
parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha (default: 16)')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout (default: 0.1)')
parser.add_argument('--target_modules', type=list, default=['query', 'value'], help='Target modules for LoRA (default: ["query", "value"])')
parser.add_argument('--model_name', type=str, default='roberta-base', help='Model name (default: "roberta-base")')
parser.add_argument('--num_labels', type=int, help='Number of labels (e.g., CIFAR-10: 10, CIFAR-100: 100)')
parser.add_argument('--fraction_fit', type=float, default=1.0, help='Fraction of clients used for training (default: 1.0)')
parser.add_argument('--min_fit_clients', type=int, default=3, help='Minimum number of fit clients (default: 3)')
parser.add_argument('--min_evaluate_clients', type=int, default=3, help='Minimum number of evaluation clients (default: 3)')
parser.add_argument('--num_rounds', type=int, default=5, help='Number of communication rounds (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs (default: 1)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size per client/device (default: 128)')
parser.add_argument('--num_clients', type=int, default=3, help='Number of clients (default: 3)')
parser.add_argument('--algorithm', type=str, default='ILORA', choices=['FFA-LORA', 'FEDIT', 'FLORA', 'ILORA', 'LoRA_FAIR'], help='Federated algorithm (default: FEDIT)')
parser.add_argument('--save_dir', type=str, default='/path/to/save_dir', help='Directory for saving outputs (default: /path/to/save_dir)')
parser.add_argument('--dataset', type=str, default='YELP_REVIEW_FULL', choices=['QQP', 'MNLI', 'STS-B', 'WNLI', 'RTE', 'MRPC', 'qnli', 'cola', 'sst2', 'SNLI', 'IMDB', 'AG_NEWS', 'DBPedia', 'ANLI', 'PAWS', 'AMAZON_POL', 'DBPEDIA14', 'YELP_POLARITY', 'YAHOO_ANS', 'TREC', 'SICK', 'YELP_REVIEW_FULL', '20NEWSGROUPS', 'EMOTION', 'sst5', 'MR', 'HateXplain'], help='Dataset name (default: DBPedia)')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'], help='Optimizer: SGD or AdamW (default: SGD)')
parser.add_argument('--distribution', type=str, default='NON-IID', choices=['IID', 'NON-IID'], help='Data distribution: IID or NON-IID')
parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for NON-IID partitioning')
parser.add_argument('--heterogeneous_rank', type=str, default='True', choices=['True', 'False'], help='Enable heterogeneous LoRA ranks: True or False')
parser.add_argument('--heterogeneous_rank_clients', type=str, default='2,4,6', help='Client LoRA ranks (default: "2,4,6")')
parser.add_argument('--use_control', type=str, default='False', choices=['True', 'False'], help='Enable control variates (default: False)')
parser.add_argument('--lambda_reg', type=float, default=0.01, help='Regularization coefficient for LoRA_FAIR')
parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length')
parser.add_argument('--lora_scale_factor', type=float, default=0.5, help='QR initialization scale factor (default: 1.0)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
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
    print('Heterogeneous rank mode enabled.')
    args.heterogeneous_rank_clients = [int(r) for r in args.heterogeneous_rank_clients.split(',')]
else:
    print('Heterogeneous rank mode disabled.')
    args.heterogeneous_rank_clients = [args.lora_r_client] * args.num_clients
import glob
for file_path in glob.glob(os.path.join(args.save_dir, '*.pth')):
    try:
        os.remove(file_path)
        print(f': {file_path}')
    except Exception as e:
        print(f': {file_path} - {str(e)}')
for file_path in glob.glob(os.path.join(args.save_dir, '*.npy')):
    try:
        os.remove(file_path)
        print(f'npy: {file_path}')
    except Exception as e:
        print(f'npy: {file_path} - {str(e)}')
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
data_dir = args.data_dir
if args.model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
if args.dataset == 'HateXplain':
    from datasets import Dataset, DatasetDict
    import json
    import os
    dataset_dir = os.path.join(args.data_dir, 'HateXplain')
    dataset_file = os.path.join(dataset_dir, 'Data', 'dataset.json')
    split_file = os.path.join(dataset_dir, 'Data', 'post_id_divisions.json')
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f'HateXplain\n: {dataset_file}\n git clone ')
    print('HateXplain...')
    with open(dataset_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    texts = []
    labels = []
    post_ids = []
    for (post_id, item) in raw_data.items():
        if item['post_tokens']:
            text = ' '.join(item['post_tokens'])
            texts.append(text)
            post_ids.append(post_id)
            label_counts = {}
            for annotator in item['annotators']:
                label = annotator['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            majority_label = max(label_counts.items(), key=lambda x: x[1])[0]
            label_map = {'normal': 0, 'offensive': 1, 'hate': 2}
            labels.append(label_map.get(majority_label, 0))
    print(f' {len(texts)} ')
    print(f': ={labels.count(0)}, ={labels.count(1)}, ={labels.count(2)}')
    if os.path.exists(split_file):
        print('...')
        with open(split_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        train_ids = set(splits['train'])
        val_ids = set(splits['val'])
        test_ids = set(splits['test'])
        train_data = {'text': [], 'label': []}
        val_data = {'text': [], 'label': []}
        test_data = {'text': [], 'label': []}
        for (text, label, post_id) in zip(texts, labels, post_ids):
            if post_id in train_ids:
                train_data['text'].append(text)
                train_data['label'].append(label)
            elif post_id in val_ids:
                val_data['text'].append(text)
                val_data['label'].append(label)
            elif post_id in test_ids:
                test_data['text'].append(text)
                test_data['label'].append(label)
        dataset = DatasetDict({'train': Dataset.from_dict(train_data), 'validation': Dataset.from_dict(val_data), 'test': Dataset.from_dict(test_data)})
    else:
        print('...')
        from sklearn.model_selection import train_test_split
        (train_texts, temp_texts, train_labels, temp_labels) = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
        (val_texts, test_texts, val_labels, test_labels) = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
        dataset = DatasetDict({'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}), 'validation': Dataset.from_dict({'text': val_texts, 'label': val_labels}), 'test': Dataset.from_dict({'text': test_texts, 'label': test_labels})})
    args.num_labels = 3

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    print(f'HateXplain ')
    print(f': {len(full_train_dataset)}')
    print(f': {len(full_test_dataset)}')
    print(f": {len(tokenized_dataset['test'])}")
    print(f': {args.num_labels}')
    print(f': 0=, 1=, 2=')
if args.dataset == 'MR':
    import os
    from datasets import Dataset, DatasetDict
    dataset_dir = os.path.join(args.data_dir, 'review_polarity', 'txt_sentoken')
    pos_dir = os.path.join(dataset_dir, 'pos')
    neg_dir = os.path.join(dataset_dir, 'neg')
    if not os.path.exists(pos_dir) or not os.path.exists(neg_dir):
        raise FileNotFoundError(f'MR \n: {pos_dir}\n: {neg_dir}\n')
    texts = []
    labels = []
    print('...')
    for filename in os.listdir(pos_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(pos_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        labels.append(1)
            except Exception as e:
                print(f' {file_path} : {e}')
    print('...')
    for filename in os.listdir(neg_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(neg_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        labels.append(0)
            except Exception as e:
                print(f' {file_path} : {e}')
    print(f' {len(texts)} ')
    print(f': {labels.count(1)} ')
    print(f': {labels.count(0)} ')
    from sklearn.model_selection import train_test_split
    (train_texts, temp_texts, train_labels, temp_labels) = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
    (val_texts, test_texts, val_labels, test_labels) = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
    args.num_labels = 2

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    print(f'MR ')
    print(f': {len(full_train_dataset)}')
    print(f': {len(full_test_dataset)}')
    print(f": {len(tokenized_dataset['test'])}")
    print(f': {args.num_labels}')
if args.dataset == 'sst5':
    from datasets import load_dataset
    import json
    import os
    dataset_path = os.path.join(args.data_dir, 'sst5')
    train_file = os.path.join(dataset_path, 'train.jsonl')
    dev_file = os.path.join(dataset_path, 'dev.jsonl')
    test_file = os.path.join(dataset_path, 'test.jsonl')
    if not os.path.exists(train_file) or not os.path.exists(dev_file) or (not os.path.exists(test_file)):
        raise FileNotFoundError(f'SST-5 \n: {train_file}\n: {dev_file}\n: {test_file}\n')
    dataset = load_dataset('json', data_files={'train': train_file, 'validation': dev_file, 'test': test_file})
    args.num_labels = 5
    sst5_labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    print(f'SST-5 : {sst5_labels}')

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    print(f'SST-5 ')
    print(f': {len(full_train_dataset)}')
    print(f': {len(full_test_dataset)}')
    print(f": {len(tokenized_dataset['test'])}")
    print(f': {args.num_labels}')
if args.dataset == 'EMOTION':
    from datasets import load_dataset
    import os
    dataset_path = os.path.join(args.data_dir, 'EMOTION')
    train_file = os.path.join(dataset_path, 'train-00000-of-00001.parquet')
    validation_file = os.path.join(dataset_path, 'validation-00000-of-00001.parquet')
    test_file = os.path.join(dataset_path, 'test-00000-of-00001.parquet')
    if not os.path.exists(train_file) or not os.path.exists(validation_file) or (not os.path.exists(test_file)):
        raise FileNotFoundError(f'Emotion \n: {train_file}\n: {validation_file}\n: {test_file}\n')
    dataset = load_dataset('parquet', data_files={'train': train_file, 'validation': validation_file, 'test': test_file})
    args.num_labels = 6
    emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    print(f'Emotion : {emotion_labels}')

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    print(f'Emotion ')
    print(f': {len(full_train_dataset)}')
    print(f': {len(full_test_dataset)}')
    print(f": {len(tokenized_dataset['test'])}")
    print(f': {args.num_labels}')
if args.dataset == '20NEWSGROUPS':
    import tarfile
    import os
    from sklearn.datasets import fetch_20newsgroups
    from datasets import Dataset, DatasetDict
    import pandas as pd
    dataset_dir = os.path.join(args.data_dir, '20news-bydate')
    train_dir = os.path.join(dataset_dir, '20news-bydate-train')
    test_dir = os.path.join(dataset_dir, '20news-bydate-test')
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print('20 Newsgroups...')

        def load_20newsgroups_from_dir(data_dir):
            data = []
            labels = []
            label_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            label_to_id = {name: idx for (idx, name) in enumerate(label_names)}
            for label_name in label_names:
                label_dir = os.path.join(data_dir, label_name)
                if not os.path.isdir(label_dir):
                    continue
                for filename in os.listdir(label_dir):
                    filepath = os.path.join(label_dir, filename)
                    if os.path.isfile(filepath):
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            data.append(content)
                            labels.append(label_to_id[label_name])
                        except Exception as e:
                            print(f' {filepath} : {e}')
                            continue
            return (data, labels, label_names)
        (train_data, train_labels, label_names) = load_20newsgroups_from_dir(train_dir)
        (test_data, test_labels, _) = load_20newsgroups_from_dir(test_dir)
        train_dataset = Dataset.from_dict({'text': train_data, 'label': train_labels})
        test_dataset = Dataset.from_dict({'text': test_data, 'label': test_labels})
        dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
        args.num_labels = len(label_names)
        print(f'20 Newsgroups num_labels: {args.num_labels}')
        print(f': {len(train_data)}')
        print(f': {len(test_data)}')
    else:
        print('20 Newsgroups not found locally. Falling back to scikit-learn...')
        try:
            newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), data_home=args.data_dir, download_if_missing=True)
            newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), data_home=args.data_dir, download_if_missing=True)
            train_dataset = Dataset.from_dict({'text': newsgroups_train.data, 'label': newsgroups_train.target})
            test_dataset = Dataset.from_dict({'text': newsgroups_test.data, 'label': newsgroups_test.target})
            dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
            args.num_labels = len(newsgroups_train.target_names)
            print(f'20 Newsgroups num_labels: {args.num_labels}')
            print(f': {len(newsgroups_train.data)}')
            print(f': {len(newsgroups_test.data)}')
        except Exception as e:
            print(f'20 Newsgroups: {e}')
            print('')
            exit(1)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
    print('20 Newsgroups ')
from datasets import load_dataset
import random
if args.dataset == 'QQP':
    dataset_path = f'{args.data_dir}/glue/qqp'
    dataset = load_dataset(dataset_path)

    def preprocess_function(example):
        return tokenizer(example['question1'], example['question2'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_size = len(tokenized_dataset['train'])
    test_size = len(tokenized_dataset['validation'])
    train_subset_size = train_size // 2
    test_subset_size = test_size // 2
    full_train_dataset = tokenized_dataset['train'].select(range(train_subset_size))
    full_test_dataset = tokenized_dataset['validation'].select(range(test_subset_size))
    args.num_labels = 2
    print(f'QQP half-size subset: train={train_subset_size}, test={test_subset_size}')
if args.dataset == 'YELP_POLARITY':
    import pandas as pd
    from datasets import Dataset, DatasetDict
    import numpy as np
    from sklearn.model_selection import train_test_split
    print('Yelp Review Polarity...')
    train_df = pd.read_csv(f'{args.data_dir}/Yelp reviews - Polarity/data/Yelp reviews - Polarity/yelp_review_polarity_csv/train.csv', header=None, names=['label', 'text'])
    test_df = pd.read_csv(f'{args.data_dir}/Yelp reviews - Polarity/data/Yelp reviews - Polarity/yelp_review_polarity_csv/test.csv', header=None, names=['label', 'text'])
    train_df['label'] = train_df['label'] - 1
    test_df['label'] = test_df['label'] - 1
    print(f': {len(train_df)}')
    print(f': {len(test_df)}')
    print(f":\n{train_df['label'].value_counts().sort_index()}")
    print(f":\n{test_df['label'].value_counts().sort_index()}")
    target_train_size = 50000
    target_test_size = 10000
    if len(train_df) > target_train_size:
        print(f'Subsampling train set to {target_train_size} examples...')
        (train_df, _) = train_test_split(train_df, train_size=target_train_size, random_state=42, stratify=train_df['label'])
    if len(test_df) > target_test_size:
        print(f'Subsampling test set to {target_test_size} examples...')
        (test_df, _) = train_test_split(test_df, train_size=target_test_size, random_state=42, stratify=test_df['label'])
    print(f': {len(train_df)}')
    print(f': {len(test_df)}')
    print(f":\n{train_df['label'].value_counts().sort_index()}")
    print(f":\n{test_df['label'].value_counts().sort_index()}")
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    args.num_labels = 2

    def preprocess_function(examples):
        return {**tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64), 'label': examples['label']}
    print('tokenization...')
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
    print(f'Yelp Review Polarity ')
    print(f': {len(full_train_dataset)}')
    print(f': {len(full_test_dataset)}')
if args.dataset == 'YAHOO_ANS':
    from datasets import load_dataset
    import os
    local_dir = os.path.join(args.data_dir, 'yahoo_answers_topics')
    required = ['train-00000-of-00002.parquet', 'train-00001-of-00002.parquet', 'test-00000-of-00001.parquet']
    if not all((os.path.isfile(os.path.join(local_dir, f)) for f in required)):
        raise FileNotFoundError('Missing Yahoo! Answers parquet files:\n' + local_dir + '\nRequired files: ' + ', '.join(required))
    dataset = load_dataset(local_dir)
    SUBSET_SIZE_TRAIN = 100000
    SUBSET_SIZE_TEST = 10000
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(min(SUBSET_SIZE_TRAIN, len(dataset['train']))))
    dataset['test'] = dataset['test'].shuffle(seed=42).select(range(min(SUBSET_SIZE_TEST, len(dataset['test']))))
    print(f" YAHOO_ANS : {len(dataset['train'])}, {len(dataset['test'])}")
    labels = [int(ex['topic']) for ex in dataset['train']]
    min_label = min(labels)
    max_label = max(labels)
    args.num_labels = max_label - min_label + 1
    print(f':{min_label} ~ {max_label}, num_labels = {args.num_labels}')

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
if args.dataset == 'SICK':
    import pandas as pd
    from datasets import Dataset, DatasetDict
    sick_dir = os.path.join(args.data_dir, 'SICK')
    raw_txt = os.path.join(sick_dir, 'SICK.txt')
    if not os.path.isfile(raw_txt):
        raise FileNotFoundError(f'[SICK] Missing file: {raw_txt}\nPlease place SICK.txt under {sick_dir}.')
    sick_task = getattr(args, 'sick_task', 'sts').lower()
    assert sick_task in ('sts', 'te'), f'--sick_task must be sts or te, got {sick_task}'
    df = pd.read_csv(raw_txt, sep='\t')
    if 'entailment_label' not in df.columns:
        raise KeyError('[SICK] Missing required column: entailment_label')
    train_df = df.iloc[:4500].copy()
    test_df = df.iloc[4500:].copy()

    def _make_ds(dataframe):
        return Dataset.from_dict({'sentence1': dataframe['sentence_A'].astype(str).tolist(), 'sentence2': dataframe['sentence_B'].astype(str).tolist(), 'label': dataframe['relatedness_score'].astype(float).tolist() if sick_task == 'sts' else dataframe['entailment_label'].astype(str).tolist()})
    full_train_dataset = _make_ds(train_df)
    full_test_dataset = _make_ds(test_df)
    if sick_task == 'sts':
        args.num_labels = 1
    else:
        lbl2id = {'ENTAILMENT': 0, 'NEUTRAL': 1, 'CONTRADICTION': 2}
        full_train_dataset = full_train_dataset.map(lambda ex: {'label': lbl2id[ex['label']]}, num_proc=4)
        full_test_dataset = full_test_dataset.map(lambda ex: {'label': lbl2id[ex['label']]}, num_proc=4)
        args.num_labels = 3

    def _preproc(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=args.max_length)
    full_train_dataset = full_train_dataset.map(_preproc, batched=True, remove_columns=['sentence1', 'sentence2'])
    full_test_dataset = full_test_dataset.map(_preproc, batched=True, remove_columns=['sentence1', 'sentence2'])
    full_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    print(f'[SICK] ,={sick_task.upper()},={len(full_train_dataset)},={len(full_test_dataset)}')
if args.dataset == 'YELP_REVIEW_FULL':
    from datasets import load_dataset, DatasetDict
    import os
    dataset_path = os.path.join(args.data_dir, 'yelp_review_full')
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    train_file = os.path.join(train_path, '0000.parquet')
    test_file = os.path.join(test_path, '0000.parquet')
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f'Yelp Review Full \n: {train_file}\n: {test_file}\n')
    dataset = load_dataset('parquet', data_files={'train': train_file, 'test': test_file})
    SUBSET_SIZE_TRAIN = 100000
    SUBSET_SIZE_TEST = 10000
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(min(SUBSET_SIZE_TRAIN, len(dataset['train']))))
    dataset['test'] = dataset['test'].shuffle(seed=42).select(range(min(SUBSET_SIZE_TEST, len(dataset['test']))))
    print(f" YELP_REVIEW_FULL : {len(dataset['train'])}, {len(dataset['test'])}")
    args.num_labels = 5

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
    print(f'Yelp Review Full ')
    print(f': {len(full_train_dataset)}')
    print(f': {len(full_test_dataset)}')
    print(f': {args.num_labels}')
if args.dataset == 'MNLI':
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    dataset_path = f'{args.data_dir}/glue/mnli'
    dataset = load_dataset(dataset_path)

    def preprocess_function(example):
        return tokenizer(example['premise'], example['hypothesis'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    args.num_labels = 3
if args.dataset == 'STS-B':
    dataset_path = f'{args.data_dir}/glue/stsb'
    dataset = load_dataset(dataset_path)

    def preprocess_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    args.num_labels = 1
if args.dataset == 'WNLI':
    dataset_path = f'{args.data_dir}/glue/wnli'
    dataset = load_dataset(dataset_path)

    def preprocess_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    args.num_labels = 2
if args.dataset == 'RTE':
    dataset_path = f'{args.data_dir}/glue/rte'
    dataset = load_dataset(dataset_path)

    def preprocess_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    args.num_labels = 2
if args.dataset == 'SNLI':
    try:
        dataset_path = f'{args.data_dir}/snli'
        dataset = load_dataset(dataset_path)
    except:
        try:
            print('Hugging Face HubSNLI...')
            dataset = load_dataset('snli', cache_dir=args.data_dir)
        except ConnectionError as e:
            print(f': {e}')
            print('')
            exit(1)
    dataset = dataset.filter(lambda x: x['label'] != -1)
    args.num_labels = 3

    def preprocess_function(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
if args.dataset == 'TREC':
    from datasets import load_dataset
    import os
    local_script_dir = os.path.join(args.data_dir, 'CogComp', 'trec')
    trec_py_path = os.path.join(local_script_dir, 'trec.py')
    dataset = load_dataset(path=local_script_dir, cache_dir=local_script_dir, trust_remote_code=True)
    if args.num_labels is None:
        args.num_labels = 6
    if args.num_labels == 6:
        label_key = 'coarse_label'
    elif args.num_labels == 50:
        label_key = 'fine_label'
    else:
        raise ValueError('TREC num_labels must be 6 or 50')

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    def rename_label(example):
        example['label'] = example[label_key]
        return example
    tokenized_dataset = tokenized_dataset.map(rename_label)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
if args.dataset == 'ANLI':
    try:
        dataset_path = f'{args.data_dir}/anli'
        dataset = load_dataset(dataset_path)
    except Exception:
        print(' Hugging Face Hub  ANLI ...')
        dataset = load_dataset('anli', cache_dir=args.data_dir)
    args.num_labels = 3

    def preprocess_function(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    available = set(tokenized_dataset.keys())
    if {'train_r1', 'dev_r1'} <= available:
        train_split = 'train_r1'
        eval_split = 'dev_r1'
    elif {'train', 'validation'} <= available:
        train_split = 'train'
        eval_split = 'validation'
    elif {'train', 'test'} <= available:
        try:
            _ = tokenized_dataset['test'][0]['label']
            train_split = 'train'
            eval_split = 'test'
        except Exception:
            raise ValueError(f"ANLI 'test' split is unavailable. Please use 'validation' or 'dev_r*'.")
    else:
        raise ValueError(f' ANLI ,:{sorted(list(available))}')
    full_train_dataset = tokenized_dataset[train_split]
    full_test_dataset = tokenized_dataset[eval_split]
if args.dataset == 'PAWS':
    try:
        dataset_path = f'{args.data_dir}/paws'
        dataset = load_dataset(dataset_path)
    except:
        try:
            print('Hugging Face HubPAWS...')
            dataset = load_dataset('paws', 'labeled_final', cache_dir=args.data_dir)
        except ConnectionError as e:
            print(f': {e}')
            print('')
            exit(1)
    args.num_labels = 2

    def preprocess_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
if args.dataset == 'AMAZON_POL':
    try:
        dataset_path = f'{args.data_dir}/amazon_polarity'
        dataset = load_dataset(dataset_path)
    except:
        try:
            print('Hugging Face HubAMAZON_POLARITY...')
            dataset = load_dataset('amazon_polarity', cache_dir=args.data_dir)
        except ConnectionError as e:
            print(f': {e}')
            print('')
            exit(1)
    args.num_labels = 2

    def preprocess_function(examples):
        titles = examples.get('title')
        contents = examples.get('content')
        if isinstance(titles, list) and isinstance(contents, list):
            texts = [((t or '') + ' ' + (c or '')).strip() for (t, c) in zip(titles, contents)]
        else:
            t = examples.get('title') or ''
            c = examples.get('content') or ''
            texts = (t + ' ' + c).strip()
        return tokenizer(texts, truncation=True, padding='max_length', max_length=64)
    cols = dataset['train'].column_names
    cols_to_remove = [c for c in cols if c != 'label']
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=cols_to_remove)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
elif args.dataset == 'DBPEDIA14':
    from datasets import load_dataset
    import numpy as np
    dataset = load_dataset('dbpedia_14')
    target_train_size = 50000
    target_test_size = 10000
    print(f'DBPedia14,...')
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
if args.dataset == 'DBPedia':
    try:
        dataset_path = f'{args.data_dir}/dbpedia_14'
        dataset = load_dataset(dataset_path)
    except:
        try:
            print('Hugging Face HubDBPedia...')
            dataset = load_dataset('dbpedia_14', cache_dir=args.data_dir)
        except ConnectionError as e:
            print(f': {e}')
            print('')
            exit(1)
    args.num_labels = 14

    def preprocess_function(examples):
        titles = examples.get('title')
        contents = examples.get('content')
        if isinstance(titles, list) and isinstance(contents, list):
            texts = [((t or '') + ' ' + (c or '')).strip() for (t, c) in zip(titles, contents)]
        else:
            t = examples.get('title') or ''
            c = examples.get('content') or ''
            texts = (t + ' ' + c).strip()
        return tokenizer(texts, truncation=True, padding='max_length', max_length=64)
    cols = dataset['train'].column_names
    cols_to_remove = [c for c in cols if c != 'label']
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=cols_to_remove)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['test']
if args.dataset == 'MRPC':
    dataset_path = f'{args.data_dir}/glue/mrpc'
    dataset = load_dataset(dataset_path)

    def preprocess_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length', max_length=64)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    full_train_dataset = tokenized_dataset['train']
    full_test_dataset = tokenized_dataset['validation']
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    args.num_labels = 2
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
if args.dataset == 'cola':
    dataset_path = f'{args.data_dir}/glue/cola'
    dataset = load_dataset(dataset_path)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=64)
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
        raise ValueError(f': {distribution}')
train_indices = split_data(full_train_dataset, distribution=args.distribution)
test_indices = split_data(full_test_dataset, distribution=args.distribution)

def get_model(r):
    import transformers
    transformers.logging.set_verbosity_error()
    if args.model_name == 'roberta-base':
        print('RoBERTa...')
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    merged_model_path = os.path.join(args.save_dir, 'merged_model.pth')
    if os.path.exists(merged_model_path) and args.algorithm == 'FLORA':
        print('...')
        model.load_state_dict(torch.load(merged_model_path))
    else:
        print('')
    from peft import LoraConfig, get_peft_model, TaskType
    if args.model_name == 'roberta-base':
        lora_config = LoraConfig(r=r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=['query', 'value'], bias='none', task_type=TaskType.SEQ_CLS, modules_to_save=['classifier'])
    model = get_peft_model(model, lora_config)
    if args.algorithm == 'ILORA':
        print('Applying OLoRA initialization')
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
        results = sorted(results, key=lambda x: x[1].metrics['client_id'])
        if args.algorithm == 'FEDIT':
            print('FedIT (Federated Incremental Training)')
            print('FedIT :')
            total_examples = sum([metrics.num_examples for (_, metrics) in results])
            if args.heterogeneous_rank == 'False':
                print('FEDIT:')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'client_weights: {client_weights}')
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'-: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    print(f'- LoRA: {client_rank}')
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
            if args.heterogeneous_rank == 'True':
                print('FEDIT with heterogeneous ranks:')
                max_rank = max(args.heterogeneous_rank_clients)
                print('FEDITmax_rank:', max_rank)
                num_classifier_params = 4
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'client_weights: {client_weights}')
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'-: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    print(f'- LoRA: {client_rank}')
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
        if args.algorithm == 'FLORA':
            print('FLoRA (Federated Low-Rank Adaptation)')
            if args.heterogeneous_rank == 'False':
                print('FLORA:')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'client_weights: {client_weights}')
                aggregated_params = None
                first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
                num_classifier_params = 4
                for (client_idx, (_, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'-: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    print(f'- LoRA: {client_rank}')
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
                                elif i >= len(aggregated_params) - num_classifier_params:
                                    aggregated_params[i] += p_k * param
                            elif len(param.shape) == 1:
                                if i >= len(aggregated_params) - num_classifier_params:
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
            if args.heterogeneous_rank == 'True':
                print('FLORA:')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'client_weights: {client_weights}')
                num_classifier_params = 4
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'-: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    print(f'- LoRA: {client_rank}')
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    lora_params = client_params[:-num_classifier_params]
                    classifier_params = client_params[-num_classifier_params:]
                    if aggregated_params is None:
                        aggregated_params = []
                        param_idx = 0
                        for param in lora_params:
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:
                                    aggregated_params.append(p_k * param)
                                    param_idx += 1
                                elif param.shape[1] == client_rank:
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
                        for param in lora_params:
                            if len(param.shape) == 2:
                                if param.shape[0] == client_rank:
                                    aggregated_params[param_idx] = np.concatenate([aggregated_params[param_idx], p_k * param], axis=0)
                                    param_idx += 1
                                elif param.shape[1] == client_rank:
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
            print('FFA-LORA')
            if args.heterogeneous_rank == 'False':
                print('FFA-LORA:')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'client_weights: {client_weights}')
                num_classifier_params = 4
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_id = fit_res.metrics['client_id']
                    p_k = client_weights[client_id]
                    print(f'\n {client_id} :')
                    print(f'-: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[int(client_id)]
                    print(f'- LoRA: {client_rank}')
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    lora_params = client_params[:-num_classifier_params]
                    classifier_params = client_params[-num_classifier_params:]
                    if aggregated_params is None:
                        aggregated_params = [np.zeros_like(param) for param in client_params]
                        for (i, param) in enumerate(lora_params):
                            if len(param.shape) == 2 and param.shape[0] == client_rank:
                                aggregated_params[i] = param
                    param_idx = 0
                    for param in lora_params:
                        if len(param.shape) == 2:
                            if param.shape[1] == client_rank:
                                aggregated_params[param_idx] += p_k * param
                                param_idx += 1
                            elif param.shape[0] == client_rank:
                                param_idx += 1
                            else:
                                aggregated_params[param_idx] += p_k * param
                                param_idx += 1
                        else:
                            aggregated_params[param_idx] += p_k * param
                            param_idx += 1
                    for (i, param) in enumerate(classifier_params):
                        aggregated_params[param_idx + i] += p_k * param
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
            if args.heterogeneous_rank == 'True':
                print('FFA-LORA:')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'client_weights: {client_weights}')
                max_rank = max(args.heterogeneous_rank_clients)
                print(f': {max_rank}')
                num_classifier_params = 4
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_id = fit_res.metrics['client_id']
                    p_k = client_weights[client_id]
                    print(f'\n {client_id} :')
                    print(f'-: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[int(client_id)]
                    print(f'- LoRA: {client_rank}')
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    lora_params = client_params[:-num_classifier_params]
                    classifier_params = client_params[-num_classifier_params:]
                    filled_lora_params = []
                    for param in lora_params:
                        if len(param.shape) == 2:
                            if param.shape[1] == client_rank:
                                padding = ((0, 0), (0, max_rank - client_rank))
                                filled_param = np.pad(param, padding, 'constant')
                                filled_lora_params.append(filled_param)
                            elif param.shape[0] == client_rank:
                                padding = ((0, max_rank - client_rank), (0, 0))
                                filled_param = np.pad(param, padding, 'constant')
                                filled_lora_params.append(filled_param)
                            else:
                                filled_lora_params.append(param)
                        else:
                            filled_lora_params.append(param)
                    filled_params = filled_lora_params + classifier_params
                    if aggregated_params is None:
                        aggregated_params = [np.zeros_like(p) for p in filled_params]
                        param_idx = 0
                        for param in filled_lora_params:
                            if len(param.shape) == 2 and param.shape[0] == max_rank:
                                aggregated_params[param_idx] = param
                                param_idx += 1
                            else:
                                param_idx += 1
                    param_idx = 0
                    for param in filled_lora_params:
                        if len(param.shape) == 2:
                            if param.shape[1] == max_rank:
                                aggregated_params[param_idx] += p_k * param
                                param_idx += 1
                            elif param.shape[0] == max_rank:
                                param_idx += 1
                            else:
                                aggregated_params[param_idx] += p_k * param
                                param_idx += 1
                        else:
                            aggregated_params[param_idx] += p_k * param
                            param_idx += 1
                    for (i, param) in enumerate(classifier_params):
                        aggregated_params[param_idx + i] += p_k * param
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
        if args.algorithm == 'ILORA':
            if args.heterogeneous_rank == 'False':
                print('ILORA:')
                c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                try:
                    self.c = np.load(c_save_path, allow_pickle=True).tolist()
                    print('c')
                except:
                    print(f'c')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'client_weights: {client_weights}')
                if args.use_control == 'True':
                    print('\n control: delta_cic')
                    delta_c_aggregated = [np.zeros_like(p) for p in self.c]
                    for (client_idx, (_, fit_res)) in enumerate(results):
                        client_idx = fit_res.metrics['client_id']
                        delta_ci_path = os.path.join(args.save_dir, f'client_{client_idx}_delta_ci.npy')
                        try:
                            delta_ci = np.load(delta_ci_path, allow_pickle=True).tolist()
                            for i in range(len(delta_c_aggregated)):
                                delta_c_aggregated[i] += 1 / args.num_clients * delta_ci[i]
                        except Exception as e:
                            print(f' {client_idx}  delta_ci : {str(e)}')
                            continue
                    for i in range(len(self.c)):
                        self.c[i] += delta_c_aggregated[i]
                    c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                    np.save(c_save_path, np.array(self.c, dtype=object))
                    print(f' c  {c_save_path}')
                aggregated_params = None
                for (client_idx, (_, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'-: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    print(f'- LoRA: {client_rank}')
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
                        param_idx = 0
                        for (i, param) in enumerate(client_params):
                            if len(param.shape) == 2:
                                if param.shape[0] == args.lora_r_client:
                                    aggregated_params[param_idx] = np.concatenate([aggregated_params[param_idx], p_k * param], axis=0)
                                    param_idx += 1
                                elif param.shape[1] == args.lora_r_client:
                                    aggregated_params[param_idx] = np.concatenate([aggregated_params[param_idx], param], axis=1)
                                    param_idx += 1
                                else:
                                    param_idx += 1
                            else:
                                param_idx += 1
                num_classifier_params = 4
                classifier_params = [np.zeros_like(param) for param in client_params[-num_classifier_params:]]
                for (client_idx, (_, fit_res)) in enumerate(results):
                    p_k = client_weights[fit_res.metrics['client_id']]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    for i in range(num_classifier_params):
                        classifier_params[i] += p_k * client_params[-(num_classifier_params - i)]
                aggregated_params[-num_classifier_params:] = classifier_params
                num_layers = (len(aggregated_params) - 2) // 4
                print(f'num_layers: {num_layers}')
                for layer_idx in range(num_layers):
                    base_idx = layer_idx * 4
                    q_A = aggregated_params[base_idx]
                    q_B = aggregated_params[base_idx + 1]
                    Q_full = q_B @ q_A
                    if np.isnan(Q_full).any():
                        print(f': Q_fullNaN')
                    elif np.isinf(Q_full).any():
                        print(f': Q_fullinf')
                    (Q_q, R_q) = np.linalg.qr(Q_full, mode='reduced')
                    q_A_prime = R_q[:args.lora_r_client, :]
                    q_B_prime = Q_q[:, :args.lora_r_client]
                    aggregated_params[base_idx] = q_A_prime
                    aggregated_params[base_idx + 1] = q_B_prime
                    v_A = aggregated_params[base_idx + 2]
                    v_B = aggregated_params[base_idx + 3]
                    V_full = v_B @ v_A
                    if np.isnan(V_full).any():
                        print(f': V_fullNaN')
                    elif np.isinf(V_full).any():
                        print(f': V_fullinf')
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
                print('ILORA:')
                num_classifier_params = 4
                if args.use_control == 'True':
                    print('ILORA+control:')
                    c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                    try:
                        self.c = np.load(c_save_path, allow_pickle=True).tolist()
                        print('c')
                    except:
                        print(f'c')
                max_rank = max(args.heterogeneous_rank_clients)
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                print(f'client_weights: {client_weights}')
                if args.use_control == 'True':
                    print('\n control: delta_cic()')
                    delta_c_aggregated = [np.zeros_like(p) for p in self.c]
                    for (client_idx, (_, fit_res)) in enumerate(results):
                        client_idx = fit_res.metrics['client_id']
                        delta_ci_path = os.path.join(args.save_dir, f'client_{client_idx}_delta_ci.npy')
                        try:
                            delta_ci = np.load(delta_ci_path, allow_pickle=True).tolist()
                            print(f'\n  {client_idx} delta_ci :')
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
                            print(f' {client_idx}  delta_ci : {str(e)}')
                            continue
                    for i in range(len(self.c)):
                        self.c[i] += delta_c_aggregated[i]
                    if args.use_control == 'True':
                        print('\n c (control ):')
                        for (i, c) in enumerate(self.c[:3]):
                            print(f'  c[{i}].shape: {c.shape}, : {np.mean(c):.4f}')
                        c_save_path = os.path.join(args.save_dir, 'global_c.npy')
                        np.save(c_save_path, np.array(self.c, dtype=object))
                        print(f' c  {c_save_path}')
                aggregated_params = None
                for (client_idx, (client_proxy, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'-: {p_k:.4f}')
                    client_rank = args.heterogeneous_rank_clients[client_idx]
                    print(f'- LoRA: {client_rank}')
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
                print(f'num_layers: {num_layers}')
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
                    print(f' {target_rank}  {Decomposition_save_paths[rank_idx]}')
                if server_round == args.num_rounds - 1 and max_rank_params is not None:
                    print('\n,...')
                    from flwr.common import ndarrays_to_parameters
                    max_rank_parameters = ndarrays_to_parameters(max_rank_params)
                    params_dict = zip([n for (n, p) in self.global_model.named_parameters() if p.requires_grad], fl.common.parameters_to_ndarrays(max_rank_parameters))
                    state_dict = {k: torch.tensor(v) for (k, v) in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=False)
                    final_model_path = os.path.join(args.save_dir, 'final_global_model.pth')
                    torch.save(self.global_model.state_dict(), final_model_path)
                    print(f' {final_model_path}')
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
                aggregated_metrics = {}
                return (aggregated_parameters, aggregated_metrics)
        if args.algorithm == 'LoRA_FAIR':
            if args.heterogeneous_rank == 'False':
                print(' LoRA_FAIR  ()')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                for (client_idx, (_, fit_res)) in enumerate(results):
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
                num_layers = (len(first_client_params) - 2) // 4
                hidden_size = 768
                avg_A_query = [np.zeros((args.lora_r_client, hidden_size)) for _ in range(num_layers)]
                avg_B_query = [np.zeros((hidden_size, args.lora_r_client)) for _ in range(num_layers)]
                avg_A_value = [np.zeros((args.lora_r_client, hidden_size)) for _ in range(num_layers)]
                avg_B_value = [np.zeros((hidden_size, args.lora_r_client)) for _ in range(num_layers)]
                global_delta_w_query = [np.zeros((hidden_size, hidden_size)) for _ in range(num_layers)]
                global_delta_w_value = [np.zeros((hidden_size, hidden_size)) for _ in range(num_layers)]
                for (client_idx, (_, fit_res)) in enumerate(results):
                    client_idx = fit_res.metrics['client_id']
                    p_k = client_weights[client_idx]
                    print(f'\n {client_idx} :')
                    print(f'-: {p_k:.4f}')
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
                    updated_B_query.append(avg_B_query[layer_idx] + delta_B_query)
                    R_value = global_delta_w_value[layer_idx] - avg_B_value[layer_idx] @ avg_A_value[layer_idx]
                    A = avg_A_value[layer_idx]
                    inv_part = np.linalg.inv(A @ A.T + lambda_reg * I)
                    delta_B_value = R_value @ A.T @ inv_part
                    updated_B_value.append(avg_B_value[layer_idx] + delta_B_value)
                aggregated_params = []
                for layer_idx in range(num_layers):
                    aggregated_params.append(avg_A_query[layer_idx])
                    aggregated_params.append(updated_B_query[layer_idx])
                    aggregated_params.append(avg_A_value[layer_idx])
                    aggregated_params.append(updated_B_value[layer_idx])
                num_classifier_params = 4
                classifier_params = [np.zeros_like(param) for param in client_params[-num_classifier_params:]]
                for (client_idx, (_, fit_res)) in enumerate(results):
                    p_k = client_weights[fit_res.metrics['client_id']]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    for i in range(num_classifier_params):
                        classifier_params[i] += p_k * client_params[-(num_classifier_params - i)]
                aggregated_params.extend(classifier_params)
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_params)
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
            if args.heterogeneous_rank == 'True':
                print(' LoRA_FAIR  ()')
                total_examples = sum([metrics.num_examples for (_, metrics) in results])
                print(f'total_examples: {total_examples}')
                client_weights = {fit_res.metrics['client_id']: fit_res.num_examples / total_examples for (_, fit_res) in results}
                max_rank = max(args.heterogeneous_rank_clients)
                print(f': {max_rank}')
                first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
                num_classifier_params = 4
                num_layers = (len(first_client_params) - num_classifier_params) // 4
                print(f'num_layers: {num_layers}')
                layer_dims = []
                for layer_idx in range(num_layers):
                    base_idx = layer_idx * 4
                    q_A = first_client_params[base_idx]
                    layer_dims.append(q_A.shape[1])
                print(f': {layer_dims}')
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
                    print(f'\n {client_id} (={client_rank}) :')
                    print(f'-: {p_k:.4f}')
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
                classifier_params = [np.zeros_like(param) for param in first_client_params[-num_classifier_params:]]
                for (_, fit_res) in results:
                    p_k = client_weights[fit_res.metrics['client_id']]
                    client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    for i in range(num_classifier_params):
                        classifier_params[i] += p_k * client_params[-(num_classifier_params - i)]
                aggregated_params.extend(classifier_params)
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
from flwr.client import Client, NumPyClient

class CIFAR10Client(NumPyClient):

    def __init__(self, cid, train_indices, test_indices):
        seed_local_for_client(args.seed, cid)
        self.cid = cid
        print('cid=', cid)
        client_rank = args.heterogeneous_rank_clients[int(cid)]
        print(f' {cid} !', f'rank={client_rank}')
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
        if args.algorithm == 'FFA-LORA' and args.heterogeneous_rank == 'False':
            self.model.load_state_dict(state_dict, strict=False)
        if args.algorithm == 'LoRA_FAIR' and args.heterogeneous_rank == 'False':
            self.model.load_state_dict(state_dict, strict=False)
        if args.algorithm == 'FFA-LORA' and args.heterogeneous_rank == 'True':
            print(f'FFA-LORA  -  {self.cid} ')
            params_dict = zip([name for (name, param) in self.model.named_parameters() if param.requires_grad], parameters)
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            print(f' {self.cid} : {client_rank}')
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
            print('')
            self.model.load_state_dict(truncated_state_dict, strict=False)
        if args.algorithm == 'LoRA_FAIR' and args.heterogeneous_rank == 'True':
            print(f'LoRA_FAIR  -  {self.cid} ')
            params_dict = zip([name for (name, param) in self.model.named_parameters() if param.requires_grad], parameters)
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            print(f' {self.cid} : {client_rank}')
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
                    print(f' {self.cid}  {client_rank} Decomposition...')
                    trainable_state = torch.load(param_path)
                    self.model.load_state_dict(trainable_state, strict=False)
                else:
                    print(f':  {client_rank}  {param_path}')
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
        print('fit')
        seed_local_for_client(args.seed, int(self.cid))
        self.set_parameters(parameters)
        if args.algorithm == 'ILORA':
            print('ILORA!')
        elif args.algorithm == 'LoRA_FAIR':
            print('LoRA_FAIR!')
        elif args.algorithm == 'FFA-LORA':
            print('FFA-LORA!')
        elif args.algorithm == 'FEDIT':
            print('FEDIT!')
        else:
            print('!')
            client_rank = args.heterogeneous_rank_clients[int(self.cid)]
            print('fitclient_rank:', client_rank)
            self.model = get_model(client_rank)
        initial_params = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
        if args.use_control == 'True':
            ci_path = os.path.join(args.save_dir, f'client_{self.cid}_ci.npy')
            if os.path.exists(ci_path):
                self.ci = np.load(ci_path, allow_pickle=True).tolist()
                print(f' {self.cid}  ci ')
            c_save_path = os.path.join(args.save_dir, 'global_c.npy')
            try:
                c = np.load(c_save_path, allow_pickle=True).tolist()
                print(f' {self.cid}  c')
                client_rank = args.heterogeneous_rank_clients[int(self.cid)]
                max_rank = max(args.heterogeneous_rank_clients)
                truncated_c = []
                for (param_idx, param) in enumerate(c):
                    if len(param.shape) == 2:
                        if param.shape[0] == max_rank and param.shape[0] > client_rank:
                            truncated_param = param[:client_rank, :]
                            print(f' {param_idx}: {param.shape} -> {truncated_param.shape} ()')
                        elif param.shape[1] == max_rank and param.shape[1] > client_rank:
                            truncated_param = param[:, :client_rank]
                            print(f' {param_idx}: {param.shape} -> {truncated_param.shape} ()')
                        else:
                            truncated_param = param
                        truncated_c.append(truncated_param)
                    else:
                        truncated_c.append(param)
                c = truncated_c
                print(f' {self.cid} (={client_rank}) c')
            except:
                c = self.ci
                print(f' {self.cid}  c, ci')
            if len(c) != len(self.ci):
                print(':  c  ci , ci')
                c = self.ci
        import copy
        if args.use_control == 'True':
            original_ci = copy.deepcopy(self.ci)
            control_option = 1
            if control_option == 1:
                print('control Option I: ')
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
            print(f' {self.cid} delta_ci {delta_ci_path}')
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
    if args.training_mode == 'Centralized':
        model = get_model(args.lora_r_client)
        print('...')
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
            print(f': {file_path} - {str(e)}')
    for file_path in glob.glob(os.path.join(args.save_dir, '*.npy')):
        try:
            os.remove(file_path)
            print('')
        except Exception as e:
            print(f'npy: {file_path} - {str(e)}')
