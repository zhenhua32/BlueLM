# Copyright 2023 vivo.
#
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from utils.sampler import DistributedSampler
from datasets import load_dataset


def build_loader(tokenizer, args, logger=None):
    """
    加载数据
    """
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

    # 输入列和输出列
    prompt_column = args.prompt_column
    response_column = args.response_column

    def preprocess_data_function(examples):
        """处理数据"""
        # 总长度是输入的长度加上输出的长度
        max_seq_length = args.max_source_length + args.max_target_length
        all_input_ids = []
        all_labels = []
        for i in tqdm(range(len(examples[prompt_column]))):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                # 转换成 ids
                a_ids = tokenizer.encode(text=query, truncation=True,
                                         max_length=args.max_source_length - 1)
                b_ids = tokenizer.encode(text=answer, truncation=True,
                                         max_length=args.max_target_length - 1)
                context_length = len(a_ids)
                # 在最前面加上 bos，最后面加上 eos
                input_ids = [tokenizer.bos_token_id] + a_ids + b_ids + [tokenizer.eos_token_id]
                labels = [tokenizer.bos_token_id] + a_ids + b_ids + [tokenizer.eos_token_id]
                if args.finetune:
                    # 如果是微调, 需要更新标签.
                    labels = [-100] * (context_length + 1) + b_ids + [tokenizer.eos_token_id]
                # 计算需要填充的长度, 然后在右侧填充 pad_id. 并且更新 labels
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

                all_input_ids.append(input_ids)
                all_labels.append(labels)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_labels = torch.tensor(all_labels, dtype=torch.long)

        return TensorDataset(all_input_ids, all_labels)

    # 获取训练数据集
    train_dataset = raw_datasets["train"]
    dataset = preprocess_data_function(train_dataset)

    sampler = DistributedSampler(dataset, shuffle=True, batch_size=args.batch_size_per_device)

    data_loader = DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size_per_device,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    return data_loader
