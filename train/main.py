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

import os
import torch
import json

import torch.distributed as dist
import deepspeed

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import FusedAdam
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from utils.dataloader import build_loader
from utils.ds_utils import get_train_ds_config
from utils.utils import (
    get_optimizer_grouped_parameters,
    to_device,
    to_half,
    to_bf16,
    init_distributed_mode,
    parse_args,
    create_logger,
    AmpScaler,
)
from utils.finetune_peft import get_peft_lora_model


def train(args, model, train_dataloader, optimizer: torch.optim.Optimizer, lr_scheduler, log_writer=None, logger=None):
    """
    训练流程
    """
    is_main_process = dist.get_rank() == 0
    train_progress_bar = tqdm(range(args.max_steps), disable=not is_main_process)
    global_step = 0
    if not args.deepspeed:
        loss_scaler = AmpScaler()
    for epoch in range(int(args.epochs)):
        model.train()

        for step, batch in enumerate(train_dataloader):
            if global_step <= args.max_steps:
                # 获取输入数据
                input_ids, labels = batch
                input_ids = to_device(input_ids)
                labels = to_device(labels)
                if args.deepspeed:
                    if model.fp16_enabled():
                        input_ids = to_half(input_ids)
                        labels = to_half(labels)
                    elif model.bfloat16_enabled():
                        input_ids = to_bf16(input_ids)
                        labels = to_bf16(labels)
                    # 模型前进一步
                    output = model(input_ids=input_ids, labels=labels)
                else:
                    with torch.cuda.amp.autocast():
                        output = model(input_ids=input_ids, labels=labels)

                # 计算 loss
                loss = output["loss"]
                loss /= args.gradient_accumulation_steps

                if args.deepspeed:
                    # 反向传播
                    model.backward(loss)
                    if global_step != 0 and global_step % args.gradient_accumulation_steps == 0:
                        model.step()
                else:
                    loss_scaler(
                        loss, optimizer, lr_scheduler, update_grad=(global_step % args.gradient_accumulation_steps == 0)
                    )

                # 打印当前步的信息
                logger.info(f"epoch: {epoch + 1}, current_step: {global_step}, loss: {loss.item():.4f}")
                global_step += 1
                train_progress_bar.update(1)

                # 记录日志
                if log_writer is not None:
                    log_writer.add_scalar(f"train/loss", loss.item(), global_step)
                    log_writer.flush()

                # 保存模型
                if global_step % args.save_steps == 0 and is_main_process:
                    logger.info("Start saving checkpoint")
                    path = os.path.join(args.output_dir, "epoch_{}_step_{}".format(epoch + 1, global_step))
                    model.module.save_pretrained(path)
                    logger.info("Finish saving checkpoint")

    # 保存最终的模型
    if is_main_process:
        logger.info("Start saving checkpoint")
        path = os.path.join(args.output_dir, "final")
        model.module.save_pretrained(path)
        logger.info("Finish saving checkpoint")
    dist.barrier()


def main():
    """
    整体训练的流程
    """
    # 解析参数
    args = parse_args()
    init_distributed_mode(args)

    # initialize logger
    logger = create_logger(name="Bluelm", rank=dist.get_rank())
    logger.info(f"Training arguments: {args}")

    # Set seed before initializing model.
    set_seed(args.seed)

    # 更新参数中的 seq_len
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config.seq_len = args.seq_len

    # initialize tokenizer and model
    logger.info("Start initializing tokenizer and model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, config=config, trust_remote_code=True, torch_dtype=torch.float16
    )
    logger.info("Finish initializing tokenizer and model ...")

    # enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 加载 lora 模型
    if args.lora_rank is not None:
        model = get_peft_lora_model(model, args)
        logger.info("Load model in lora mode")

    if dist.get_rank() == 0 and args.tensorboard_dir:
        log_writer = SummaryWriter(log_dir=args.tensorboard_dir)
    else:
        log_writer = None

    # build data loader 初始化数据加载器
    train_dataloader = build_loader(tokenizer, args, logger=logger)
    logger.info("Finish Load train_dataloader")

    if hasattr(args, "max_steps") and args.max_steps > 0:
        args.start_epoch = 0
        args.epochs = args.max_steps // len(train_dataloader) + int(args.max_steps % len(train_dataloader) > 0)
    else:
        args.start_step = 0
        args.max_steps = args.epochs * len(train_dataloader)

    # 初始化模型, 优化器, 学习率调度器
    if args.deepspeed:
        if hasattr(args, "deepspeed_config") and args.deepspeed_config != None:
            ds_config = json.load(open(args.deepspeed_config, "r", encoding="utf-8"))
        else:
            ds_config = get_train_ds_config()
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size_per_device
        ds_config["train_batch_size"] = args.batch_size_per_device * args.world_size * args.gradient_accumulation_steps
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
        lr_scheduler = CosineAnnealingLR(optimizer, args.max_steps)
        logger.info("Start initializing deepspeed")
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )
        logger.info("Finish initializing deepspeed")
    else:
        # 先看下不使用 deepspeed 的情况. 不用 deepspeed 根本跑不起来, 显存占用太多了
        model.cuda()
        logger.info(f"model dtype: {model.dtype}, deivce: {model.device}")
        parameters = model.parameters()
        optimizer = AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.95))
        lr_scheduler = CosineAnnealingLR(optimizer, args.max_steps)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], find_unused_parameters=True
            )
            model._set_static_graph()

    logger.info("Finish Load model and optimizer")

    # 开始训练
    train(args, model, train_dataloader, optimizer, lr_scheduler, log_writer=log_writer, logger=logger)


if __name__ == "__main__":
    main()
