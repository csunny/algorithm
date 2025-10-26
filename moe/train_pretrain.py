import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time 
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model_config import MoeConfig, MoeForCausalLM
from lm_dataset import PretrainDataset


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)
        sys.stdout.flush()

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)    
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (step % args.log_interval == 0):
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}), loss:{:.3f}, lr:{:.12f} epoch_time:{} min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[0]['lr'],
                    spend_time / (step + 1) * iter_per_epoch / 60 - spend_time // 60
                )) 
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "train/loss": loss.item() * args.accumulation_steps,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch + step / iter_per_epoch
                }, step=epoch * iter_per_epoch + step)

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
          
            model.eval()
            moe_path = '_moe' if llm_config.use_moe else ''
            ckpt_path = f'{args.save_dir}/moe_pretrain_{llm_config.hidden_size}{moe_path}.pt'

            if isinstance(model, DDP):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckpt_path)
            model.train()


def init_model(llm_config):
    model = MoeForCausalLM(llm_config).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(".")

    Logger(f'LLM Model Params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M')
    return model, tokenizer

def init_destributed_mode():  # 修复函数名 distributed
    if not ddp:
        return
    global ddp_local_rank, DEVICE  # 修复变量名 rank

    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.getenv("RANK", "0"))
    ddp_local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 修复变量名
    DEVICE = f'cuda:{ddp_local_rank}'  # 修复变量名
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Mixture of Experts Model")
    parser.add_argument("--out_dir", type=str, default="../out")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="moe_pretrain")
    parser.add_argument("--wandb_run_name", type=str, default="moe_pretrain_run")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--hidden_size", type=int, default=512)

    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--use_moe", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default="../datasets/moe_dataset/pretrain_hq.jsonl")

    args = parser.parse_args()
    llm_config = MoeConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                           use_moe=args.use_moe)

    args.save_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"""
        moe-pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}-UseMoE-{args.use_moe}
    """ 

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda', dtype=getattr(torch, args.dtype))  # 修复 autocast 警告

    ddp = int(os.getenv("RANK", "-1")) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"  # 修复变量名

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_destributed_mode()  # 移除参数
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):  # 修复变量名
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    else:
        wandb = None

    model, tokenizer = init_model(llm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)  # 修复参数名
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])  # 修复变量名

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)