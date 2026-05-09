#taken and modified from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py

#torchrun --nproc_per_node=1 trypticon/scripts/data_parallel.py 

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
import wandb

from trypticon.model.gpt2 import GPTModel
from trypticon.scripts.data_loader import Partition


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def all_reduce(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_gradients(model, world_size):
    for param in model.parameters():
        if param.grad is not None:
            all_reduce(param.grad)
            param.grad.div_(world_size)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss

def train(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, device, rank, world_size):

    train_losses, val_losses = [], []
    global_step = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()

        for (input_batch, target_batch) in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            all_reduce_gradients(model, world_size)

            optimizer.step()

            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if rank == 0:
                    print(f"epoch: {epoch} global step: {global_step} train loss: {train_loss} val loss: {val_loss}")
                    wandb.log({"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "val_loss": val_loss})
    
    return train_losses, val_losses


def main(gpt_config, settings, wandb_run):

    torch.manual_seed(42)
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}/{world_size} using device {device}")

    if rank == 0:
        wandb.init(
            entity="jinooo",
            project="trypticon",
            config={"gpt_config": gpt_config, "settings": settings},
            name=wandb_run
        )

    with open("data/test.txt", "r", encoding="utf-8") as file:
            text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    partition = Partition(
        data=text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        shuffle=True,
        drop_last=True,
        world_size=world_size
    )
    partition.create_loader()
    train_loader = partition.get_partition(rank)

    val_partition = Partition(
        data=text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        shuffle=False,
        drop_last=False,
        world_size=world_size
    )
    val_partition.create_loader()
    val_loader = val_partition.get_partition(rank)

    model = GPTModel(gpt_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    train_losses, val_losses = train(
        model, train_loader, val_loader, optimizer,
        num_epochs=settings["num_epochs"], eval_freq=5, device=device, rank=rank, world_size=world_size
    )

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()

    return train_losses, val_losses, model


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,   
        "context_length": 256,  
        "emb_dim": 768,         
        "n_heads": 12,        
        "n_layers": 12,         
        "drop_rate": 0.1,      
        "qkv_bias": False      
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    train_losses, val_losses, model = main(GPT_CONFIG_124M, OTHER_SETTINGS, "ddp_run")
