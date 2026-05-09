#taken and modified from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py

#torchrun --nproc_per_node=2 trypticon/scripts/data_parallel.py 
#torchrun --nproc_per_node=2 trypticon/scripts/data_parallel.py --wandb

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


def all_reduce_gradients(model, world_size, debug=False):
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

def train(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, device, rank, world_size, debug=False, wandb=False):

    train_losses, val_losses = [], []
    global_step = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()

        for (input_batch, target_batch) in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            if debug:
                print(f"[Rank {rank}] loss: {loss.item():.10f}")
            loss.backward()

            all_reduce_gradients(model, world_size, debug=debug)

            optimizer.step()

            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if rank == 0:
                    print(f"epoch: {epoch} global step: {global_step} train loss: {train_loss} val loss: {val_loss}")
                    if wandb:
                        wandb.log({"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "val_loss": val_loss})
    
    return train_losses, val_losses


def main(gpt_config, settings, wandb_run):

    torch.manual_seed(42)
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}/{world_size} using device {device}")

    use_wandb = settings.get("wandb", False)

    if rank == 0 and use_wandb:
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
        model.parameters(), 
        lr=settings["learning_rate"], 
        weight_decay=settings["weight_decay"]
    )

    train_losses, val_losses = train(
        model, 
        train_loader, 
        val_loader, 
        optimizer,
        num_epochs=settings["num_epochs"], 
        eval_freq=5, 
        device=device, 
        rank=rank, 
        world_size=world_size,
        debug=settings.get("debug", False),
        wandb=use_wandb
    )

    if rank == 0 and use_wandb:
        wandb.finish()

    dist.destroy_process_group()

    return train_losses, val_losses, model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="trypticon data parallelism")

    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--qkv_bias", type=bool, default=False)

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--run_name", type=str, default="ddp_run")
    parser.add_argument("--wandb", type=bool, default=False)

    args = parser.parse_args()

    gpt_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias
    }

    settings = {
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "eval_freq": args.eval_freq,
        "debug": args.debug,
        "wandb": args.wandb
    }

    return gpt_config, settings, args.run_name


if __name__ == "__main__":
    gpt_config, settings, run_name = parse_args()
    train_losses, val_losses, model = main(gpt_config, settings, run_name)
