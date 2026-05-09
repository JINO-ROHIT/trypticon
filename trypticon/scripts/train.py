#taken and modified from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py

import torch
import torch.nn as nn
import tiktoken
from tqdm.auto import tqdm
import wandb

from trypticon.model.gpt2 import GPTModel
from trypticon.scripts.data_loader import create_dataloader_v1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device} for training")

def calc_loss_batch(input_batch, target_batch, model):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model)
        val_loss = calc_loss_loader(val_loader, model)
    model.train()
    return train_loss, val_loss

def train(model, train_loader, val_loader, optimizer, num_epochs, eval_freq):

    train_losses, val_losses = [], []
    global_step = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()

        for (input_batch, target_batch) in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()

            global_step +=1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"epoch: {epoch} global step: {global_step} train loss: {train_loss} val loss: {val_loss}")

                wandb.log({"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "val_loss": val_loss})
    
    return train_losses, val_losses


def main(gpt_config, settings, wandb_run):

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        entity="jinooo",
        project="trypticon",
        config={"gpt_config": gpt_config, "settings": settings},
        name=wandb_run
    )

    with open("data/test.txt", "r", encoding="utf-8") as file:
            text_data = file.read()

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )


    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )


    train_losses, val_losses = train(
        model, train_loader, val_loader, optimizer,
        num_epochs=settings["num_epochs"], eval_freq=5
    )

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

    train_losses, val_losses, model = main(GPT_CONFIG_124M, OTHER_SETTINGS, "single_gpu_run")
    wandb.finish()
