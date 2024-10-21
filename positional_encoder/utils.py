import torch
from torch import Tensor, nn
from typing import Callable, Tuple, Type
import numpy as np

import os
from tempfile import TemporaryDirectory

from datasets import load_dataset
from transformers import GPT2Tokenizer

import time

import math
import matplotlib.pyplot as plt


# Data processing functions
def tokenize_and_encode(data, tokenizer):
    return [tokenizer.encode(entry["text"]) for entry in data]


def data_process(encoded_data: list) -> Tensor:
    """Converts list of tokenized data into a flat Tensor."""
    return torch.cat([torch.tensor(x, dtype=torch.long) for x in encoded_data if x])


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into `bsz` separate sequences, removing extra elements."""
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    return data.view(bsz, seq_len).t().contiguous()


# Main function to get data
def get_data(
    batch_size: int, dataset_name: str = "wikitext-2-v1"
) -> Tuple[Tensor, Tensor, Tensor]:
    dataset = load_dataset("wikitext", dataset_name)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_data = tokenize_and_encode(dataset["train"], tokenizer)
    val_data = tokenize_and_encode(dataset["validation"], tokenizer)
    test_data = tokenize_and_encode(dataset["test"], tokenizer)

    train_tensor = data_process(train_data)
    val_tensor = data_process(val_data)
    test_tensor = data_process(test_data)

    train_tensor = batchify(train_tensor, batch_size)
    val_tensor = batchify(val_tensor, batch_size)
    test_tensor = batchify(test_tensor, batch_size)

    return train_tensor, val_tensor, test_tensor, tokenizer


def get_batch(
    source: Tensor, i: int, device: str, max_seq_len: int = 100
) -> Tuple[Tensor, Tensor]:
    """
    Arguments:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int
        max_seq_len: int
        device: str

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(max_seq_len, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data.to(device), target.to(device)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


def train_epoch(
    model: Type[nn.Module],
    train_data: Tensor,
    criterion: Callable,
    optimizer,
    scheduler,
    n_tokens: int,
    epoch: int,
    device: str,
    max_seq_len: int = 100,
    verbose: bool = True,
    scheduler_step_every_batch: bool = False,
    log_interval: int = 200,
) -> float:
    """
    Arguments:
        model: nn.Module
        train_data: Tensor, shape ``[full_seq_len, batch_size]``
        max_seq_len: int
        criterion: Callable, returns the loss function
        optimizer:
        scheduler:
        n_tokens: int
        epoch: int
        device: str
        verbose: bool
        scheduler_step_every_batch: bool
        log_interval: int
    """

    model.train()  # turn on train mode
    total_loss = 0.0
    epoch_loss = 0.0
    start_time = time.time() if verbose else None
    src_mask = generate_square_subsequent_mask(max_seq_len).to(device)

    num_batches = len(train_data) // max_seq_len
    for batch, i in enumerate(range(0, train_data.size(0) - 1, max_seq_len)):
        data, targets = get_batch(train_data, i, device)
        seq_len = data.size(0)
        if seq_len != max_seq_len:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        output_flat = output.view(-1, n_tokens)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        cur_loss = loss.item()
        epoch_loss += cur_loss * seq_len
        total_loss += cur_loss

        if verbose and batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(
                f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | "
                f"lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
            )
            total_loss = 0
            start_time = time.time()

        if scheduler_step_every_batch:
            scheduler.step()
    if not scheduler_step_every_batch:
        scheduler.step()
    return epoch_loss / (len(train_data) - 1)


def evaluate(
    model: Type[nn.Module],
    eval_data: Tensor,
    n_tokens: int,
    criterion: Callable,
    device: str,
    max_seq_len: int = 100,
) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.0
    src_mask = generate_square_subsequent_mask(max_seq_len).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, max_seq_len):
            data, targets = get_batch(eval_data, i, device)
            seq_len = data.size(0)
            if seq_len != max_seq_len:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, n_tokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def train(
    model: Type[nn.Module],
    train_data: Tensor,
    val_data: Tensor,
    test_data: Tensor,
    n_tokens: int,
    n_epochs: int,
    criterion: Callable,
    device: str,
    optimizer=None,
    scheduler=None,
    verbose: bool = True,
    scheduler_step_every_batch: bool = False,
    log_interval: int = 100,
) -> Tuple[list, list, float]:
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=5.0) if optimizer is None else optimizer
    )
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        if scheduler is None
        else scheduler
    )
    train_loss_hist, val_loss_hist = [], []
    best_val_loss = float("inf")

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.time()
            train_loss = train_epoch(
                model,
                train_data,
                criterion,
                optimizer,
                scheduler,
                n_tokens,
                epoch,
                device,
                verbose=verbose,
                scheduler_step_every_batch=scheduler_step_every_batch,
                log_interval=log_interval,
            )
            val_loss = evaluate(model, val_data, n_tokens, criterion, device)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            if verbose:
                print("-" * 92)
                print(
                    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}"
                )
                print("-" * 92)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
        model.load_state_dict(
            torch.load(best_model_params_path)
        )  # load best model states

    test_loss = evaluate(model, test_data, n_tokens, criterion, device)
    test_ppl = math.exp(test_loss)
    if verbose:
        print("=" * 92)
        print(
            f"| End of training | test loss {test_loss:5.2f} | "
            f"test ppl {test_ppl:8.2f}"
        )
        print("=" * 92)

    return train_loss_hist, val_loss_hist, test_loss


def plot_losses(losses: dict, title: str = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    for label in losses:
        if isinstance(losses[label], tuple):
            train_loss, val_loss = losses[label]
            x = np.linspace(1, len(train_loss), len(train_loss))
            p = ax.plot(x, train_loss, label=f"{label} train", ls="--")
            ax.plot(x, val_loss, label=f"{label} val", color=p[0].get_color())
        else:
            loss = losses[label]
            x = np.linspace(1, len(loss), len(loss))
            ax.plot(x, loss, label=label)
    plt.legend()
    plt.savefig(f"part_2_plots/p2_{title.lower()}.png".replace(" ", "_"), dpi=300)
