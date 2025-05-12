import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import evaluate
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="BERT多任务微调脚本")
    parser.add_argument('--mode', '-m',choices=['train', 'test'], required=True, help='train or test')
    parser.add_argument('--epoch','-e', type=int, default=None, help='checkpoint{epoch}to load')
    parser.add_argument('--epochs', type=int, default=15, help='epochs to train')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--accumulation', type=int, default=4, help='gradient accumulation step')
    return parser.parse_args()

def get_datasets(tokenizer):
    mnli = load_dataset("glue", "mnli")
    sst2 = load_dataset("glue", "sst2")
    qqp = load_dataset("glue", "qqp")
    task_to_id = {"mnli": 0, "qqp": 1, "sst2": 2}

    def tokenize_sst2(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True
        )
    def tokenize_mnli(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True
        )
    def tokenize_qqp(examples):
        return tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True
        )
    def preprocess_function(examples, task_name):
        if task_name == "sst2":
            tokenized_inputs = tokenize_sst2(examples)
            tokenized_inputs["labels"] = examples["label"]
        elif task_name == "mnli":
            tokenized_inputs = tokenize_mnli(examples)
            tokenized_inputs["labels"] = examples["label"]
        elif task_name == "qqp":
            tokenized_inputs = tokenize_qqp(examples)
            tokenized_inputs["labels"] = examples["label"]
        else:
            raise ValueError("Unknown task name")
        tokenized_inputs["task_id"] = [task_to_id[task_name]] * len(tokenized_inputs["input_ids"])
        tokenized_inputs["task_name"] = [task_name] * len(tokenized_inputs["input_ids"])
        return tokenized_inputs

    processed_sst2 = sst2.map(lambda examples: preprocess_function(examples, "sst2"), batched=True)
    processed_mnli = mnli.map(lambda examples: preprocess_function(examples, "mnli"), batched=True)
    processed_qqp = qqp.map(lambda examples: preprocess_function(examples, "qqp"), batched=True)
    return processed_sst2, processed_mnli, processed_qqp

class MultiTaskBert(nn.Module):
    def __init__(self, bert_model_name, num_labels_dict):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.task_heads = nn.ModuleDict()
        for task_name, num_labels in num_labels_dict.items():
            self.task_heads[task_name] = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, token_type_ids, task_name):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        logits = self.task_heads[task_name](cls_embedding)
        return logits

def get_loaders(processed_sst2, processed_mnli, processed_qqp, batch_size):
    sst2_loader = DataLoader(processed_sst2["train"], batch_size=batch_size, shuffle=True)
    mnli_loader = DataLoader(processed_mnli["train"], batch_size=batch_size, shuffle=True)
    qqp_loader = DataLoader(processed_qqp["train"], batch_size=batch_size, shuffle=True)
    sst2_eval = DataLoader(processed_sst2["validation"], batch_size=batch_size)
    mnli_eval = DataLoader(processed_mnli["validation_matched"], batch_size=batch_size)
    qqp_eval = DataLoader(processed_qqp["validation"], batch_size=batch_size)
    return {
        "sst2": sst2_loader,
        "mnli": mnli_loader,
        "qqp": qqp_loader
    }, {
        "sst2": sst2_eval,
        "mnli": mnli_eval,
        "qqp": qqp_eval
    }

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", use_fast=True)
    processed_sst2, processed_mnli, processed_qqp = get_datasets(tokenizer)
    loaders, _ = get_loaders(processed_sst2, processed_mnli, processed_qqp)
    model = MultiTaskBert("bert-large-uncased", {"mnli": 3, "qqp": 2, "sst2": 2}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_loss = float("inf")
    if args.epoch is not None:
        checkpoint = torch.load('../../model/MultiTaskBert/checkpoint_{epoch}.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = args.epoch
        best_loss = checkpoint['loss']
        print(f"resume training: from epoch {start_epoch}，loss={best_loss:.4f}")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        total_batches = 0
        for task_name, loader in loaders.items():
            loop = tqdm(loader, leave=False, desc=f"Epoch {epoch+1}/{args.epochs} - {task_name}")
            for step, batch in enumerate(loop):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    batch["task_name"][0]
                )
                loss = criterion(logits, labels)
                loss = loss / args.accumulation_steps
                loss.backward()
                if (step + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                total_batches += 1
        epoch_loss = running_loss / total_batches
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint,'../../model/MultiTaskBert/checkpoint_{epoch}.pth')
        print(f"Epoch {epoch+1}/{args.epochs} Average training loss: {epoch_loss:.4f}")

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", use_fast=True)
    processed_sst2, processed_mnli, processed_qqp = get_datasets(tokenizer)
    _, evalers = get_loaders(processed_sst2, processed_mnli, processed_qqp)
    model = MultiTaskBert("bert-large-uncased", {"mnli": 3, "qqp": 2, "sst2": 2}).to(device)
    assert args.epoch is not None, "you have to specify checkpoint to load"
    checkpoint = torch.load('../../model/MultiTaskBert/checkpoint_{epoch}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for name, evaler in evalers.items():
        metric = evaluate.load("glue", name)
        print(f"evaluating {name} dataset")
        for batch in tqdm(evaler):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                logits = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    batch["task_name"][0]
                )
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
        result = metric.compute()
        print(f"{name} evaluation result: {result}")

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
