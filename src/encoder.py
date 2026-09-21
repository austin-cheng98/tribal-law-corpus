"""Minimal fine-tuning loop for a pretrained encoder, shared by the probe and the
classification experiments. Plain PyTorch so the training regime is explicit."""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.environ.get("ENC_MODEL", "nlpaueb/legal-bert-base-uncased")
MAXLEN, BATCH, EPOCHS, LR = 256, 16, 3, 3e-5


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _encode(tok, texts):
    e = tok(list(texts), truncation=True, max_length=MAXLEN, padding="max_length",
            return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


def fit_predict(train_texts, train_y, test_texts, labels, seed=0, model=MODEL,
                epochs=EPOCHS, lr=LR):
    """Fine-tune on the training fold and return predicted labels for the test fold."""
    torch.manual_seed(seed)
    dev = device()
    lab2i = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=len(labels)).to(dev)
    ids, am = _encode(tok, train_texts)
    y = torch.tensor([lab2i[v] for v in train_y])
    dl = DataLoader(TensorDataset(ids, am, y), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * len(dl)), pct_start=0.1)
    net.train()
    for _ in range(epochs):
        for bi, bm, by in dl:
            opt.zero_grad()
            out = net(input_ids=bi.to(dev), attention_mask=bm.to(dev), labels=by.to(dev))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()

    net.eval()
    tids, tam = _encode(tok, test_texts)
    pred = []
    with torch.no_grad():
        for i in range(0, len(tids), 64):
            lo = net(input_ids=tids[i:i + 64].to(dev),
                     attention_mask=tam[i:i + 64].to(dev)).logits
            pred.append(lo.argmax(-1).cpu().numpy())
    del net
    if dev == "mps":
        torch.mps.empty_cache()
    return np.array([labels[i] for i in np.concatenate(pred)], dtype=object)
