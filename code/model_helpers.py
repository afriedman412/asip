"""
This is mostly from 'asip_script_pipeline_NNNN.ipynb'

Or copied/migrated from elsewhere into there.

Wrote a lot of the same code a few times!

Only real metric that might affect anything is TOK_MAX_LEN,
I think I was careful about setting that earlier to be more efficient.
"""
from config import (EMOTION_MODELS, TOXICITY_MODEL)
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from functools import partial
import torch
import pandas as pd

device = 0 if torch.cuda.is_available() else -1
TOK_MAX_LEN = 512
SENT_MAX_SEQ_LEN = 128


SENT_EMB = SentenceTransformer("BAAI/bge-base-en-v1.5")

SENT_EMB.max_seq_length = SENT_MAX_SEQ_LEN
TOK_TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")
TOK_MODEL = AutoModel.from_pretrained("roberta-base")


def flatten_scores(row):
    return {d['label']: d['score'] for d in row}


def scores_to_df(model_output):
    raw_ = model_output.to_pandas()['raw']
    return pd.DataFrame(raw_.apply(flatten_scores).tolist())

# probably the most relevant


def make_pipeline(hf_model: str, task: str):
    """
    Made this during consolidation

    Probably works better than anything else in here
    """
    device = 0 if torch.cuda.is_available() else -1
    pipe_params = {
        'task': task,
        'model': hf_model,
        'truncation': True,
        'max_length': TOK_MAX_LEN,
        'device': device
    }

    if task == "text-classification":
        pipe_params['topK'] = None
        pipe_params['padding'] = True

    pipe_ = pipeline(**pipe_params)
    return pipe_


def make_classifier(hf_model: str):
    """Create a text-classification pipeline that returns all label scores."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=hf_model,
        truncation=True,
        max_length=150,
        padding=True,
        top_k=None,                # return all labels
        device=device
    )


def run_pipeline_raw(batch, classifier, **kwargs):
    """Run classifier on a batch of texts, returning label-score dicts."""
    texts = batch["text"]
    res = classifier(texts, **kwargs)
    if not isinstance(res, list):
        res = [res]
    # Normalize each sample: list[ list[{"label":, "score":}, ...] ]
    out = []
    for r in res:
        if isinstance(r, list):
            scores = {d["label"]: float(d["score"]) for d in r}
        else:
            scores = {r["label"]: float(r["score"])}
        out.append(scores)
    return {"scores": out}


def full_model_run(hf_model: str, dataset_):
    """Run one HF model and return dataset with .map() results."""
    cl = make_classifier(hf_model)
    c_raw = partial(run_pipeline_raw, classifier=cl)
    output = dataset_.map(c_raw, batched=True, batch_size=16)
    return output


def process_lines_all(batch):
    """Run toxicity + all emotion models in parallel on a batch of lines."""
    texts = batch["text"]
    results = {}

    # Instantiate pipelines once (outside map)
    if not hasattr(process_lines_all, "_pipes"):
        device = 0 if torch.cuda.is_available() else -1
        process_lines_all._pipes = {
            "toxicity": pipeline(
                "text-classification",
                model=TOXICITY_MODEL,
                truncation=True,
                max_length=150,
                padding=True,
                top_k=None,
                device=device,
            ),
            **{
                f"emo_{i}": make_classifier(m)
                for i, m in enumerate(EMOTION_MODELS)
            },
        }

    pipes = process_lines_all._pipes

    def run_pipe(name, pipe):
        res = pipe(texts)
        if not isinstance(res, list):
            res = [res]
        out = []
        for r in res:
            # handle both single dict and list-of-dicts
            if isinstance(r, list):
                out.append({d["label"]: float(d["score"]) for d in r})
            elif isinstance(r, dict) and "label" in r:
                out.append({r["label"]: float(r.get("score", 1.0))})
            else:
                out.append({})
        return name, out

    # Parallelize over models (each runs GPU sequentially but thread-safe)
    with ThreadPoolExecutor(max_workers=len(pipes)) as ex:
        futures = [ex.submit(run_pipe, name, pipe)
                   for name, pipe in pipes.items()]
        for fut in as_completed(futures):
            name, scores = fut.result()
            results[name] = scores

    return results


# for emeddings
def masked_mean_std(last_hidden_state, attention_mask):
    # last_hidden_state: (B, T, H); attention_mask: (B, T)
    mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
    denom = mask.sum(1).clamp(min=1)     # (B, 1)
    mean = (last_hidden_state * mask).sum(1) / denom  # (B, H)

    # masked variance = E[x^2] - (E[x])^2
    ex2 = (last_hidden_state**2 * mask).sum(1) / denom  # (B, H)
    var = (ex2 - mean**2).clamp(min=0)
    std = var.sqrt()  # (B, H)
    return mean, std


def map_token_pool(batch, max_length=150):
    # Tokenize and send to GPU
    enc = TOK_TOKENIZER(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = TOK_MODEL(**enc)
        mean, std = masked_mean_std(
            out.last_hidden_state, enc["attention_mask"])
        pooled = torch.cat([mean, std], dim=-1)   # (B, 2*H)

    # Convert to list of Python floats for datasets
    pooled = pooled.cpu().numpy()
    return {"token_emb": [x for x in pooled]}


# probably redundant
def run_one_model(pipe, batch):
    texts = batch["text"]  # ← this is crucial

    raw = pipe(texts)
    out = []

    for r in raw:
        if isinstance(r, list):
            out.append({d["label"]: float(d["score"]) for d in r})
        elif isinstance(r, dict):
            out.append({r["label"]: float(r["score"])})
        else:
            out.append({})

    batch["preds"] = out   # attach predictions
    return batch
