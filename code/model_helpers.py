"""
Helpers for running emotion/toxicity models and generating embeddings.

This is mostly from 'asip_script_pipeline_NNNN.ipynb', consolidated here.
"""

from config import EMOTION_MODELS, TOXICITY_MODEL, TOPIC_MODEL, TOPICS
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from functools import partial
import torch
import pandas as pd


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

DEVICE = 0 if torch.cuda.is_available() else -1

TOK_MAX_LEN = 512
SENT_MAX_SEQ_LEN = 128
CLASSIFIER_MAX_LEN = 150


# ---------------------------------------------------------------------------
# Models / tokenizers
# ---------------------------------------------------------------------------

SENT_EMB = SentenceTransformer("BAAI/bge-base-en-v1.5")
SENT_EMB.max_seq_length = SENT_MAX_SEQ_LEN

TOK_TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")
TOK_MODEL = AutoModel.from_pretrained("roberta-base")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def flatten_scores(row):
    """Convert a list of {'label': ..., 'score': ...} dicts into a flat dict."""
    return {d["label"]: d["score"] for d in row}


def scores_to_df(model_output):
    """
    Convert a datasets object with a 'raw' column of HF outputs into a DataFrame
    of label columns.
    """
    raw_ = model_output.to_pandas()["raw"]
    return pd.DataFrame(raw_.apply(flatten_scores).tolist())


def _normalize_pipeline_output(res):
    """
    Normalize HF pipeline outputs into consistent:
        list[ dict(label -> score) ]
    """

    # Ensure list
    if not isinstance(res, list):
        res = [res]

    out = []
    for r in res:

        # ----- ZERO–SHOT CASE -----
        if isinstance(r, dict) and ("labels" in r and "scores" in r):
            out.append({label: float(score)
                       for label, score in zip(r["labels"], r["scores"])})
            continue

        # ----- STANDARD LIST[dict] CASE -----
        if isinstance(r, list):
            out.append({d["label"]: float(d["score"]) for d in r})
            continue

        # ----- SINGLE DICT CASE -----
        if isinstance(r, dict) and "label" in r:
            out.append({r["label"]: float(r.get("score", 1.0))})
            continue

        # Fallback – empty dict
        out.append({})

    return out


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def make_pipeline(hf_model: str, task: str, **pipe_kwargs):
    """
    Generic HF pipeline factory.

    Parameters in pipe_kwargs override defaults.
    """
    pipe_params = {
        "task": task,
        "model": hf_model,
        "device": DEVICE,
        "truncation": True,
        "max_length": TOK_MAX_LEN,
    }
    pipe_params.update(pipe_kwargs)

    # Standard text-classification defaults
    if task == "text-classification":
        pipe_params.setdefault("padding", True)
        pipe_params.setdefault("top_k", None)  # return all labels

    return pipeline(**pipe_params)


def make_classifier(hf_model: str, max_length: int = CLASSIFIER_MAX_LEN):
    """
    Convenience wrapper for a text-classification pipeline that returns all
    label scores.
    """
    return make_pipeline(
        hf_model,
        task="text-classification",
        max_length=max_length,
        padding=True,
        top_k=None,
    )

# ---------------------------------------------------------------------------
# Running classifiers over datasets / batches
# ---------------------------------------------------------------------------


def run_pipeline_raw(batch, classifier, **kwargs):
    """
    Run a classifier on a datasets batch, returning label→score dicts.

    Returns:
        {"scores": list[dict(label -> score)]}
    """
    texts = batch["text"]
    res = classifier(texts, **kwargs)
    scores = _normalize_pipeline_output(res)
    return {"scores": scores}


def full_model_run(hf_model: str, dataset_):
    """
    Run one HF model over an entire datasets.Dataset and return the mapped
    dataset with a new 'scores' column.
    """
    cl = make_classifier(hf_model)
    c_raw = partial(run_pipeline_raw, classifier=cl)
    output = dataset_.map(c_raw, batched=True, batch_size=16)
    return output


def process_lines_all(batch):
    """
    Run toxicity + all emotion models in parallel on a batch of lines.

    Input batch must have:
        batch["text"] -> list[str]

    Returns a dict:
        {
          "toxicity": list[dict(label -> score)],
          "emo_0":   list[dict(label -> score)],
          "emo_1":   ...,
          ...
        }
    """
    texts = batch["text"]
    results = {}

    # Instantiate pipelines once (static attribute)
    if not hasattr(process_lines_all, "_pipes"):
        process_lines_all._pipes = {
            "toxicity": make_classifier(TOXICITY_MODEL),
            **{f"emo_{i}": make_classifier(m)
               for i, m in enumerate(EMOTION_MODELS)},
            # # "topic": make_pipeline(
            # #     TOPIC_MODEL, "zero-shot-classification", multi_label=True
            # )
        }

    pipes = process_lines_all._pipes

    def run_pipe(name, pipe_):
        if name == "topic":
            res = pipe_(
                texts,
                candidate_labels=TOPICS,
                multi_label=True
            )
        else:
            res = pipe_(texts)

        scores = _normalize_pipeline_output(res)
        return name, scores

    # Parallelize over models (each runs on GPU sequentially but across
    # models we can use threads).
    with ThreadPoolExecutor(max_workers=len(pipes)) as ex:
        futures = [ex.submit(run_pipe, name, pipe_)
                   for name, pipe_ in pipes.items()]
        for fut in as_completed(futures):
            name, scores = fut.result()
            results[name] = scores

    return results


# ---------------------------------------------------------------------------
# Embeddings (token-level, pooled)
# ---------------------------------------------------------------------------

def masked_mean_std(last_hidden_state, attention_mask):
    """
    Compute masked mean and std over time dimension.

    last_hidden_state: (B, T, H)
    attention_mask:    (B, T)
    """
    mask = attention_mask.unsqueeze(-1)          # (B, T, 1)
    denom = mask.sum(1).clamp(min=1)            # (B, 1)

    mean = (last_hidden_state * mask).sum(1) / denom  # (B, H)

    # masked variance = E[x^2] - (E[x])^2
    ex2 = (last_hidden_state ** 2 * mask).sum(1) / denom
    var = (ex2 - mean ** 2).clamp(min=0)
    std = var.sqrt()
    return mean, std


def map_token_pool(batch, max_length: int = CLASSIFIER_MAX_LEN):
    """
    Map function for datasets: create pooled token embeddings per line.

    Returns:
        {"token_emb": list[np.ndarray (2*hidden_dim,)]}
    """
    enc = TOK_TOKENIZER(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        out = TOK_MODEL(**enc)
        mean, std = masked_mean_std(
            out.last_hidden_state, enc["attention_mask"])
        pooled = torch.cat([mean, std], dim=-1)  # (B, 2*H)

    pooled = pooled.cpu().numpy()
    return {"token_emb": [x for x in pooled]}
