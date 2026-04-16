import os
import json
import torch
import spacy
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rouge_score import rouge_scorer
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading

# ============================== #
#         Parameter Class        #
# ============================== #
class Args:
    model_name = 'bert-base-uncased'     # Name or path of the pretrained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = min(8, multiprocessing.cpu_count())  # Number of threads to use
    datasets = ['14lap', '14res', '15res', '16res']    # Datasets to process
    data_dir = 'ASTE'                                  # Root data directory
    use_stemmer = False
    max_length = 512                                   # Max token length for model input

args = Args()

# ============================== #
#      Thread-local Cache        #
# ============================== #
thread_local = threading.local()

def get_tokenizer():
    """Thread-safe loading of the tokenizer."""
    if not hasattr(thread_local, "tokenizer"):
        thread_local.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return thread_local.tokenizer

def get_spacy():
    """Thread-safe loading of the spaCy language model."""
    if not hasattr(thread_local, "nlp"):
        thread_local.nlp = spacy.load("en_core_web_sm")
    return thread_local.nlp

def preload_thread_local():
    """Preload the tokenizer and spaCy model in the main thread (to speed up thread startup)."""
    get_tokenizer()
    get_spacy()

# ============================== #
#   Embedding and Structure      #
# ============================== #
def get_embedding(text):
    """
    Get contextualized token embeddings and [CLS] embedding for a sentence.
    """
    tokenizer = get_tokenizer()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=args.max_length)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)
        cls_embedding = hidden[0].to(args.device)  
    return hidden, cls_embedding

def get_structure(text):
    """
    Extract POS and dependency tags from a sentence using spaCy.
    """
    doc = get_spacy()(text)
    pos = [token.pos_ for token in doc]
    dep = [token.dep_ for token in doc]
    return pos, dep

# ============================== #
#   Similarity Computation       #
# ============================== #
def compute_bertscore_f1_batch_fast(test_embed, train_tensor, mask_tensor):
    """
    Fast BERTScore-like F1 computation in batch mode.
    test_embed: tensor [seq_len, hidden_dim]
    train_tensor: tensor [N, max_len, hidden_dim]
    mask_tensor: tensor [N, max_len], 1 for valid tokens, 0 for padding
    """
    test_embed = test_embed[1:-1]   # Remove [CLS] and [SEP] tokens
    N = train_tensor.shape[0]
    test_expand = test_embed.unsqueeze(0).expand(N, -1, -1)  # [N, seq_len, dim]
    norm_test = F.normalize(test_expand, dim=-1)
    sim = torch.bmm(norm_test, train_tensor.transpose(1, 2))  # [N, seq_len, max_len]

    # Precision: for each test token, find the most similar train token (mean over tokens)
    prec = sim.max(dim=2)[0].mean(dim=1)
    # Recall: for each train token, find most similar test token (masked mean)
    recall = sim.max(dim=1)[0]
    recall_sum = (recall * mask_tensor).sum(dim=1)
    recall_mean = recall_sum / mask_tensor.sum(dim=1).clamp(min=1)

    f1 = 2 * prec * recall_mean / (prec + recall_mean + 1e-8)
    return f1.tolist()

def compute_rouge_scores(candidate_tokens, reference_tokens):
    """
    Compute average ROUGE F1 score (rouge1, rouge2, rougeL, and optionally rouge3)
    for two token lists.
    """
    candidate_str = ' '.join(candidate_tokens)
    reference_str = ' '.join(reference_tokens)
    metrics = ['rouge1', 'rouge2', 'rougeL']
    # For long enough sequences, add rouge3 as an additional metric
    if len(candidate_tokens) >= 3 and len(reference_tokens) >= 3:
        metrics.append('rouge3')

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=args.use_stemmer)
    scores = scorer.score(candidate_str, reference_str)

    f1 = sum([scores[m].fmeasure for m in metrics]) / len(metrics)
    return f1

# ============================== #
#     Retrieval and Ranking      #
# ============================== #
def calculate_similarities(sample, train_features, cls_tensor, token_tensor, mask_tensor):
    """
    Compute similarity scores between a test sample and all training features.
    Returns a sorted list (descending) by final similarity.
    """
    last_hidden, cls_embedding = get_embedding(sample['input'])
    pos_tags, dep_tags = get_structure(sample['input'])

    # CLS embedding cosine similarity
    cls_sims = F.cosine_similarity(cls_embedding.unsqueeze(0), cls_tensor)
    # Token-level embedding similarity (BERTScore-style F1)
    bert_sims = compute_bertscore_f1_batch_fast(last_hidden, token_tensor, mask_tensor)

    similarities = []
    for i, feat in enumerate(train_features):
        # Syntactic similarity using ROUGE between POS and DEP tag sequences
        pos_sim = compute_rouge_scores(pos_tags, feat['pos'])
        dep_sim = compute_rouge_scores(dep_tags, feat['dep'])
        cls_sim = cls_sims[i]
        bert_sim = bert_sims[i]
        # Final similarity is sum of all four scores
        final_sim = cls_sim + bert_sim + pos_sim + dep_sim

        similarities.append({
            'input': feat['input'],
            'label': feat['label'],
            'cls_sim': round(float(cls_sim), 4),
            'bert_sim': round(float(bert_sim), 4),
            'pos_sim': round(float(pos_sim), 4),
            'dep_sim': round(float(dep_sim), 4),
            'final_sim': round(float(final_sim), 4)
        })

    similarities = sorted(similarities, key=lambda x: x['final_sim'], reverse=True)
    return similarities

# ============================== #
#    Main Sample Processing      #
# ============================== #
def process_single_sample(sample, train_features, cls_tensor, token_tensor, mask_tensor):
    """
    For a single test sample, retrieve and rank all training examples by similarity.
    """
    similarities = calculate_similarities(sample, train_features, cls_tensor, token_tensor, mask_tensor)
    return {
        "input": sample["input"],
        "label": sample["label"],
        "examples": similarities
    }

# ============================== #
#           Main Loop            #
# ============================== #
if __name__ == "__main__":
    # Load transformer model once globally (not per-thread)
    model = AutoModel.from_pretrained(args.model_name).to(args.device).eval()
    preload_thread_local()

    for dataset in args.datasets:
        print(f"🚀 Using {args.num_workers} threads...📘 Processing: {dataset}")
        train_path = os.path.join(args.data_dir, dataset, 'retrieval_features.pt')
        test_path = os.path.join(args.data_dir, dataset, 'test.jsonl')
        out_path = os.path.join(args.data_dir, dataset, 'test_examples.jsonl')

        # Load precomputed training features (with embeddings)
        train_features = torch.load(train_path)
        cls_tensor = torch.tensor([x["cls_embedding"] for x in train_features], device=args.device)

        # Prepare batched token-level embedding tensors and mask for efficient computation
        d = len(train_features[0]["token_embeddings"][0])              # Embedding dimension
        max_len = max(len(x["token_embeddings"]) - 2 for x in train_features)  # Exclude [CLS]/[SEP]
        N = len(train_features)

        token_tensor = torch.zeros((N, max_len, d), device=args.device)
        mask_tensor = torch.zeros((N, max_len), dtype=torch.bool, device=args.device)

        for i, feat in enumerate(train_features):
            tokens = torch.tensor(feat["token_embeddings"][1:-1], device=args.device)  # Remove [CLS]/[SEP]
            token_tensor[i, :tokens.shape[0], :] = tokens
            mask_tensor[i, :tokens.shape[0]] = 1
        token_tensor = F.normalize(token_tensor, dim=-1)

        # Load test set
        with open(test_path, 'r', encoding='utf8') as f:
            test_data = [json.loads(line) for line in f]

        # Partial function for multiprocessing
        process_func = partial(
            process_single_sample,
            train_features=train_features,
            cls_tensor=cls_tensor,
            token_tensor=token_tensor,
            mask_tensor=mask_tensor
        )

        # Thread pool for parallel processing of test samples
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(tqdm(executor.map(process_func, test_data), total=len(test_data)))

        # Write output with ranked examples to JSONL
        with open(out_path, 'w', encoding='utf8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Finished processing {dataset}, new ranked data saved to {out_path}")

    print("🎉🎉🎉 All datasets processed.")
