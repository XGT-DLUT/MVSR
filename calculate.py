import json
import os
import ast

# ========== Configuration ==========
DATA_DIR = 'ASTE'
MODEL = "gpt-4o"   # MODEL should be one of ['gpt-4o', 'gpt-4o-mini']
TOPK = 32               # For gpt-4o-mini, use [5, 10]; for gpt-4o, use [32]

# ========== Print Table Header ==========
print("{:^8} | {:^7} | {:^7} | {:^7} | {:^8}".format("Dataset", "P", "R", "F1", "Fail"))
print("-" * 50)

dataset_list = ['14lap', '14res', '15res', '16res']

# ========== Metric Calculation ==========
for dataset in dataset_list:
    save_path = os.path.join(DATA_DIR, dataset, f'test_{MODEL}_{TOPK}.jsonl')
    with open(save_path, 'r', encoding='utf8') as f:
        datas = [json.loads(line) for line in f]

    total_gold = 0    # Total number of gold (reference) triplets
    total_pred = 0    # Total number of predicted triplets
    total_correct = 0 # Number of correctly predicted triplets
    fail_count = 0    # Number of samples that could not be evaluated

    # Evaluate precision, recall, and F1 for each dataset
    for data in datas:
        try:
            labels = json.loads(data['label'])
        except Exception:
            labels = []
        gold_triplets = {tuple(item.lower() for item in label) for label in labels}

        try:
            preds_raw = ast.literal_eval(data['response'])
            pred_triplets = {tuple(item.lower() for item in triplet) for triplet in preds_raw}
        except Exception:
            pred_triplets = set()
            fail_count += 1

        total_gold += len(gold_triplets)
        total_pred += len(pred_triplets)
        total_correct += len(gold_triplets & pred_triplets)  # Set intersection

    # Compute metrics, handle division by zero
    precision = total_correct / total_pred if total_pred else 0.0
    recall = total_correct / total_gold if total_gold else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    # Print results per dataset
    print("{:^8} | {:^7.2f} | {:^7.2f} | {:^7.2f} | {:^8}".format(
        dataset, precision * 100, recall * 100, f1 * 100, fail_count)
    )