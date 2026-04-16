'''Prepare retrieval features from the training set, including semantic and syntactic features of each sentence.'''
import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import spacy

# === Configuration ===
MODEL_NAME = 'bert-base-uncased'  # Path to the pretrained transformer model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'      # Use GPU if available, otherwise CPU
DATASET_LIST = ['14lap', '14res', '15res', '16res']          # Names of datasets to process

# === Utility Functions ===
def extract_pos_dep(doc):
    """Extract POS tags and dependency tags from a spaCy Doc object."""
    pos_tags = [token.pos_ for token in doc]  # Part-of-speech tags
    dep_tags = [token.dep_ for token in doc]  # Dependency parse tags
    return pos_tags, dep_tags

# === Main Processing Pipeline ===
def process_dataset(dataset_name, tokenizer, model, nlp):
    # Define input/output file paths
    input_path = f'ASTE/{dataset_name}/train.jsonl'                    # Path to the input training set
    output_path = f'ASTE/{dataset_name}/retrieval_features.pt'         # Output path for all features (torch serialized)
    output_debug_path = f'ASTE/{dataset_name}/retrieval_features_human.jsonl'  # Output path for human-readable version

    # Load all training samples
    with open(input_path, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f]

    features = []

    with torch.no_grad(): 
        for item in tqdm(data, desc=f"Processing {dataset_name}"):
            text = item['input']

            # === Semantic Feature Extraction ===
            # Encode the sentence and get contextualized embeddings from the model
            encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = output.last_hidden_state.squeeze(0)  # Tensor: [seq_len, hidden_dim]
            cls_embedding = last_hidden[0]  # Use [CLS] token embedding as sentence embedding

            # === Syntactic Feature Extraction ===
            doc = nlp(text)                        # Use spaCy to parse the sentence
            pos, dep = extract_pos_dep(doc)        # Get POS and dependency tags

            # Collect all features for the current example
            features.append({
                'input': text,                                     # Original input text
                'label': item.get('label', ''),                    # Label
                'tokens': tokenizer.convert_ids_to_tokens(input_ids[0]),  # Tokenized input
                'input_ids': input_ids[0].cpu().tolist(),          # Token IDs
                'token_embeddings': last_hidden.cpu().tolist(),    # Embeddings for each token
                'cls_embedding': cls_embedding.cpu().tolist(),     # [CLS] sentence embedding
                'pos': pos,                                       # POS tags
                'dep': dep                                        # Dependency tags
            })

    # Save full feature list to .pt file (torch binary format)
    torch.save(features, output_path)
    print(f"✅ Saved {len(features)} samples to {output_path}")

    # Save a simplified human-readable version for debugging/visualization
    with open(output_debug_path, 'w', encoding='utf8') as f:
        for item in features:
            debug_item = {
                'input': item['input'],
                'label': item['label'],
                'tokens': item['tokens'],
                'input_ids': item['input_ids'],
                'pos': item['pos'],
                'dep': item['dep']
            }
            f.write(json.dumps(debug_item, ensure_ascii=False) + '\n')
    print(f"✅ Debug version saved to {output_debug_path}")


# === Script Entry Point ===
if __name__ == "__main__":
    print("🔧 Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)     # Load tokenizer
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)  # Load transformer model and move to device
    model.eval()                                              # Set model to evaluation mode (no dropout)
    nlp = spacy.load("en_core_web_sm")                        # Load spaCy for English syntactic analysis

    # Process all datasets one by one
    for dataset in DATASET_LIST:
        process_dataset(dataset, tokenizer, model, nlp)

    print("🎉 All datasets processed.")
