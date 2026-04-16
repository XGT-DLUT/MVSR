# Codebase for SSFF

This repository contains the source code and necessary data files for reproducing the experiments presented in our paper on Semantic-Syntactic Multidimensional Fusion for In-Context Learning in Aspect Sentiment Triplet Extraction.

## 📁 Directory Structure
Here is the complete directory structure. Due to file size limitations, some files are not uploaded.
```
.
├──📁 ASTE                                    # All datasets
│   ├──📁 14lap                               # Dataset for laptop reviews (2014)
│   │    ├──📄 train.jsonl                    # Training set  
│   │    ├──📄 dev.jsonl                      # Development set
│   │    ├──📄 test.jsonl                     # Test set
│   │    ├──📄 test_examples.jsonl            # Test set with added 'examples' per instance (not uploaded)  
│   │    ├──📦 retrieval_features.pt          # Precomputed training example features (not uploaded)  
│   │    ├──📄 retrieval_features_human.jsonl # Human-readable feature file (not uploaded)  
│   │    ├──📄 test_gpt-4o_32.jsonl           # GPT-4o predictions (32-shot)
│   │    ├──📄 test_gpt-4o-mini_5.jsonl       # GPT-4o-mini predictions (5-shot)
│   │    └──📄 test_gpt-4o-mini_10.jsonl      # GPT-4o-mini predictions (10-shot)
│   ├── 📁 14res                              # Dataset for restaurant reviews (2014)
│   ├── 📁 15res                              # Dataset for restaurant reviews (2015)
│   └── 📁 16res                              # Dataset for restaurant reviews (2016)
├──🐍 features.py                             # Script for constructing semantic/syntactic features
├──🐍 retrieval.py                            # Script for retrieving similar examples 
├──🐍 inference.py                            # Script to interact with the GPT API for inference
├──🐍 calculate.py                            # Script for calculating precision, recall, and F1
├──📄 requirements.txt                        # Python package dependencies     
└──📝 README.md                               # This file
```
---

## 🚀 Usage Instructions
Follow these steps to reproduce our experimental workflow:

### 0. Quickly Reproducing the Main Results
Run the following script to quickly reproduce the primary results reported in the paper:
```bash
python calculate.py
```
Modify the `MODEL` and `TOPK` parameters in calculate.py to select the model and number of demonstration examples.

### 1. Environment Setup
Fully reproduce the project's workflow by setting up the environment:
```bash
conda create -n mvsr python=3.10
conda activate mvsr
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
Recommended Runtime Configuration:
- **CUDA**: 11.8  
- **PyTorch**: 2.3.0  
- **Python**: 3.10.14

### 2. Construct Training Example Features  
Generate semantic and syntactic features for each training example:
```bash
python features.py
```

### 3. Retrieve Similar Examples
Retrieve similar examples for test samples:
```bash
python retrieval.py
```

### 4. Perform Inference
Perform inference using specified models and demonstrations:
```bash
python inference.py
```

### 5. Evaluate Results
Calculate micro-averaged precision, recall, and F1:
```bash
python calculate.py
```

## 📌 Additional Notes
- GPT prediction results are cached to facilitate rapid reproduction.

- Results from individual runs may vary slightly from the averages reported in the paper (which use three random seeds).

