import os
import json
import time
import random
import multiprocessing
from tqdm import tqdm
from openai import OpenAI

# ======================== Configuration Class ========================
class Args:
    # OpenAI API settings
    api_key = ""                  # ← Replace with your OpenAI API Key
    base_url = ""                 # Optional: set if using a custom endpoint
    model = "gpt-4o-mini"
    temperature = 1e-8
    seed = 1

    # Data and task settings
    data_dir = 'ASTE'
    topk = 5                      # Number of retrieved examples per prompt
    system_prompt = "You're an expert in textual sentiment analysis."
    dataset_list = ['14lap', '14res', '15res', '16res']

    # Prompt template for few-shot ICL
    few_shot_template = (
        "Given a review, extract its aspect term, opinion term and the corresponding sentiment polarity, "
        "where the categories of sentiment polarity are positive, negative and neutral. "
        "Please generate responses strictly in the following format: "
        '[["aspect1", "opinion1", "sentiment1"], ["aspect2", "opinion2", "sentiment2"], ...]. '
        "Here are some examples:\n{examples_str}Review: {input}\nAnswer: "
    )

args = Args()


# ======================== Utility Functions ========================
def init_worker():
    """
    Worker initializer: creates an OpenAI client in each subprocess.
    """
    global client
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

def build_prompt(data):
    """
    Construct a prompt string for GPT API, including top-k retrieved examples.
    """
    examples = data.get('examples', [])[:args.topk]
    examples_str = ""
    for example in examples:
        examples_str += f"Review: {example['input']}\nAnswer: {example['label']}\n"
    return args.few_shot_template.format(examples_str=examples_str, input=data["input"])

def worker(worker_args):
    """
    Process a chunk of test examples: for each, construct prompt and query OpenAI.
    Results are saved in a temporary .jsonl file (one per worker).
    """
    dataset, worker_id, data_chunk = worker_args
    output_file = os.path.join(args.data_dir, dataset, f'temp_{worker_id}.jsonl')
    results = []

    for record_id, data in data_chunk:
        prompt = build_prompt(data)

        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=args.temperature,
                seed=args.seed
            )
            response_text = completion.choices[0].message.content.strip()
        except Exception as e:
            response_text = f"[ERROR] {e}"

        results.append({
            "id": record_id,
            "input": data["input"],
            "label": data.get("label"),
            "response": response_text
        })

    with open(output_file, 'w', encoding='utf8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ======================== Main Task Function ========================
def get_response(dataset):
    """
    For a given dataset, read the test set, use multiprocessing to query the GPT API,
    and aggregate the results into a final output file.
    """
    print(f"\n📘 Processing dataset: {dataset}")
    dataset_dir = os.path.join(args.data_dir, dataset)
    test_path = os.path.join(dataset_dir, "test_examples.jsonl")
    output_path = os.path.join(dataset_dir, f'test_{args.model}_{args.topk}.jsonl')

    # Load test data (already with top-k retrieved examples per instance)
    with open(test_path, 'r', encoding='utf8') as fr:
        raw_data = json.load(fr)
    tasks = [(i + 1, item) for i, item in enumerate(raw_data)]  # Add sample IDs

    cpu_count = multiprocessing.cpu_count()
    num_chunks = min(cpu_count, len(tasks))
    chunk_size = (len(tasks) + num_chunks - 1) // num_chunks
    chunks = [tasks[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    args_list = [(dataset, idx, chunk) for idx, chunk in enumerate(chunks)]

    start_time = time.time()
    with multiprocessing.Pool(processes=num_chunks, initializer=init_worker) as pool:
        list(tqdm(pool.imap(worker, args_list), total=num_chunks, desc=f"⏳ {dataset}"))

    duration = time.time() - start_time

    # Merge all worker results into the final output file
    with open(output_path, 'w', encoding='utf8') as fw:
        for i in range(num_chunks):
            temp_path = os.path.join(dataset_dir, f'temp_{i}.jsonl')
            with open(temp_path, 'r', encoding='utf8') as fr:
                fw.writelines(fr.readlines())
            os.remove(temp_path)

    print(f"✅ Finished: {dataset} | Records: {len(tasks)} | Time: {duration:.1f}s")


# ======================== Entry Point ========================
def main():
    random.seed(args.seed)
    for dataset in args.dataset_list:
        get_response(dataset)
    print("\n🎉 All datasets processed.")


if __name__ == "__main__":
    main()
