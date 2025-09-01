import json
import logging
import re
import time
from tqdm import tqdm
import networkx as nx
from openai import AzureOpenAI

# ============== Settings ==============
AZURE_API_KEY = "put your api key here"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_ENDPOINT = "put your endpoint name here"
AZURE_DEPLOYMENT_NAME = "put your model deployment name here"

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

# Model and summarization parameters
MODEL_NAME = 'gpt-4.1-mini'
MAX_COMPLETION_TOKENS = 100   # Maximum tokens generated per request
TEMPERATURE = 0               # Temperature for deterministic output
AVG_ONES = 7                  # Expected average number of sentences to select
MAX_NEIGHBORS_PER_HOP = 20    # Max neighbors shown per hop in graph
RANK_METHOD = "degree"        # Metric for sentence ranking ("degree" or "pagerank")
COVERAGE_RATIO = 0.9          # Ratio for cumulative score coverage
MAX_PROMPT_TOKENS = 22000     # Maximum allowed tokens per prompt
REQUEST_INTERVAL = 1.5        # Delay between API requests (seconds)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

# ============== Function ==============

def load_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def compute_graph_features(node_info):
    """
    Build a graph from node information (sentences and neighbors),
    compute PageRank and degree centrality, and add these features
    back into the node_info structure.
    """
    G = nx.Graph()
    for node in node_info:
        sid = node["sentence_id"]
        G.add_node(sid)
        for hop in ["1-hop", "2-hop", "3-hop"]:
            for neighbor in node.get("neighbors", {}).get(hop, []):
                nid = neighbor[-1] if isinstance(neighbor, list) else neighbor
                G.add_edge(sid, nid)
    pagerank = nx.pagerank(G)
    degree = dict(G.degree())
    for node in node_info:
        sid = node["sentence_id"]
        node["degree"] = degree.get(sid, 0)
        node["pagerank"] = round(pagerank.get(sid, 0.0), 6)
    return node_info

def determine_top_k_by_single_metric(node_info, metric="degree", coverage_ratio=0.9):
    """
    Determine the number of top sentences (k) needed to reach a target coverage ratio
    of the total score (e.g., 90% of total PageRank or degree).
    """
    scores = sorted([node.get(metric, 0.0) for node in node_info], reverse=True)
    total = sum(scores)
    if total == 0:
        return 10
    cumulative = 0.0
    for i, score in enumerate(scores):
        cumulative += score
        if cumulative >= coverage_ratio * total:
            return i + 1
    return len(scores)

def get_top_k_sentence_ids(node_info, k):
    scored = [(node['sentence_id'], node.get(RANK_METHOD, 0.0)) for node in node_info]
    sorted_ids = sorted(scored, key=lambda x: x[1], reverse=True)
    return set(sid for sid, _ in sorted_ids[:k])

def summarize_neighbors_with_partial_text(node_info, text_list, top_k_ids, include_degree, include_pagerank, show_neighbors, max_neighbors_per_hop, use_top_k_mask):
    lines = []
    for node in node_info:
        sid = node.get("sentence_id", -1)

        if use_top_k_mask and sid not in top_k_ids:
            continue

        parts = [f"Sentence {sid+1}:"]
        parts.append(text_list[sid] if sid < len(text_list) else "[Text missing]")

        if include_degree:
            parts.append(f"Degree: {node.get('degree', 0)}")
        if include_pagerank:
            parts.append(f"PageRank: {node.get('pagerank', 0.0)}")

        if show_neighbors:
            neighbor_info_parts = []
            for hop_key in ["1-hop", "2-hop", "3-hop"]:
                raw_neighbors = node.get("neighbors", {}).get(hop_key, [])
                neighbor_ids = sorted(set(
                    item[-1] if isinstance(item, (list, tuple)) else item
                    for item in raw_neighbors
                ))
                if len(neighbor_ids) > max_neighbors_per_hop:
                    short_list = neighbor_ids[:max_neighbors_per_hop]
                    remainder = len(neighbor_ids) - max_neighbors_per_hop
                    neighbor_info_parts.append(f"{hop_key}: {short_list} (+{remainder} more)")
                else:
                    neighbor_info_parts.append(f"{hop_key}: {neighbor_ids}")
            parts.append(", ".join(neighbor_info_parts))

        lines.append("\n".join(parts))

    return "\n\n".join(lines)

def build_graph_prompt(node_info, text_list, top_k, config):
    top_k_ids = get_top_k_sentence_ids(node_info, top_k)
    graph_representation = summarize_neighbors_with_partial_text(
        node_info, text_list, top_k_ids,
        config["INCLUDE_DEGREE"], config["INCLUDE_PAGERANK"],
        config["SHOW_NEIGHBORS"], MAX_NEIGHBORS_PER_HOP,
        config["USE_TOP_K_MASK"]
    )
    graph_info = f"""
## **Graph + Partial Text Information:**
{f"Only the top {top_k} most important sentences include full text." if config["USE_TOP_K_MASK"] else "All sentences are visible."}

{graph_representation}"""
    prompt = f"""
You are an expert in extractive summarization.
Your task is to **select the most important sentences** from a document.

{graph_info}

**On average, select around {AVG_ONES:.2f} key sentences.**

## **Required Output Format:**
Return your response strictly in **valid JSON format**:

```json
{{
  "selected_sentences": [1, 3, 5]
}}

DO NOT return any extra text, explanations, or formatting—only the JSON object.
Now, return only the JSON response:
"""
    return prompt.strip()

def estimate_prompt_tokens(text_list, avg_token_per_sentence=25):
    return len(text_list) * avg_token_per_sentence

def parse_selected_indices(response_text: str):
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip()
    try:
        data = json.loads(cleaned)
        return data.get("selected_sentences", [])
    except json.JSONDecodeError:
        logging.error(f"JSONDecodeError, raw content was:\n{cleaned}")
        return []

def process_documents(input_file: str, output_file: str, config: dict):
    docs = load_data(input_file)
    processed_docs = []

    for idx, entry in enumerate(tqdm(docs, desc="Processing Documents")):
        node_info = entry.get("node_info", [])
        text_list = entry.get("text", [])
        golden_summary = entry.get("golden_summary", [])

        if not node_info or not text_list:
            processed_docs.append({
                "text": text_list,
                "golden_summary": golden_summary,
                "selected_indices": [],
                "generated_summary": [],
                "top_k_used": 0,
                "masked_count": len(text_list),
                "visible_count": 0
            })
            continue

        node_info = compute_graph_features(node_info)

        top_k = determine_top_k_by_single_metric(node_info, metric=RANK_METHOD, coverage_ratio=COVERAGE_RATIO)

        if estimate_prompt_tokens(text_list) + MAX_COMPLETION_TOKENS > MAX_PROMPT_TOKENS:
            logging.warning(f"[Doc {idx}] Skipped due to long prompt (> {MAX_PROMPT_TOKENS} tokens)")
            continue

        prompt = build_graph_prompt(node_info, text_list, top_k, config)

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert in extractive summarization."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    model=AZURE_DEPLOYMENT_NAME
                )
                break
            except Exception as e:
                wait_time = 2 ** attempt
                logging.warning(f"[Retry {attempt+1}] Azure API error: {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)
        else:
            logging.error(f"[Doc {idx}] Failed after retries. Skipping.")
            continue

        raw_text = response.choices[0].message.content.strip()
        selected_ids_1_based = parse_selected_indices(raw_text)
        binary_indices = [1 if (i+1) in selected_ids_1_based else 0 for i in range(len(text_list))]
        selected_sentences = [text_list[i] for i in range(len(text_list)) if binary_indices[i] == 1]

        masked_count = len(text_list) - top_k if config["USE_TOP_K_MASK"] else 0

        doc_entry = {
            "text": text_list,
            "golden_summary": golden_summary,
            "selected_indices": binary_indices,
            "generated_summary": selected_sentences,
            "top_k_used": top_k,
            "masked_count": masked_count,
            "visible_count": len(text_list) - masked_count
        }

        processed_docs.append(doc_entry)

        if idx % 10 == 0:
            save_data(processed_docs, output_file)

        time.sleep(REQUEST_INTERVAL)

    save_data(processed_docs, output_file)
    logging.info(f"Saved results to {output_file}")

# ============== main ==============
if __name__ == "__main__":
    import os
    input_file_path = './data/pubmed_dataset_200_graph.json'
    os.makedirs("./azure-41-mini_pubmed_results", exist_ok=True)
    # Different experimental configurations
    combinations = [
        {"INCLUDE_PAGERANK": False, "INCLUDE_DEGREE": False, "SHOW_NEIGHBORS": True, "USE_TOP_K_MASK": True},
        {"INCLUDE_PAGERANK": False, "INCLUDE_DEGREE": True, "SHOW_NEIGHBORS": False, "USE_TOP_K_MASK": True},
        {"INCLUDE_PAGERANK": False, "INCLUDE_DEGREE": False, "SHOW_NEIGHBORS": False, "USE_TOP_K_MASK": True},
    ]
    # Run experiments with different configs
    def config_to_filename(config: dict) -> str:
        ne = "NEIon" if config["SHOW_NEIGHBORS"] else "NEIoff"
        pr = "PRon" if config["INCLUDE_DEGREE"] else "SCoff"
        tk = "TOPKon" if config["USE_TOP_K_MASK"] else "TOPKoff"
        return f"exp_{pr}_{tk}_{ne}.json"

    for config in combinations:
        fname = config_to_filename(config)
        output_file_path = f'./azure-41-mini_pubmed_results/{fname}'
        logging.info(f"Running: {fname} with config {config}")
        process_documents(input_file_path, output_file_path, config)
