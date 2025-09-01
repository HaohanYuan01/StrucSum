# StrucSum

**StrucSum** is a graph-based extractive summarization framework.  

It builds a graph from sentence relationships, computes structural features.

StrucSum uses **LLMs** to automatically select the most important sentences for summarization.  

Arxiv: [StrucSum](https://arxiv.org/abs/2505.22950)

---
## Environment


```
pip install
  - tqdm
  - networkx
  - openai
  - torch
  - sentence-transformers
```

## 📂 Data Preprocessing


Go to the `data/` folder and run the preprocessing script (e.g., for PubMed):

```bash
cd data
python garph_to_prompt.py
```

This will generate structured graph-format data (e.g., `pubmed_test_200_graph.json`), which will be used as input to the summarization pipeline.

---

## 🚀 Experiment

The script `run_strucsum.py` is the core of this project. It performs the following steps:

1. Load preprocessed documents  
2. Construct sentence graphs and compute graph features  
3. Build summarization prompts  
4. Call **Azure** API for extractive summarization  
5. Save results into JSON files  

Run it with:

```bash
python run_strucsum.py
```

---

### ⚙️ Key Parameters and Configuration

All parameters are defined at the beginning of `run_strucsum.py`.
The configuration of using graph-based strategies (NAP, CAP, CGM) is defined at the end of `run_strucsum.py`.

 

#### **Experiment Settings**
Different experiment configurations are predefined in the script:

```python
combinations = [
    {"INCLUDE_PAGERANK": False, "INCLUDE_DEGREE": False, "SHOW_NEIGHBORS": True, "USE_TOP_K_MASK": True},
    {"INCLUDE_PAGERANK": False, "INCLUDE_DEGREE": True, "SHOW_NEIGHBORS": False, "USE_TOP_K_MASK": True},
    {"INCLUDE_PAGERANK": False, "INCLUDE_DEGREE": False, "SHOW_NEIGHBORS": False, "USE_TOP_K_MASK": True},
]
```

Each configuration controls whether to show neighbors (NAP), include PageRank/Degree values (CAP), and whether to mask only top-k sentences (CGM).
Results are saved separately for each configuration.

---

## 📊 Output Format

After running the script, results are saved in the `./azure-41-mini_pubmed_results/` directory.  
Each experiment produces a JSON file, for example:

```json
{
  "text": ["Sentence A", "Sentence B", "Sentence C"],
  "golden_summary": ["Reference summary sentence"],
  "selected_indices": [1, 0, 1],
  "generated_summary": ["Sentence A", "Sentence C"],
  "top_k_used": 5,
  "masked_count": 2,
  "visible_count": 8
}
```

- `text` — the original sentences  
- `golden_summary` — reference/ground-truth summary  
- `selected_indices` — binary indicator (1 = selected, 0 = not selected)  
- `generated_summary` — sentences selected by the model  
- `top_k_used` — number of top-k sentences considered by the algorithm 
- `masked_count` — number of sentences hidden by masking  
- `visible_count` — number of sentences that are actually shown in the prompt

---


## 📌 Notes

- Make sure you have a valid Azure OpenAI deployment and update your API key, endpoint, and deployment name in `run_strucsum.py`.  
- If the input document is too long (exceeding `MAX_PROMPT_TOKENS`), it will be skipped with a warning.  
- You can adjust ranking method (`degree` / `pagerank`), coverage ratio, and masking strategies for different experiments.  

---


