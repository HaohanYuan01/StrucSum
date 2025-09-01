import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def compute_similarity(model, sentences, batch_size=64):
    """
    Compute pairwise similarity for a list of sentences using a pre-loaded model.

    Parameters:
        model: A pre-loaded SentenceTransformer model.
        sentences (list): List of sentences.
        batch_size (int): Batch size for model.encode() to prevent high memory usage.

    Returns:
        torch.Tensor: Pairwise similarity matrix of shape (n, n).
    """
    # Convert sentences to embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True, batch_size=batch_size)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    return similarity_matrix

def create_document_graph(sentences, similarity_matrix, threshold=0.5,
                          max_hop=3, store_paths=True):
    """
    Create graph data for a document based on sentence similarity.

    Parameters:
        sentences (list): List of sentences.
        similarity_matrix (torch.Tensor): n x n similarity matrix.
        threshold (float): Similarity threshold for edge creation.
        max_hop (int): Maximum hop distance to explore neighbors.
        store_paths (bool): Whether to store full paths to neighbors or just node IDs.

    Returns:
        list: Node information for each sentence (node), including multi-hop neighbors.
    """
    n = len(sentences)
    # Step 1: Build adjacency list from similarity_matrix
    adjacency_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and similarity_matrix[i][j].item() >= threshold:
                adjacency_list[i].append(j)

    # Step 2: For each node, find neighbors up to max_hop
    node_info = []
    for node_id in range(n):
        node_data = {
            "sentence_id": node_id,
            "sentence": sentences[node_id],
            "neighbors": {}
        }

        visited = {node_id}
        current_hop_nodes = [node_id]
        # If storing paths, keep a dictionary {node_id: [path...]}
        paths = {node_id: [node_id]} if store_paths else None

        for hop in range(1, max_hop + 1):
            next_hop_nodes = []
            for current_node in current_hop_nodes:
                neighbors = [nbr for nbr in adjacency_list[current_node] if nbr not in visited]
                visited.update(neighbors)

                if store_paths:
                    for nbr in neighbors:
                        paths[nbr] = paths[current_node] + [nbr]

                next_hop_nodes.extend(neighbors)

            if next_hop_nodes:
                if store_paths:
                    node_data["neighbors"][f"{hop}-hop"] = [
                        tuple(paths[nbr]) for nbr in next_hop_nodes
                    ]
                else:
                    node_data["neighbors"][f"{hop}-hop"] = next_hop_nodes

            current_hop_nodes = next_hop_nodes

        node_info.append(node_data)

    return node_info

def process_documents(input_json, output_json, threshold=0.7, device="cuda",
                      batch_size=64, max_hop=3, store_paths=True):
    """
    Process documents to extract graph-based node information and embed them back into the document structure.

    Parameters:
        input_json (str): Path to the input JSON file.
        output_json (str): Path to save the output JSON file.
        threshold (float): Similarity threshold for edge creation.
        device (str): Device to run the model (e.g., "cuda" or "cpu").
        batch_size (int): Batch size for encoding sentences to avoid memory issues.
        max_hop (int): Maximum hop distance to explore neighbors (default 3).
        store_paths (bool): Whether to store the actual path or just the node IDs in neighbors.
    """
    # 1) Load model once
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    # 2) Load input JSON
    with open(input_json, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    for doc in tqdm(documents, desc="Processing documents"):
        # Safely get text
        sentences = doc.get("text", [])
        if not sentences:
            # If there's no sentence, skip graph construction
            doc["node_info"] = []
            continue

        # Attempt to get labels
        labels = doc.get("label", [0] * len(sentences))
        if len(labels) != len(sentences):
            print(f"[Warning] Label size {len(labels)} != #Sentences {len(sentences)} in doc: {doc.get('id', 'Unknown')}")

        # 3) Compute similarity matrix
        similarity_matrix = compute_similarity(model, sentences, batch_size=batch_size)

        # 4) Build graph (neighbors up to max_hop)
        node_info = create_document_graph(
            sentences,
            similarity_matrix,
            threshold=threshold,
            max_hop=max_hop,
            store_paths=store_paths
        )

        # 5) Attach labels
        for node in node_info:
            sid = node["sentence_id"]
            node["label"] = labels[sid] if sid < len(labels) else 0

        # 6) Embed node_info back into the document
        doc["node_info"] = node_info

    # 7) Save to output JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)

    print(f"Processed data saved to {output_json}")

# ====================

if __name__ == "__main__":
    input_json = "./pubmed_test_200.json"
    output_json = "./pubmed_test_200_graph.json"

  
    process_documents(
        input_json=input_json,
        output_json=output_json,
        threshold=0.7,
        device="cuda",
        batch_size=64,
        max_hop=3,
        store_paths=True
    )

