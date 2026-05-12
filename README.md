# Dynamic Segmented Memory (DSM)

DSM is a high-performance memory engine designed for efficient context management in Large Language Models. It implements a hybrid retrieval architecture combining hierarchical category routing, semantic vector search, and associative graph expansion.

Developed by [Nare Labs](https://narelabs.com).

## Overview

Traditional context management (RAG) often lacks semantic structure and associative depth. DSM addresses this by organizing knowledge into a triplet state: `(S, T, G)`

- **S (Segments)**: Atomic units of text with semantic embeddings and metadata.
- **T (Hierarchy)**: A dynamic category tree used for high-level beam-search routing.
- **G (Graph)**: A semantic graph preserving associative links between related segments.

## Why DSM vs. Standard RAG?

Standard RAG (Retrieval-Augmented Generation) often treats context as a flat list of chunks, leading to several limitations that DSM solves:

| Challenge | Standard RAG | DSM Engine |
| :--- | :--- | :--- |
| **Context Fog** | Chunks are retrieved in isolation. | **Graph edges** preserve logical flow and dependencies. |
| **Search Speed** | Linear vector search over $N$ chunks. | **Hierarchical routing** ($T$) enables $O(\log N)$ pruning. |
| **Associativity** | Cannot "hop" to related concepts. | **Graph expansion** ($G$) finds connected "needles" automatically. |
| **Organization** | Flat database. | **Dynamic Tree** ($T$) mirrors the project structure. |

## Key Features

- **O(log N) Routing**: Hierarchical category tree allows for fast candidate pruning.
- **Associative Retrieval**: Graph-based expansion enables the model to "follow" code dependencies across files.
- **Sliding Window Ingestion**: Automatic character-based chunking with configurable overlap.
- **Priority Scoring**: Ensembled scoring based on similarity, recency (LRU), and manual importance.
- **Offline First**: Uses local deterministic hashing for embeddings by default (no external API required).

## Performance

Results from benchmark tests using Qwen-2.5-1.5B on 160k token datasets:

| Feature | Performance |
| :--- | :--- |
| **Retrieval Latency** | ~430ms (10k segments) |
| **Efficiency Ratio** | ~400,000x over dense attention |
| **Storage Density** | ~70MB per 100k tokens (JSON) |

## Installation

```bash
git clone https://github.com/narelabs/dsm
cd dsm
pip install -e .
```

## Quick Start

```python
from dsm import DynamicSegmentedMemory

# 1. Initialize Memory Engine
memory = DynamicSegmentedMemory(storage_path=".dsm/data.json")

# 2. Ingest Content
memory.write(
    "The core logic resides in narefield/core/fission.py. It handles node splitting.",
    category_path="Research / Engine / Core",
    importance=0.8
)

# 3. Retrieve Context
# DSM will use Category Routing + Graph Hops to find relevant blocks
active = memory.active_context("Where is the node splitting logic?", k=3)

print(active.context_text)
```

## Architecture

1. **Category Tree**: Routes queries to relevant clusters using beam search.
2. **Vector Index**: Performs fallback k-NN search for direct hits.
3. **Semantic Graph**: Expands seeds to capture contextually linked data.
4. **Active Context**: Builds a bounded prompt buffer for the LLM.

## License

MIT License. Developed by Nare Labs.
[narelabs.com](https://narelabs.com)
