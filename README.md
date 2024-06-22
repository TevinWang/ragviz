# <img src="https://boston.lti.cs.cmu.edu/tevinw/ragviz/ui/ragviz-square.png" alt="drawing" width="50" height="50"/> <p>RAGViz</p>

RAGViz (Retrieval Augmented Generation Visualization) is a tool that visualizes both document and token-level attention on the documents provided as context to the LLM.

- RAGViz provides an add/remove document functionality to compare the generated tokens when certain documents are not included in the context.
- Combining both functionalities allows for a diagnosis on the effectiveness and influence of certain retrieved documents.

### Architecture

You can see the architecture of our RAGViz demo here:
- The Pile-CC English documents are used for retrieval
- Documents are partioned into 4 DiskANN indexes on separate nodes, each with ~20 million documents
- Documents are embedded into feature vectors using the AnchorDR model
- LLaMa2 generation/attention output done with vLLM and HuggingFace transformers library
- Frontend UI is adapted from Lepton search engine

### Customization

**Snippets:** You can modify the snippets used for context in RAG by adding a new file and class in `backend/snippet`, adding it to `backend/ragviz.py` and `frontend/src/app/components/search.tsx`. We currently offer a "naive first" snippet method where we take the first 128 tokens, and a "sliding window" snippet method where we compute inner product similarity between a window of 128 tokens and the query.

**Datasets:** New datasets for retrieval can be added using a new file and class in `backend/search`, and modifying `backend/ragviz.py` accordingly. We currently have implemented both a implementation for Clueweb22B english documents and the Pile-CC dataset.

**LLMs:** You can set the model path of the model for RAG inside of `backend/.env.example`. We used `meta-llama/Llama-2-7b-chat-hf` for the demo.

