# Document Embeddings

Based on the results of: https://arxiv.org/abs/2304.14796

## Implemented Methods

- Average Pooling, with adjustable range for sentences used.
- PERT weighted average pooling


## Usage

Wrapper for Sentence-Embedding, which is used to provide embedding functionality

```python
sentence_model = SentenceEmbedding("...")

document_model = AverageDocumentEmbedder(sentence_model)
document_model = PERTDocumentEmbedder(sentence_model)

document_model = DocumentEmbedder(sentence_model)

doc1 = "Arbitrary text"
doc2 = "..."

document_model.encode([doc1, doc2, ...])


```
