# Viettel AI Race — Self-hosted Medical NLP

Pipeline production chạy local, không gọi API khi inference:

```text
MedicalDocument
  → normalize + raw/normalized offset map
  → GLiNER NER
  → deduplicate / overlap / confidence / raw offsets
  → assertion rules + scope + optional local 3B-4B verifier
  → ICD-10 / RxNorm exact + BM25 + optional dense + RRF
  → optional BGE cross-encoder reranker
  → optional local 3B-4B disambiguator
  → validation + competition output
```

## Chạy dependency-free với mock/local model

Các tầng rule, scope, exact alias, BM25, RRF, lexical reranker, validation và
serializer chỉ dùng Python standard library. `ClinicalNLPPipeline` nhận một model
có interface `predict_entities`, nên có thể inject GLiNER thật hoặc mock trong test.

```python
from src.clinical_pipeline import (
    ClinicalNLPPipeline,
    build_default_medical_linker,
)
from src.ner import load_gliner_model
from src.preprocessing import load_documents

model = load_gliner_model("models/gliner-medical", device="cpu")
pipeline = ClinicalNLPPipeline(
    model,
    entity_linker=build_default_medical_linker(),
)

for document in load_documents("notes.jsonl"):
    result = pipeline.process(document)
    print(result.competition_output())
```

`load_gliner_model` chỉ tải mạng nếu truyền Hugging Face model ID chưa có trong
cache. Trong môi trường cuộc thi, dùng đường dẫn checkpoint local.

## Bật hybrid retrieval đầy đủ

Các adapter model nặng được lazy-import và không bắt buộc cho test cơ bản:

```python
from src.linking import (
    CandidateGenerator,
    CrossEncoderReranker,
    HybridEntityLinker,
    SentenceTransformerDenseRetriever,
    load_icd_entries,
)

entries = load_icd_entries("work/icd_mapping_final.json")
dense = SentenceTransformerDenseRetriever(
    entries,
    "models/bge-m3",
    local_files_only=True,
)
generator = CandidateGenerator(entries, dense=dense)
reranker = CrossEncoderReranker(
    "models/bge-reranker-v2-m3",
    local_files_only=True,
)
icd_linker = HybridEntityLinker(
    entries,
    candidate_generator=generator,
    reranker=reranker,
)
```

Model đề xuất được lưu local:

- NER: GLiNER checkpoint đã fine-tune.
- Dense: `BAAI/bge-m3`.
- Reranker: `BAAI/bge-reranker-v2-m3`.
- Verify/disambiguation tùy chọn: Qwen 3B-4B hoặc Llama 3B instruct.

## Chuẩn bị training data GLiNER

Content và label synthetic vẫn lưu riêng. Chỉ tạo `MedicalDocument` trong RAM ở
thời điểm build dataset:

```python
from src.ner import build_gliner_dataset
from src.preprocessing import (
    load_synthetic_data,
    synthetic_to_documents,
)

split = load_synthetic_data(
    "work/generated/contents.jsonl",
    "work/generated/labels.jsonl",
)
documents = synthetic_to_documents(split)
result = build_gliner_dataset(
    documents,
    on_error="skip",
    error_log_path="work/gliner_dataset_errors.jsonl",
)
train_data = result.samples
```

## Test

```bash
python -m unittest discover -s tests -v
```

Các model BGE/Qwen/GLiNER không được tải trong unit test; test dùng index local và
mock model để đảm bảo pipeline chạy offline.
