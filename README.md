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

entries = load_icd_entries("data/icd_mapping_final.json")
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
    "data/generated/contents.jsonl",
    "data/generated/labels.jsonl",
)
documents = synthetic_to_documents(split)
result = build_gliner_dataset(
    documents,
    on_error="skip",
    error_log_path="data/gliner_dataset_errors.jsonl",
)
train_data = result.samples
```

## Fine-tune GLiNER NER

Kiểm tra dataset và split mà chưa load model:

```bash
python -m src.train_ner `
  --contents data/generated/contents.jsonl `
  --labels data/generated/labels.jsonl `
  --model models/gliner-base `
  --output models/medical-gliner `
  --validation-ratio 0.1 `
  --dry-run
```

Chạy fine-tuning local:

```bash
python -m src.train_ner `
  --contents data/generated/contents.jsonl `
  --labels data/generated/labels.jsonl `
  --model models/gliner-base `
  --output models/medical-gliner `
  --max-steps 2000 `
  --batch-size 8 `
  --device cuda `
  --bf16
```

Mặc định `--model` phải là đường dẫn local. Trainer ghi lại:

```text
models/medical-gliner/
├── train_dataset.json
├── eval_dataset.json
├── dataset_errors.jsonl
├── training_manifest.json
└── checkpoint/model files
```

## Chạy test set và sinh output

Kiểm tra toàn bộ input trước, bước này không load model:

```powershell
python -m src.run_inference --dry-run
```

Sau khi checkpoint fine-tune đã nằm ở `models/medical-gliner`, chạy local bằng GPU:

```powershell
python -m src.run_inference `
  --input data/test_set/input `
  --model models/medical-gliner `
  --device cuda `
  --output data/test_set/output/predictions.jsonl
```

Nếu máy không có CUDA, đổi `--device cuda` thành `--device cpu`. Kết quả gồm:

```text
data/test_set/output/
├── predictions.jsonl
└── prediction_errors.jsonl
```

Mỗi dòng của `predictions.jsonl` có dạng:

```json
{"note_id":"1","entities":[{"text":"...","type":"CHẨN_ĐOÁN","position":[10,20],"assertions":[],"candidates":["I10"]}]}
```

`note_id` lấy từ tên file (`1.txt` thành `"1"`). Offset trong output luôn được
map về văn bản gốc. Nếu có file lỗi, chương trình vẫn ghi các sample thành công vào
output, ghi chi tiết vào `prediction_errors.jsonl`, rồi trả exit code 1 để tránh bỏ
sót lỗi khi chạy tự động.

## Test

```bash
python -m unittest discover -s tests -v
```

Các model BGE/Qwen/GLiNER không được tải trong unit test; test dùng index local và
mock model để đảm bảo pipeline chạy offline.
