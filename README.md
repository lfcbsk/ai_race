# Viettel AI Race — Self-hosted Medical NLP

Pipeline xử lý hồ sơ y tế tiếng Việt chạy local:

```text
TXT test set
  → MedicalDocument
  → normalize + offset map
  → GLiNER NER
  → lọc confidence / deduplicate / map về raw offset
  → assertion: isNegated / isHistorical / isFamily
  → ICD-10 và RxNorm linking
  → validate schema
  → output.zip (output/1.json, ..., output/100.json)
```

Inference mặc định không gọi API. GLiNER chạy từ checkpoint local; assertion dùng
rule local; linking dùng ontology JSON, exact match, BM25 và lexical reranker.

## 1. Yêu cầu

- Windows, Linux hoặc macOS.
- `uv`.
- Python 3.11 hoặc 3.12; khuyến nghị Python 3.12.
- Khoảng 5 GB trống cho environment và model GLiNER.
- NVIDIA GPU được khuyến nghị khi fine-tune. Inference có thể chạy CPU.

Kiểm tra `uv`:

```powershell
uv --version
```

Nếu chưa có `uv`, cài trên Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Tài liệu chính thức: <https://docs.astral.sh/uv/getting-started/installation/>

## 2. Tạo environment và cài package

Chạy tại thư mục gốc của repository:

```powershell
uv python install 3.12
uv sync --python 3.12
```

`uv sync` tự tạo `.venv`, đọc `pyproject.toml` và cài các package chính:

- `torch`: chạy và fine-tune model.
- `gliner`: NER inference/training.
- `huggingface-hub`: tải checkpoint bằng CLI `hf`.
- `rapidfuzz`, `pandas`: xử lý ontology và synthetic data.
- `requests`, `python-dotenv`: các script generation/validation có API.

Không cần activate virtual environment khi dùng `uv run`. Kiểm tra cài đặt:

```powershell
uv run python -c "import torch, gliner; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Package tùy chọn

Chỉ cài dense retrieval:

```powershell
uv sync --extra retrieval
```

Cài mọi dependency, gồm dense retrieval và Google Gemini generation:

```powershell
uv sync --all-extras
```

Pipeline CLI mặc định không cần hai extra này.

### PyTorch GPU

Trên máy NVIDIA đã cài driver, để `uv` tự chọn PyTorch backend tương thích:

```powershell
uv sync --python 3.12
uv pip install --reinstall torch --torch-backend=auto
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Nếu kết quả đầu tiên là `False`, chạy với `--device cpu`. Không dùng `--bf16` trên
CPU. Xem hướng dẫn PyTorch của uv tại
<https://docs.astral.sh/uv/guides/integration/pytorch/>.

## 3. Model cần tải

### Bắt buộc: GLiNER multilingual

| Mục đích | Hugging Face ID | Thư mục local | Bắt buộc |
|---|---|---|---|
| NER baseline và base checkpoint để fine-tune | `urchade/gliner_multi-v2.1` | `models/gliner-base` | Có |
| Model NER sau fine-tune | Không tải; trainer tự sinh | `models/medical-gliner` | Có cho output cuối |

Tải model base khoảng 2.31 GB:

```powershell
uv run hf download urchade/gliner_multi-v2.1 `
  --local-dir models/gliner-base
```

Model: <https://huggingface.co/urchade/gliner_multi-v2.1>

Kiểm tra model đã tải:

```powershell
Get-ChildItem models/gliner-base
```

Không cần token Hugging Face vì model public. Khi nộp/chạy self-hosted, copy cả
thư mục `models/medical-gliner` sang máy inference; không truyền Hugging Face ID.

### Model tùy chọn — chưa được bật trong CLI mặc định

| Thành phần | Hugging Face ID | Khi nào cần |
|---|---|---|
| Dense retrieval | `BAAI/bge-m3` | Muốn semantic retrieval cho ICD/RxNorm |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | Muốn rerank candidate bằng model |
| Assertion/linking verifier | Qwen/Llama instruct local | Muốn kiểm tra rule hoặc disambiguation bằng LLM |

Chỉ tải BGE nếu bạn đã cài extra `retrieval` và tự cấu hình các adapter trong
`src/linking`:

```powershell
uv run hf download BAAI/bge-m3 --local-dir models/bge-m3
uv run hf download BAAI/bge-reranker-v2-m3 `
  --local-dir models/bge-reranker-v2-m3
```

Hai model BGE không làm thay đổi kết quả của `src.run_inference` hiện tại vì CLI
đang dùng linker lexical mặc định.

## 4. Cấu trúc repository và flow dữ liệu

Các thư mục chính:

```text
.
├── scripts/                     # Tạo và kiểm tra synthetic data
│   ├── common.py                # Path, schema, JSON/JSONL, cấu hình Qwen
│   ├── prepare_sources.py       # Raw ontology/ViMedNER → processed + catalogs
│   ├── generate.py              # Catalogs + Qwen → text có marker + CaseSpec
│   └── validate_medical.py      # Căn offset, validate và export dataset
├── src/
│   ├── preprocessing/           # Load document, normalize, quản lý offset
│   ├── ner/                     # GLiNER dataset, training và inference
│   ├── assertion/               # isNegated/isHistorical/isFamily
│   ├── linking/                 # ICD-10/RxNorm candidate và reranking
│   ├── validation_output/       # Kiểm tra và serialize output cuộc thi
│   ├── clinical_pipeline.py     # Ghép NER → assertion → linking
│   ├── train_ner.py             # CLI fine-tune NER
│   └── run_inference.py         # CLI TXT → JSONL + output.zip
├── data/
│   ├── raw/                     # Input của synthetic pipeline
│   ├── synthetic/               # Toàn bộ artifact synthetic
│   └── test_set/                # Input/output inference
├── tests/                       # Unit test
├── models/                      # Checkpoint tải hoặc fine-tune; tạo khi cần
├── .env                         # Cấu hình generation; không commit secret
└── pyproject.toml               # Dependency và cấu hình tool
```

Chi tiết `data/synthetic` sau khi chạy đủ ba stage:

```text
data/
├── raw/
│   ├── ViMedNER.txt
│   ├── DM ICD10-19_8_BYT.xlsx
│   └── rxnorm_data/
│       ├── rxnorm_IN.json
│       ├── rxnorm_BN.json
│       └── ...
├── synthetic/
│   ├── processed/
│   │   ├── vimedner_entities.jsonl
│   │   ├── vimedner_audit.json
│   │   ├── rxnorm_concepts.jsonl
│   │   ├── rxnorm_families.jsonl
│   │   ├── icd10_concepts.jsonl
│   │   ├── unresolved_treatments.jsonl
│   │   └── unresolved_diseases.jsonl
│   ├── catalogs/
│   │   ├── drug_surfaces.jsonl
│   │   ├── disease_surfaces.jsonl
│   │   ├── symptom_surfaces.jsonl
│   │   ├── hard_negatives.jsonl
│   │   ├── lab_tests.jsonl
│   │   ├── assertion_scenarios.json
│   │   ├── document_profiles.json
│   │   └── generation_pool.json
│   ├── generated/
│   │   ├── case_specs.jsonl
│   │   ├── marked_notes.jsonl
│   │   ├── generation_failures.jsonl
│   │   ├── accepted_samples.jsonl
│   │   └── rejected_samples.jsonl
│   └── final/
│       ├── end_to_end_train.jsonl
│       ├── end_to_end_val.jsonl
│       ├── gliner_train.jsonl
│       ├── gliner_val.jsonl
│       ├── assertion_train.jsonl
│       ├── assertion_val.jsonl
│       └── split_manifest.json
└── test_set/
    ├── input/
    │   ├── 1.txt
    │   └── ...
    ├── predictions.jsonl
    ├── prediction_errors.jsonl
    └── output.zip
```

Flow synthetic:

```text
data/raw
  │
  ▼  scripts.prepare_sources
processed + catalogs
  │
  ▼  scripts.generate + Qwen OpenAI-compatible API
case_specs.jsonl + marked_notes.jsonl
  │
  ▼  scripts.validate_medical
accepted/rejected + final train/val datasets
```

Flow inference:

```text
data/test_set/input/*.txt
  → load + normalize
  → GLiNER NER
  → assertion detection
  → ICD-10/RxNorm linking
  → schema validation
  → predictions.jsonl
  → output.zip/output/<note_id>.json
```

## 5. Kiểm tra project

Chạy unit test:

```powershell
uv run python -m pytest
```

GitHub Actions tại `.github/workflows/pytest.yml` tự chạy pytest bằng Python 3.11
và 3.12 sau mỗi lần push hoặc khi tạo/cập nhật pull request. CI chỉ cài dependency
nhóm `dev`, không tải PyTorch hay model vì unit test sử dụng mock model.

Kiểm tra 100 file test mà không load model:

```powershell
uv run python -m src.run_inference --dry-run
```

Kết quả dự kiến:

```json
{
  "num_files": 100,
  "first_note_id": "1",
  "last_note_id": "100"
}
```

## 6. Kiểm tra training data

`src.train_ner` hỗ trợ hai loại input:

- cặp `contents.jsonl` + `labels.jsonl` theo format synthetic split cũ;
- một file GLiNER token-level có `tokenized_text` và `ner`.

Dry-run với cặp content/label:

```powershell
uv run python -m src.train_ner `
  --contents path/to/contents.jsonl `
  --labels path/to/labels.jsonl `
  --model models/gliner-base `
  --output models/medical-gliner `
  --validation-ratio 0.1 `
  --dry-run
```

Dry-run với dataset GLiNER token-level:

```powershell
uv run python -m src.train_ner `
  --train path/to/gliner_tokenized.jsonl `
  --model models/gliner-base `
  --output models/medical-gliner `
  --validation-ratio 0.1 `
  --dry-run
```

Dry-run preprocess, kiểm tra span và chia train/eval nhưng chưa load model.
Sample lỗi từ cặp content/label được loại; khi train thật, chi tiết được ghi vào
`models/medical-gliner/dataset_errors.jsonl`.

Không truyền trực tiếp `data/synthetic/final/gliner_train.jsonl` vào `--train`:
file exporter này đang dùng character offset, trong khi trainer yêu cầu token
offset.

## 7. Fine-tune GLiNER

### NVIDIA GPU

```powershell
uv run python -m src.train_ner `
  --contents path/to/contents.jsonl `
  --labels path/to/labels.jsonl `
  --model models/gliner-base `
  --output models/medical-gliner `
  --max-steps 2000 `
  --batch-size 8 `
  --device cuda `
  --bf16
```

Nếu hết VRAM, giảm `--batch-size` xuống `4`, `2` hoặc `1`. Chỉ dùng `--bf16` khi
GPU hỗ trợ BF16.

### CPU

```powershell
uv run python -m src.train_ner `
  --contents path/to/contents.jsonl `
  --labels path/to/labels.jsonl `
  --model models/gliner-base `
  --output models/medical-gliner `
  --max-steps 2000 `
  --batch-size 2 `
  --device cpu
```

Fine-tune CPU có thể rất chậm. Sau training:

```text
models/medical-gliner/
├── train_dataset.json
├── eval_dataset.json
├── dataset_errors.jsonl
├── training_manifest.json
└── model/checkpoint files
```

## 8. Chạy test set ra output

### Output baseline, chưa fine-tune

Dùng để kiểm tra end-to-end; chất lượng NER chưa phải kết quả cuối:

```powershell
uv run python -m src.run_inference `
  --input data/test_set/input `
  --model models/gliner-base `
  --device cpu `
  --chunk-size 300 `
  --chunk-overlap 50 `
  --output data/test_set/baseline_predictions.jsonl `
  --zip-output data/test_set/baseline_output.zip
```

GLiNER giới hạn chiều dài mỗi lần inference, nên pipeline mặc định chia văn bản
thành chunk 300 từ với 50 từ overlap. Prediction của từng chunk được cộng lại
offset toàn văn bản; entity trùng trong vùng overlap được deduplicate trước khi
ghi output. Có thể điều chỉnh bằng `--chunk-size` và `--chunk-overlap`.

Pipeline cũng bật mặc định rule-based recall cho thuốc. Rule đối chiếu tên và
biệt dược trong `data/drug_mapping_final.json`, nhưng chỉ nhận match trong mục
thuốc hoặc dòng có ngữ cảnh kê đơn như hàm lượng, đường dùng và lịch dùng
(`mg`, `ml`, `po`, `iv`, `bid`, `daily`, `SCDC`, sáng/trưa/chiều/tối). Kết quả
rule được merge với GLiNER rồi deduplicate và linking như bình thường. Dùng
`--disable-drug-rules` nếu cần chạy ablation chỉ với model.

### Output cuối bằng model fine-tune

CPU:

```powershell
uv run python -m src.run_inference `
  --input data/test_set/input `
  --model models/medical-gliner `
  --device cpu `
  --output data/test_set/predictions.jsonl `
  --zip-output data/test_set/output.zip
```

NVIDIA GPU:

```powershell
uv run python -m src.run_inference `
  --input data/test_set/input `
  --model models/medical-gliner `
  --device cuda `
  --output data/test_set/predictions.jsonl `
  --zip-output data/test_set/output.zip
```

Output được tạo tại:

```text
data/test_set/
├── predictions.jsonl
├── output.zip
└── prediction_errors.jsonl
```

`predictions.jsonl` là output trung gian, mỗi dòng chứa một bản ghi. Sau khi
inference xong, pipeline đọc lại file này và tách từng dòng thành một file JSON
riêng bên trong `output.zip`.

Giải nén `output.zip` sẽ có đúng cấu trúc file nộp:

```text
output/
├── 1.json
├── 2.json
├── ...
└── 100.json
```

Mỗi file JSON chứa nhãn của đúng một bản ghi. Ví dụ `output/1.json`:

```json
{"note_id":"1","entities":[{"text":"tăng huyết áp","type":"CHẨN_ĐOÁN","position":[120,133],"assertions":[],"candidates":["I10"]}]}
```

- Tên output lấy từ tên input: `1.txt` → `output/1.json`.
- `position` là `[start, end]`, exclusive-end, trên văn bản TXT gốc.
- File được xử lý theo thứ tự `1.txt`, `2.txt`, ..., `100.txt`.
- Sample lỗi được ghi riêng vào `prediction_errors.jsonl`.
- Kể cả khi một sample inference lỗi, ZIP vẫn có file JSON tương ứng với
  `entities: []`, vì vậy 100 file TXT luôn tạo ra đủ 100 file JSON.
- Nếu có ít nhất một sample lỗi, CLI vẫn giữ prediction thành công nhưng trả exit
  code `1` để pipeline triển khai không bỏ sót lỗi.

Kiểm tra nhanh nội dung ZIP:

```powershell
tar -tf data/test_set/output.zip
Get-Content data/test_set/prediction_errors.jsonl
```

Nếu không có lỗi, ZIP phải có đủ `output/1.json` đến `output/100.json` và file
error rỗng.

Nếu checkpoint truyền vào `--model` chưa tồn tại hoặc không load được, CLI vẫn
tạo đủ các file JSON với `entities: []` và ghi nguyên nhân vào
`prediction_errors.jsonl`. Đây chỉ là output dự phòng để bảo đảm đúng cấu trúc;
muốn có dự đoán thật cần đặt checkpoint tại `models/medical-gliner` hoặc dùng một
model Hugging Face với `--allow-remote-model`.

## 9. Đánh giá baseline trên gold data

```powershell
uv run python -m src.main_pipeline `
  --gold data/validated/validated_pass.jsonl `
  --model models/medical-gliner
```

Lệnh trả metric exact-match và overlap-match theo entity type.

## 10. Tạo lại synthetic data — không bắt buộc để inference

Synthetic pipeline tạo cả văn bản và label. Ba stage phải chạy theo thứ tự:

```text
prepare_sources → generate → validate_medical
```

### 10.1. Input bắt buộc

Đặt raw data tại đúng các đường dẫn:

```text
data/raw/ViMedNER.txt
data/raw/DM ICD10-19_8_BYT.xlsx
data/raw/rxnorm_data/*.json
```

`scripts/common.py` xác định root từ vị trí file nên có thể gọi module khi
working directory là root repository. Khuyến nghị dùng `python -m scripts...`
thay vì chạy từ bên trong thư mục `scripts`.

### 10.2. Cấu hình Qwen trong `.env`

`scripts/common.py` tự nạp `.env` ở root bằng `python-dotenv`. Ba biến bắt buộc:

```env
QWEN_BASE_URL=http://localhost:8000/v1
QWEN_API_KEY=local-key
QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Các biến tùy chọn và giá trị mặc định:

```env
QWEN_TIMEOUT_SECONDS=300
QWEN_MAX_TOKENS=900
QWEN_TEMPERATURE=0.35
QWEN_REPAIR_TEMPERATURE=0.10
QWEN_TOP_P=0.90
```

`QWEN_TIMEOUT_SECONDS`, `QWEN_MAX_TOKENS` và `QWEN_TOP_P` được dùng trực tiếp
khi gọi API. `generate.py` hiện đặt temperature là `0.45` ở lần sinh đầu và
`0.15` ở các lần repair; hai biến temperature trong config được nạp và validate
nhưng chưa override hai giá trị này.

API phải tương thích OpenAI Chat Completions và phục vụ endpoint:

```text
POST <QWEN_BASE_URL>/chat/completions
```

Với Qwen local qua vLLM, `QWEN_MODEL` phải trùng model ID server đang serve.
Với provider bên ngoài, thay URL, key và model bằng giá trị do provider cung
cấp. Không commit API key.

Kiểm tra `.env` đã được nạp mà không in key:

```powershell
uv run python -c "from scripts.common import get_qwen_config; c=get_qwen_config(); print(c.base_url); print(c.model); print('API key configured:', bool(c.api_key))"
```

Kiểm tra một request trước khi generate:

```powershell
uv run python -c "from scripts.generate import call_qwen; print(call_qwen('Bạn là trợ lý.', 'Chỉ trả lời đúng một từ: OK', temperature=0.1))"
```

### 10.3. Stage 1 — chuẩn bị processed data và catalog

```powershell
uv run python -m scripts.prepare_sources
```

Stage này:

- đọc ViMedNER, RxNorm và ICD-10;
- chuẩn hóa concept/surface;
- tách các treatment chưa resolve và disease chưa resolve để audit;
- tạo catalog thuốc, bệnh, triệu chứng, hard negative, xét nghiệm;
- tạo scenario assertion và document profile.

Các file trong `processed/` và `catalogs/` được mở bằng mode ghi mới, vì vậy
chạy lại stage này sẽ ghi đè artifact cũ.

### 10.4. Stage 2 — sinh text và label spec

Sinh mới 50 case:

```powershell
uv run python -m scripts.generate --num-samples 50 --seed 42
```

Mặc định `--num-samples=50`, `--seed=42`. Khi không có `--resume`,
`generate.py` xóa ba file generation cũ trước khi chạy:

```text
case_specs.jsonl
marked_notes.jsonl
generation_failures.jsonl
```

Sinh thêm và giữ dữ liệu cũ:

```powershell
uv run python -m scripts.generate `
  --num-samples 50 `
  --seed 43 `
  --resume
```

Nên đổi seed giữa các batch. Với `--resume`, record mới được append vào cuối
JSONL; stage không deduplicate dữ liệu cũ.

Vai trò các file:

- `case_specs.jsonl`: label/spec nguồn gồm entity type, assertion, candidate
  ICD-10/RxNorm, cấu trúc section và scenario.
- `marked_notes.jsonl`: text do LLM sinh, entity được bao bởi marker
  `[[E0]]...[[/E0]]`; liên kết với spec bằng `case_id`.
- `generation_failures.jsonl`: một record cho mỗi case lỗi, gồm `case_id` và
  thông báo lỗi.

`generate.py` bắt exception theo từng case để batch tiếp tục chạy. Vì vậy process
có thể kết thúc mà vẫn có `failed > 0`; luôn kiểm tra file failure thay vì chỉ
dựa vào exit code.

Kiểm tra nhanh số record:

```powershell
Get-Content data/synthetic/generated/case_specs.jsonl |
  Measure-Object -Line
Get-Content data/synthetic/generated/marked_notes.jsonl |
  Measure-Object -Line
Get-Content data/synthetic/generated/generation_failures.jsonl
```

### 10.5. Stage 3 — validate, căn offset và export label

```powershell
uv run python -m scripts.validate_medical
```

Stage này:

1. Ghép `case_specs.jsonl` và `marked_notes.jsonl` bằng `case_id`.
2. Kiểm tra marker đủ, đúng một lần và không bị LLM sửa nội dung.
3. Bỏ marker và tính character offset `[start, end)` trên clean text.
4. Kiểm tra entity type, assertion và candidate có trong ontology.
5. Ghi accepted/rejected.
6. Chia train/validation theo hash ổn định.
7. Export end-to-end, NER và assertion dataset.

Các output `accepted_samples.jsonl`, `rejected_samples.jsonl` và toàn bộ
`data/synthetic/final/` được ghi lại từ đầu mỗi lần validate.

Schema end-to-end rút gọn:

```json
{
  "note_id": "20260723_120000_000000",
  "text": "Bệnh nhân không sốt.",
  "entities": [
    {
      "text": "sốt",
      "type": "TRIỆU_CHỨNG",
      "position": [16, 19],
      "assertions": ["isNegated"]
    }
  ]
}
```

`gliner_*.jsonl` chứa `text` và entity character span dạng
`{"start": ..., "end": ..., "label": ...}`. `assertion_*.jsonl` chứa context có
marker `<E0>...</E0>` cùng assertion của từng entity.

Lưu ý: CLI `src.train_ner --train` hiện nhận format GLiNER token-level
`tokenized_text` + `ner`, còn `gliner_*.jsonl` của synthetic exporter là
character-span. Cần bước chuyển đổi sang token-level trước khi truyền trực tiếp
cho `--train`; không đổi tên file rồi train ngay.

### 10.6. Chạy toàn bộ synthetic flow

Sinh mới hoàn toàn:

```powershell
uv run python -m scripts.prepare_sources
uv run python -m scripts.generate --num-samples 100 --seed 42
uv run python -m scripts.validate_medical
```

Thêm batch rồi rebuild toàn bộ output validated:

```powershell
uv run python -m scripts.generate `
  --num-samples 100 `
  --seed 43 `
  --resume
uv run python -m scripts.validate_medical
```

Generation có thể gọi API bên ngoài tùy `QWEN_BASE_URL`. Inference trong
`src.run_inference` vẫn self-hosted và không sử dụng API generation.

## 11. Flow ngắn nhất từ clone đến inference output

```powershell
uv python install 3.12
uv sync --python 3.12

uv run hf download urchade/gliner_multi-v2.1 `
  --local-dir models/gliner-base

uv run python -m pytest

uv run python -m src.run_inference --dry-run

uv run python -m src.run_inference `
  --input data/test_set/input `
  --model models/gliner-base `
  --device cpu `
  --output data/test_set/baseline_predictions.jsonl `
  --zip-output data/test_set/baseline_output.zip
```

Flow này tạo baseline output để kiểm tra toàn bộ pipeline mà chưa cần fine-tune.
Để tạo output cuối, fine-tune theo mục 7 rồi thay `models/gliner-base` bằng
`models/medical-gliner`. Trên NVIDIA GPU có thể đổi `--device cpu` thành
`--device cuda`.

## 12. Lỗi thường gặp

### `Local GLiNER checkpoint not found`

Chưa tải model hoặc truyền sai đường dẫn. Chạy lại bước tải model và kiểm tra
`models/gliner-base` hoặc `models/medical-gliner`.

### `No module named gliner`

Command đang chạy ngoài environment của uv. Dùng `uv run ...` hoặc chạy lại:

```powershell
uv sync --python 3.12
```

### CUDA hết bộ nhớ

Giảm batch size:

```text
--batch-size 8 → 4 → 2 → 1
```

### Synthetic generation báo `failed` cho mọi case

Đọc lỗi thật:

```powershell
Get-Content data/synthetic/generated/generation_failures.jsonl
```

Các nguyên nhân thường gặp:

- `Thiếu biến môi trường ...`: tên biến trong `.env` không đúng hoặc giá trị
  rỗng.
- `NameResolutionError`: `QWEN_BASE_URL` vẫn là placeholder hoặc hostname sai.
- HTTP `401`: API key sai, hết hiệu lực hoặc thuộc provider khác.
- HTTP `404`: endpoint hoặc model slug không tồn tại; model free có thể đã bị
  provider gỡ.
- `marker`/`Render section thất bại`: LLM sửa, bỏ hoặc lặp marker dù API request
  thành công; script tự repair tối đa ba lần.

Thử một request nhỏ theo mục 10.2, sau đó generate một case trước:

```powershell
uv run python -m scripts.generate --num-samples 1 --seed 42
```

Chỉ tăng `--num-samples` sau khi thấy `success=1, failed=0`.

### `case_specs.jsonl` hoặc `marked_notes.jsonl` không tồn tại

Không có case nào generate thành công. `generate.py` chỉ ghi hai file này sau
khi LLM trả text vượt qua kiểm tra marker. Xem `generation_failures.jsonl` và
không chạy validate cho tới khi hai file có cùng số record.

### PowerShell hiển thị tiếng Việt bị lỗi

File vẫn được Python đọc UTF-8. Có thể đổi console sang UTF-8:

```powershell
chcp 65001
```
