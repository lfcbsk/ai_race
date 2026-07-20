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

## 4. Cấu trúc dữ liệu

```text
data/
├── generated/
│   ├── contents.jsonl
│   └── labels.jsonl
├── validated/
│   └── validated_pass.jsonl
├── test_set/
│   └── input/
│       ├── 1.txt
│       └── ...
├── icd_mapping_final.json
└── drug_mapping_final.json
```

Synthetic content và label luôn lưu riêng. Chúng chỉ được ghép thành
`MedicalDocument` trong RAM khi chuẩn bị fine-tuning.

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

Bước này preprocess synthetic data, map raw offset sang normalized offset, kiểm
tra entity text và chuyển sang format GLiNER. Chưa load model và chưa train:

```powershell
uv run python -m src.train_ner `
  --contents data/generated/contents.jsonl `
  --labels data/generated/labels.jsonl `
  --model models/gliner-base `
  --output models/medical-gliner `
  --validation-ratio 0.1 `
  --dry-run
```

Với dữ liệu hiện tại, kết quả dự kiến:

```json
{
  "train_samples": 194,
  "eval_samples": 22,
  "dataset_errors": 1
}
```

Sample lỗi bị loại khỏi training; khi train thật, chi tiết được ghi vào
`models/medical-gliner/dataset_errors.jsonl`.

## 7. Fine-tune GLiNER

### NVIDIA GPU

```powershell
uv run python -m src.train_ner `
  --contents data/generated/contents.jsonl `
  --labels data/generated/labels.jsonl `
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
  --contents data/generated/contents.jsonl `
  --labels data/generated/labels.jsonl `
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

Các file synthetic hiện đã có trong `data/generated`. Chỉ chạy phần này khi có
raw seed/ontology và muốn sinh lại dữ liệu:

```powershell
uv run python build_synthetic_data.py survey
uv run python build_synthetic_data.py build_pool
uv run python build_synthetic_data.py generate 100
```

Generation và `validate_medical.py` có thể gọi dịch vụ bên ngoài tùy cấu hình
`.env`; chúng không nằm trong inference self-hosted.

## 11. Flow ngắn nhất từ clone đến output

```powershell
uv python install 3.12
uv sync --python 3.12

uv run hf download urchade/gliner_multi-v2.1 `
  --local-dir models/gliner-base

uv run python -m src.train_ner `
  --contents data/generated/contents.jsonl `
  --labels data/generated/labels.jsonl `
  --model models/gliner-base `
  --output models/medical-gliner `
  --max-steps 2000 `
  --batch-size 8 `
  --device cuda `
  --bf16

uv run python -m src.run_inference `
  --input data/test_set/input `
  --model models/medical-gliner `
  --device cuda `
  --output data/test_set/predictions.jsonl `
  --zip-output data/test_set/output.zip
```

Trên máy CPU, thay `--device cuda` bằng `--device cpu`, bỏ `--bf16` và giảm
`--batch-size` xuống `2`.

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

### PowerShell hiển thị tiếng Việt bị lỗi

File vẫn được Python đọc UTF-8. Có thể đổi console sang UTF-8:

```powershell
chcp 65001
```
