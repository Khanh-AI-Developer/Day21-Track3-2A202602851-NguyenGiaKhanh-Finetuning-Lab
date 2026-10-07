# Danh Sách Nhiệm Vụ Chi Tiết (Task Breakdown) — Lab 21 Fine-tuning LLMs

> **Mục tiêu**: Hướng dẫn thực hiện từng bước đạt điểm tối đa (105/100) theo Phương án 2.  
> **Quy ước**: Mỗi task có mục tiêu, câu lệnh thực thi và Tiêu chuẩn nghiệm thu (Acceptance Criteria - AC).  

---

### TASK-01: Thiết lập môi trường ảo CPU & Chạy Smoke Test cục bộ
* **Môi trường**: Máy cục bộ (Windows CPU).
* **Mục tiêu**: Cài đặt phụ thuộc cho môi trường kiểm thử cục bộ và xác minh toàn bộ test case hệ thống.
* **Các bước thực hiện**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\pip install -r requirements-cpu.txt
  .venv\Scripts\python scripts/verify.py --smoke
  ```
* **Acceptance Criteria (AC)**:
  - [x] Thư mục `.venv` được tạo thành công.
  - [x] Lệnh `scripts/verify.py --smoke` thoát với code 0, toàn bộ unit tests đều `PASS`.

---

### TASK-02: Khởi tạo cấu hình `.env` & Đồng bộ GitHub cá nhân
* **Môi trường**: Máy cục bộ (Windows).
* **Mục tiêu**: Cấu hình các biến môi trường chuẩn xác và đẩy toàn bộ mã nguồn lên GitHub.
* **Các bước thực hiện**:
  ```powershell
  copy .env.example .env
  ```
  Kiểm tra file `.env` đảm bảo các dòng:
  ```ini
  COMPUTE_TIER=T4
  MASK_MODE=assistant-only
  EPOCHS=2
  ```
  *(Tuyệt đối không bật `EVAL_LIMIT`)*.
  Đẩy mã lên GitHub:
  ```powershell
  git add .
  git commit -m "docs: add specs, plans, and task breakdowns for Lab 21"
  git push origin main
  ```
* **Acceptance Criteria (AC)**:
  - [x] File `.env` tồn tại và chứa cấu hình chuẩn.
  - [x] GitHub repository cá nhân đã cập nhật commit mới nhất.

---

### TASK-03: Khởi động môi trường Google Colab GPU T4 & Xác minh Smoke
* **Môi trường**: Google Colab (T4 GPU).
* **Mục tiêu**: Khởi động notebook điều phối trên GPU và xác minh môi trường phần cứng.
* **Các bước thực hiện**:
  1. Mở file `colab/Lab21_RUN_ALL.ipynb` trên Colab.
  2. Chọn Runtime: `T4 GPU`.
  3. Chạy Ô 1 (Setup): Clone repo, cài `requirements.txt`.
  4. Chạy Ô 2 (Smoke): Chạy `python scripts/verify.py --smoke`.
* **Acceptance Criteria (AC)**:
  - [ ] GPU hiển thị là `Tesla T4`, dung lượng VRAM `~14.6 GB`.
  - [ ] `torchao >= 0.16` được xác nhận.
  - [ ] Smoke test trên Colab trả về 100% `PASS`.

---

### TASK-04: Thực thi NB1 — Dữ liệu, Chat Template & Bằng chứng Loss Mask
* **Môi trường**: Google Colab (hoặc Local CPU).
* **Mục tiêu**: Xử lý dữ liệu, kiểm tra ChatML template và chứng minh toán học loss mask.
* **Các bước thực hiện**:
  - Chạy `python notebooks/01_data_and_mask.py`.
* **Acceptance Criteria (AC)**:
  - [ ] `results/mask_proof.json` có cả 2 assert đều `true` (`answer_is_supervised` và `question_is_masked`).
  - [ ] `supervised_fraction < 0.95`.
  - [ ] `results/template_check.json` ghi nhận chính xác trạng thái khối `<think>`.
  - [ ] `results/token_stats.json` đo được p95 $\le 1024$.
  - [ ] Sinh ra tập chia cố định: `data/split/train.jsonl` (225 mẫu) và `val.jsonl` (25 mẫu).

---

### TASK-05: Thực thi NB2 — Đo & Đóng băng Ba Baseline
* **Môi trường**: Google Colab (T4 GPU).
* **Mục tiêu**: Nạp mô hình gốc `unsloth/Qwen3.5-4B`, đo lường Baseline (a) và Baseline (b) trên toàn bộ 50 mẫu target + 15 mẫu regression trước khi huấn luyện.
* **Các bước thực hiện**:
  - Chạy `python notebooks/02_baselines.py`.
* **Acceptance Criteria (AC)**:
  - [ ] Baseline (b) có độ chính xác target cao hơn Baseline (a) (`b > a`).
  - [ ] `results/baselines_frozen.json` được tạo ra với `smoke_mode = False` và `n_target = 50`.
  - [ ] Checksum SHA của `OPTIMIZED_PROMPT` được ghi nhận.

---

### TASK-06: Thực thi NB3 — Huấn luyện Cấu hình Chuẩn `correct`
* **Môi trường**: Google Colab (T4 GPU).
* **Mục tiêu**: Huấn luyện cấu hình chuẩn "LoRA Without Regret" (all-text-linear, lr=2e-4, batch=1, grad_accum=16, alpha=32).
* **Các bước thực hiện**:
  - Chạy `python notebooks/03_train_correct.py`.
* **Acceptance Criteria (AC)**:
  - [ ] 12 module text decoder được resolve chính xác (không gắn vào vision encoder).
  - [ ] Huấn luyện thành công trọn vẹn 30 steps.
  - [ ] Thư mục `adapters/correct/` chứa `adapter_model.safetensors` và `adapter_config.json`.
  - [ ] Xuất hiện dòng `correct` trong `results/runs.csv` với thông tin loss và VRAM peak.

---

### TASK-07: Thực thi NB4 — Giải phẫu Ba Cấu hình Sai
* **Môi trường**: Google Colab (T4 GPU).
* **Mục tiêu**: Chạy 3 cấu hình đối chứng (`attn_only`, `wrong_lr`, `qlora`), cùng 30 steps:
  * `attn_only`: Giải rank khớp ngân sách tham số qua `matched_rank()` ($r \approx 283$).
  * `wrong_lr`: Thang LR full-FT ($2\times 10^{-5}$).
  * `qlora`: 4-bit quantization + recast trainables về `fp32`.
* **Các bước thực hiện**:
  - Chạy `python notebooks/04_misconfig_autopsy.py`.
* **Acceptance Criteria (AC)**:
  - [ ] Tham số huấn luyện của `attn_only` và `correct` sai lệch $< 5\%$.
  - [ ] Cả 3 run hoàn tất 30 steps và lưu vào `adapters/attn_only/`, `adapters/wrong_lr/`, `adapters/qlora/`.
  - [ ] Bổ sung đủ 3 dòng trong `results/runs.csv`.

---

### TASK-08: Thực thi NB5 — Đánh giá 4 Nhóm & Phán quyết
* **Môi trường**: Google Colab (T4 GPU).
* **Mục tiêu**: Đánh giá bản fine-tune `correct` so với Baseline (b) và chấm điểm 3 cấu hình sai trên tác vụ target.
* **Các bước thực hiện**:
  - Chạy `python notebooks/05_evaluate_and_verdict.py`.
* **Acceptance Criteria (AC)**:
  - [ ] Sinh file `results/verdict.json` với phán quyết (`PASSED` hoặc `FAILED`) trên 4 nhóm chỉ số.
  - [ ] Sinh file `results/autopsy.json` xếp hạng cả 4 cấu hình theo điểm tác vụ đích.
  - [ ] Sinh file `results/qualitative.json` chứa danh sách các mẫu dự đoán.

---

### TASK-09: Thực thi NB6 — Merge Checkpoint & Hot-Swap (Bonus B1: +3 Điểm)
* **Môi trường**: Google Colab (T4 GPU).
* **Mục tiêu**: Thực hiện merge adapter và thử nghiệm hoán đổi adapter động trên một base model.
* **Các bước thực hiện**:
  - Chạy `python notebooks/06_merge_and_serve.py`.
* **Acceptance Criteria (AC)**:
  - [ ] Assert độ suy giảm điểm sau merge thành công: $\Delta \ge -0.01$.
  - [ ] Sinh file `results/merge_check.json`.
  - [ ] Kiểm thử nạp đồng thời và gọi dự đoán qua lại giữa `correct`, `attn_only`, `qlora` thành công.

---

### TASK-10: Đẩy Adapter lên Hugging Face Hub (Bonus B5: +2 Điểm)
* **Môi trường**: Google Colab (hoặc máy cá nhân).
* **Mục tiêu**: Push adapter `adapters/correct/` lên Hugging Face Hub công khai.
* **Các bước thực hiện**:
  - Đăng nhập bằng `huggingface-cli login` hoặc Python API.
  - Dùng `model.push_to_hub("<username>/lab21-qwen35-triage-vi")`.
  - Tạo file `LINKS.md` lưu URL repo và URL Hugging Face Hub.
* **Acceptance Criteria (AC)**:
  - [ ] Adapter truy cập được công khai trên Hugging Face.
  - [ ] File `LINKS.md` có đầy đủ 2 đường dẫn hợp lệ.

---

### TASK-11: Chạy Gatekeeper `verify.py` & Tải Toàn bộ Artefacts về Máy
* **Môi trường**: Google Colab & Local.
* **Mục tiêu**: Đảm bảo 100% các tiêu chí rubric đều pass trên Colab, sau đó tải dữ liệu về máy cục bộ.
* **Các bước thực hiện**:
  - Trên Colab: Chạy `python scripts/verify.py`.
  - Nén thư mục: `zip -r results_and_adapters.zip results/ adapters/correct/ LINKS.md`.
  - Tải về và giải nén vào thư mục dự án cục bộ.
* **Acceptance Criteria (AC)**:
  - [ ] `verify.py` trên Colab in ra `0 failures`.
  - [ ] Thư mục `results/` trên máy cục bộ có đầy đủ toàn bộ file json và `runs.csv`.

---

### TASK-12: Soạn thảo Báo cáo Học thuật `REPORT.md` & Đóng gói Nộp
* **Môi trường**: Máy cục bộ (Windows).
* **Mục tiêu**: Viết báo cáo khoa học xuất sắc, sâu sắc, khớp 100% số liệu và sẵn sàng nộp bài.
* **Các bước thực hiện**:
  - Điền toàn bộ số liệu thực tế từ `results/` vào `submission/REPORT.md`.
  - Viết phần giải thích nguyên nhân nhân quả 3 câu hỏi NB4.
  - Phân tích phán quyết NB5 (kể cả FAILED giải thích rõ cơ chế quên thảm hoạ & replay data).
  - Trích xuất 5 ví dụ định tính (chọn đúng $\ge 2$ ca fine-tune thua).
  - Viết kết luận $\ge 150$ từ và 3 bài học kinh nghiệm sâu sắc.
  - Chạy `python scripts/verify.py` cục bộ kiểm tra lần cuối.
* **Acceptance Criteria (AC)**:
  - [ ] `REPORT.md` không còn bất kỳ placeholder nào (`<điền>`, `<paste>`, ...).
  - [ ] `python scripts/verify.py` in ra thông báo: `Ready to submit.`
