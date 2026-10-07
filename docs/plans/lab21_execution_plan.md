# Kế Hoạch Triển Khai Thực Nghiệm (Execution Plan) — Lab 21

> **Chiến lược điểm số**: Phương án 2 (Mục tiêu 105/100 Điểm)  
> **Môi trường kết hợp**: Local PC (Windows CPU) + Google Colab (Tesla T4 GPU 16 GB)  
> **Thời gian ước tính**: ~15 phút chuẩn bị Local + ~110–130 phút GPU Colab + ~30 phút viết Report  

---

## 1. Phân Bổ Môi Trường & Luồng Dữ Liệu (Environment Workflow)

Do máy cục bộ không có card đồ hoạ rời NVIDIA (`CUDA available: False`), luồng làm việc được phân tách khoa học:

```
┌────────────────────────────────┐            ┌────────────────────────────────┐
│      MÁY CỤC BỘ (LOCAL CPU)    │            │     GOOGLE COLAB (T4 GPU)      │
│  - Clone repo, tạo venv cpu    │            │  - Mở Lab21_RUN_ALL.ipynb      │
│  - Chạy smoke test, NB1 preview│            │  - Chạy NB1 -> NB2 (Đóng băng) │
│  - Viết dàn ý REPORT.md        │   Git Push │  - Chạy NB3 -> NB4 (Train 4 run)│
│                                ├───────────►│  - Chạy NB5 (Eval 4 nhóm)      │
│                                │            │  - Chạy NB6 (Merge & Hot-swap) │
│                                │◄───────────┤  - Push adapter lên HF Hub     │
│  - Nhận results/ & adapters/   │   Git Pull │  - Chạy verify.py kiểm tra     │
│  - Hoàn thiện Report chính thức│   hoặc Zip │                                │
│  - make verify lần cuối        │   Artifacts│                                │
└────────────────────────────────┘            └────────────────────────────────┘
```

---

## 2. Các Giai Đoạn Triển Khai Chi Tiết

### Giai Đoạn 1: Chuẩn Bị & Smoke Test Cục Bộ (Máy Local, ~10 phút)
* **Mục tiêu**: Đảm bảo toàn bộ thư viện, script kiểm tra và cấu trúc thư mục sẵn sàng trước khi nạp GPU.
* **Các bước**:
  1. Tạo môi trường ảo Python 3.12 trên Windows:
     ```powershell
     python -m venv .venv
     .venv\Scripts\pip install -r requirements-cpu.txt
     ```
  2. Chạy smoke test kiểm tra tính toàn vẹn của mã nguồn:
     ```powershell
     .venv\Scripts\python scripts/verify.py --smoke
     ```
  3. Khởi tạo file `.env` từ `.env.example`:
     ```powershell
     copy .env.example .env
     ```
     Đảm bảo trong `.env` có:
     ```ini
     COMPUTE_TIER=T4
     MASK_MODE=assistant-only
     EPOCHS=2
     ```
     *(Lưu ý: Không đặt `EVAL_LIMIT` để khi nộp bài không bị gatekeeper từ chối).*
  4. Đẩy toàn bộ thay đổi và branch làm việc lên GitHub repo cá nhân.

---

### Giai Đoạn 2: Huấn Luyện & Đánh Giá Trên Colab T4 GPU (~110–130 phút)
* **Mục tiêu**: Hoàn thành toàn bộ Core Pipeline (NB1–NB5) và Thử thách Bonus (NB6, HF Hub).
* **Môi trường Colab**:
  - Truy cập: `colab/Lab21_RUN_ALL.ipynb` (hoặc mở trực tiếp trên Colab).
  - Runtime: Chọn **T4 GPU** (Python 3).
  - Lưu ý phòng ngừa F-05: Chỉ duy trì duy nhất một phiên Colab GPU (vào *Thời gian chạy > Quản lý phiên* để huỷ các phiên cũ nếu có).
  - Lưu ý phòng ngừa F-19: Tải lại trang (Reload) nếu repo vừa có cập nhật mới.

#### Chi tiết các ô lệnh trong Colab:
1. **Ô 1 — Thiết lập môi trường**:
   - Clone repo từ GitHub cá nhân (hoặc pull code mới nhất).
   - Chạy `pip install -q -r requirements.txt`.
   - Kiểm tra GPU: Xác nhận `GPU: Tesla T4`, `VRAM: 14.6 GB`, `torchao >= 0.16`.
2. **Ô 2 — Smoke Verification**:
   - Chạy `python scripts/verify.py --smoke` $\rightarrow$ Xác nhận tất cả test đều Pass.
3. **Ô 3 — Chạy Core Pipeline (NB1 $\rightarrow$ NB5)**:
   - Cấu hình:
     ```python
     COMPUTE_TIER = "T4"
     EVAL_LIMIT = ""  # Để rỗng để chạy trọn vẹn 50 mẫu đánh giá cho bài nộp chính thức!
     STAGES = "nb1 nb2 nb3 nb4 nb5"
     ```
   - Chạy kịch bản điều phối:
     ```bash
     !python scripts/colab_run.py {STAGES}
     ```
   - **Tiến trình dự kiến**:
     * `NB1`: Xây mask proof, kiểm tra `<think>`, đo p95 $\rightarrow$ `max_length = 1024` (~1 phút).
     * `NB2`: Nạp base `Qwen3.5-4B`, đo Baseline (a) và Baseline (b) trên 50 mẫu target + 15 mẫu regression. Đóng băng vào `results/baselines_frozen.json` (~20 phút).
     * `NB3`: Huấn luyện `correct` qua 30 steps (2 epochs, batch hiệu dụng 16). Lưu adapter vào `adapters/correct/` (~20 phút).
     * `NB4`: Huấn luyện 3 cấu hình sai `attn_only` (matched rank r≈283), `wrong_lr` (lr=2e-5), `qlora` (4-bit), mỗi cấu hình đúng 30 steps. Lưu adapters và cập nhật `results/runs.csv` (~50 phút).
     * `NB5`: Chấm điểm bản fine-tune `correct` so với Baseline (b), chấm 3 cấu hình sai trên tác vụ target, xuất `verdict.json`, `autopsy.json`, `qualitative.json` (~20 phút).
4. **Ô Bổ sung — Chạy NB6 (Bonus B1: +3 Điểm)**:
   - Chạy lệnh:
     ```bash
     !python notebooks/06_merge_and_serve.py
     ```
   - Xác nhận:
     * Model merge thành công sang `adapters/merged/`.
     * `results/merge_check.json` ghi nhận delta $\ge -0.01$.
     * Hot-swap các adapter `correct`, `attn_only`, `qlora` diễn ra suôn sẻ.
5. **Ô Bổ sung — Đẩy Adapter lên Hugging Face Hub (Bonus B5: +2 Điểm)**:
   - Nạp token Hugging Face (Write permission):
     ```python
     from huggingface_hub import login
     login(token="<YOUR_HF_TOKEN>")

     from peft import PeftModel
     from transformers import AutoTokenizer

     repo_id = "<your-username>/lab21-qwen35-triage-vi"
     model = PeftModel.from_pretrained(base_model, "adapters/correct")
     model.push_to_hub(repo_id)
     tok.push_to_hub(repo_id)
     print(f"Uploaded successfully to: https://huggingface.co/{repo_id}")
     ```
6. **Ô 4 — Kiểm tra Gatekeeper Tổng Thể**:
   - Chạy:
     ```bash
     !python scripts/verify.py
     ```
   - Đảm bảo các chỉ số quan trọng đều hợp lệ:
     * `mask proof asserts`: PASS
     * `supervised fraction`: PASS ($< 95\%$)
     * `full eval set used`: PASS (50 items)
     * `baseline (b) beats (a)`: PASS
     * `all runs share ONE step budget`: PASS (30 steps)
     * `attn_only is a FAIR contrast`: PASS ($< 5\%$ chênh lệch)
     * `verdict recorded`: PASS

---

### Giai Đoạn 3: Thu Thập Dữ Liệu & Viết Báo Cáo (~30–45 phút)
* **Mục tiêu**: Tải toàn bộ kết quả thực nghiệm về máy và hoàn thiện báo cáo phân tích khoa học chất lượng cao theo rubric.
* **Các bước**:
  1. Nén và tải thư mục `results/` cùng `adapters/correct/` từ Colab về máy:
     ```python
     !zip -r results_and_adapters.zip results/ adapters/correct/
     ```
  2. Giải nén vào thư mục dự án cục bộ.
  3. Viết file `submission/REPORT.md`:
     - Điền chính xác các tham số thực tế từ `results/*.json` và `results/runs.csv`.
     - Phân tích 3 câu hỏi sâu của Mục 4:
       * **Rank vs Vị trí**: So sánh `correct` vs `attn_only` (tại sao tăng rank lên 283 ở attention vẫn không bù được việc bỏ qua MLP/DeltaNet layers).
       * **Ý nghĩa của LR**: So sánh `correct` vs `wrong_lr` (tại sao LoRA cần LR lớn hơn $10\times$ so với full-FT).
       * **Đánh đổi của QLoRA**: So sánh VRAM tiết kiệm được và chi phí giảm sút độ chính xác.
     - Phân tích Phán quyết Mục 5:
       * Diễn giải nguyên nhân nhân quả nếu phán quyết là `FAILED` (ví dụ do hiện tượng quên thảm hoạ trên câu hỏi phổ thông khi không có dữ liệu replay).
     - Trình bày bảng định tính Mục 6:
       * Liệt kê 5 mẫu từ `results/qualitative.json` (chọn đúng $\ge 2$ ca fine-tune thua).
     - Kết luận $\ge 150$ từ và 3 bài học sâu sắc mang tính phản tư cá nhân.
  4. Tạo file `LINKS.md` chứa:
     - Link GitHub Repository.
     - Link Hugging Face Hub Adapter.
  5. Chạy kiểm tra gatekeeper cục bộ:
     ```powershell
     .venv\Scripts\python scripts/verify.py
     ```
     Xác nhận in ra thông báo: `Ready to submit.`

---

## 3. Kế Hoạch Ứng Phó Rủi Ro (Contingency Plans)

| Tình huống phát sinh | Cơ chế xử lý đã thiết kế |
|---|---|
| **Colab bị ngắt kết nối giữa chừng ở NB4** | Tận dụng cơ chế resume (F-24): các adapter đã huấn luyện xong (`correct`, `attn_only`, ...) không bị train lại. Khi kết nối lại chỉ cần chạy `ONLY=qlora python scripts/colab_run.py nb4` để tiếp tục. |
| **Lỗi TRL cast sang bf16 trên T4 ở run `qlora`** | Đã được xử lý bởi hàm `train.align_trainable_precision` tự động recast về `fp32` (F-23). |
| **Colab hết thời lượng GPU Free** | Đẩy kết quả hiện tại lên Git; mở session dự phòng trên Kaggle Notebooks (chọn GPU T4 $\times 1$) để chạy tiếp. |
| **Kết quả NB5 ra FAILED** | **Không sửa đổi dữ liệu hay làm yếu prompt (b)**! Giữ nguyên kết quả và viết phần phân tích học thuật xuất sắc vào báo cáo (Rubric quy định FAILED được chấm điểm trọn vẹn nếu diễn giải đúng). |
