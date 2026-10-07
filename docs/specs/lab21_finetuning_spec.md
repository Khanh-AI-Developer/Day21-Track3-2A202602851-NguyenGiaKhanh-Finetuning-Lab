# Đặc Tả Kỹ Thuật (Specification) — Lab 21 Fine-tuning LLMs

> **Mã môn học**: AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn  
> **Mục tiêu điểm số**: Phương án 2 — **105/100 Điểm** (100 điểm Core + 3 điểm Thưởng B1 + 2 điểm Thưởng B5)  
> **Định dạng nộp bài**: Option B (GitHub Repository + Hugging Face Hub Adapter Link)  

---

## 1. Mục Tiêu Dự Án & Tiêu Chuẩn Đạt Chuẩn (Success Criteria)

Dự án này thực hiện quy trình fine-tuning mô hình ngôn ngữ lớn (LLM) mã nguồn mở bằng LoRA, thực hiện thí nghiệm đối chứng có kiểm soát để chứng minh:
1. **Loss mask chính xác 100%**: Chỉ tính loss trên token câu trả lời của trợ lý ảo (Assistant), tuyệt đối không tính loss trên prompt/câu hỏi người dùng (`supervised_fraction < 0.95`).
2. **Đóng băng mốc đánh giá trung thực trước khi huấn luyện**: Baseline (b) (Prompt tối ưu + few-shot) phải vượt trội so với Baseline (a) (Prompt ngây thơ).
3. **Thí nghiệm đối chứng công bằng (Fair Comparison)**:
   - Cả 4 cấu hình (`correct`, `attn_only`, `wrong_lr`, `qlora`) dùng chung một ngân sách số bước huấn luyện (`max_steps`).
   - Cấu hình `attn_only` phải sử dụng rank khớp ngân sách tham số (`matched_rank()`, sai số tham số < 5% so với `correct`), loại bỏ thiên lệch về dung lượng tham số.
   - Mỗi cấu hình chỉ thay đổi duy nhất một biến độc lập.
4. **Đánh giá đa chiều & Liêm chính học thuật**:
   - Đánh giá trên tác vụ đích (Target field accuracy) ở NB5 thay vì dùng mất mát huấn luyện (`final_loss`) ở NB4 làm thước đo.
   - Cổng hồi quy 4 nhóm: Target, Regression (15 câu phổ quát kiểm tra suy giảm năng lực), Format (100% JSON hợp lệ), Latency.
   - Thu thập ≥ 5 ví dụ định tính, trong đó bắt buộc có ≥ 2 trường hợp fine-tune thua để chứng minh tính trung thực khoa học.
5. **Bonus B1 (Merge & Serve)**: Merge adapter vào base weights với độ suy giảm $\Delta \ge -0.01$, hỗ trợ hoán đổi nóng (hot-swap) ≥ 2 adapter trên cùng một base model nạp trong VRAM.
6. **Bonus B5 (Hugging Face Hub)**: Đẩy adapter `adapters/correct/` lên Hugging Face Hub công khai.

---

## 2. Thông Số Kiến Trúc & Cấu Hình Thực Nghiệm

### 2.1. Lựa chọn Mô hình & Phần cứng
* **Hardware Tier**: `T4` (Google Colab Free Tesla T4 14.6 GB VRAM usable).
* **Môi trường cục bộ**: Windows CPU (Dùng cho kiểm tra mã nguồn, chạy unit tests, phân tích dữ liệu, soạn thảo báo cáo).
* **Base Model**: `unsloth/Qwen3.5-4B` (Định dạng mặc định cho tier T4) / fallback `Qwen/Qwen3.5-0.8B` nếu chạy kiểm thử CPU.
* **Precision**: `fp16` (Do GPU Tesla T4 kiến trúc Turing `sm_75` không hỗ trợ phần cứng cho `bfloat16`; hệ thống sử dụng `torch.float16` kết hợp `GradScaler` để chống underflow).

### 2.2. Bộ Dữ Liệu Thực Nghiệm (Dataset)
* **Tác vụ**: Phân loại ticket Chăm sóc khách hàng tiếng Việt thành cấu trúc JSON 4 trường:
  - `intent`: Mục đích ticket (`khieu_nai`, `hoi_thong_tin`, `doi_tra`, `ho_tro_ky_thuat`, `huy_dich_vu`, `khac`).
  - `urgency`: Mức độ khẩn cấp (`thap`, `trung_binh`, `cao`, `khan_cap`).
  - `product`: Danh mục sản phẩm/dịch vụ (`dien_thoai`, `laptop`, `gia_dung`, `thoi_trang`, `my_pham`, `dich_vu_so`, `khac`).
  - `sentiment`: Sắc thái tình cảm (`tieu_cuc`, `trung_tinh`, `tich_cuc`).
* **Kích thước**: 250 mẫu (`data/train_seed.jsonl`).
* **Phân chia (Split)**: 90% Train (225 mẫu) / 10% Validation (25 mẫu), cố định `seed = 42`.
* **Tập Đánh giá**:
  - `data/eval_target.jsonl`: 50 mẫu đánh giá tác vụ phân loại JSON.
  - `data/eval_regression.jsonl`: 15 câu hỏi kiến thức phổ quát tiếng Việt kiểm tra quên thảm hoạ (Catastrophic Forgetting).

---

## 3. Đặc Tả Chi Tiết Từng Khối Thực Nghiệm (NB1 – NB6)

### NB1: Dữ Liệu, Chat Template & Loss Mask
* **Mục tiêu**: Chứng minh loss mask chỉ phủ đúng câu trả lời.
* **Các cơ chế kỹ thuật**:
  - Khắc phục `TemplateNotPrefixStable` của Qwen3.5: Sử dụng character-span offset mapping (`return_offsets_mapping=True`) thay vì so sánh danh sách token.
  - Kiểm tra khối `<think>`: Xác nhận chat template không xoá mất khối suy luận.
  - Thống kê độ dài token: Đo phân vị p95 của toàn bộ dữ liệu huấn luyện để thiết lập `max_length` tối ưu (tránh padding lãng phí hoặc cắt cụt nhãn).
* **Artefacts bắt buộc**:
  - `results/mask_proof.json` (`answer_is_supervised = True`, `question_is_masked = True`).
  - `results/template_check.json`.
  - `results/token_stats.json`.
  - `data/split/train.jsonl` và `data/split/val.jsonl`.

### NB2: Đóng Băng Ba Baseline Trước Khi Train
* **Mục tiêu**: Đo lường và đóng băng mốc sàn (a) và mốc đích (b) trước khi can thiệp trọng số.
* **Định nghĩa baseline**:
  - Baseline (a): Base model + `NAIVE_PROMPT` ("Phân loại ticket sau.").
  - Baseline (b): Base model + `OPTIMIZED_PROMPT` (Mô tả chi tiết 4 trường + few-shot mẫu JSON chuẩn).
* **Ràng buộc nghiêm ngặt**:
  - Baseline (b) phải có độ chính xác tác vụ cao hơn Baseline (a) (`(b) > (a)`).
  - Đóng băng checksum SHA-256 của `OPTIMIZED_PROMPT` và tập eval.
  - Không được dùng cờ `EVAL_LIMIT` khi tạo kết quả chính thức.
* **Artefacts bắt buộc**:
  - `results/baselines_frozen.json`.

### NB3: Huấn Luyện Cấu Hình Chuẩn ("LoRA Without Regret")
* **Mục tiêu**: Huấn luyện cấu hình tối ưu theo khuyến nghị công nghệ 2026.
* **Các tham số cấu hình**:
  - `target_modules`: Toàn bộ các lớp linear trong text decoder (12 module suffixes trên Qwen3.5: bao gồm cả projection của attention và Gated DeltaNet linear-attention; loại trừ vision encoder).
  - `r = 16`, `lora_alpha = 32` ($\alpha = 2r$).
  - `learning_rate = 2e-4` (Thang $\approx 10\times$ full fine-tuning).
  - `per_device_train_batch_size = 1`, `gradient_accumulation_steps = 16` $\rightarrow$ Batch hiệu dụng = 16 ($< 32$).
  - Pre-tokenized dataset với nhãn đã verified ở NB1 (tránh lỗi TRL gán mask rỗng).
  - Warmup: 10% tổng số optimizer steps (`warmup_steps = 3`).
* **Artefacts bắt buộc**:
  - Thư mục trọng số: `adapters/correct/`.
  - Dòng kết quả `correct` trong `results/runs.csv`.

### NB4: Giải Phẫu Ba Cấu Hình Sai (Misconfiguration Autopsy)
* **Mục tiêu**: Chạy 3 cấu hình đối chứng, mỗi cấu hình thay đổi đúng một biến so với `correct`, cùng số bước huấn luyện `max_steps`:
  1. `attn_only`: Chỉ gắn LoRA vào `q_proj, v_proj`. Rank được nâng tự động bằng `matched_rank()` lên $r \approx 283$ để sai lệch tổng tham số huấn luyện $< 5\%$.
  2. `wrong_lr`: Gắn vào `text-linear`, $r=16$, nhưng dùng learning rate thang full fine-tune ($2\times 10^{-5}$, giảm $10\times$).
  3. `qlora`: Chuyển base model sang nạp lượng tử 4-bit (`BitsAndBytesConfig`, NF4, double quant). Trainable LoRA weights được recast về `fp32` để tương thích `GradScaler`.
* **Artefacts bắt buộc**:
  - Các adapter: `adapters/attn_only/`, `adapters/wrong_lr/`, `adapters/qlora/`.
  - 3 dòng đối chứng bổ sung trong `results/runs.csv`.

### NB5: Đánh Giá Bốn Nhóm & Phán Quyết (Evaluation & Verdict)
* **Mục tiêu**: Đánh giá bản fine-tune `correct` so với mốc Baseline (b), đồng thời chấm điểm 3 cấu hình sai trên tác vụ đích.
* **Bốn nhóm chỉ số**:
  - `target`: Độ chính xác 4 trường của JSON so với nhãn chuẩn.
  - `regression`: Tỷ lệ từ khóa đúng trên 15 câu hỏi phổ quát (đo lường suy giảm nhận thức).
  - `format`: Tỷ lệ sinh JSON bóc tách thành công và đủ 4 khóa.
  - `latency`: Thời gian suy luận trung bình (ms/mẫu, greedy decoding).
* **Quy tắc phân định**:
  - Xếp hạng 4 cấu hình bằng điểm `target`, không xếp hạng bằng `final_loss`.
  - Phán quyết: `PASSED` nếu `target(ft) > target(b)` và `regression(ft) >= regression(b) - 0.02`. Nếu `FAILED`, phân tích nguyên nhân khoa học trung thực trong báo cáo.
  - Thu thập 5 trường hợp định tính: $\ge 2$ ca fine-tune thắng và $\ge 2$ ca fine-tune thua.
* **Artefacts bắt buộc**:
  - `results/verdict.json`.
  - `results/autopsy.json`.
  - `results/qualitative.json`.

### NB6: Merge Checkpoint & Hot-Swap (Bonus B1: +3 Điểm)
* **Mục tiêu**:
  - Merge adapter `correct` vào mô hình gốc bằng `model.merge_and_unload()`.
  - Đảm bảo điểm sau merge không tụt quá ngưỡng dung sai: $\text{score}_{\text{after}} - \text{score}_{\text{before}} \ge -0.01$.
  - Trải nghiệm hot-swap nạp đồng thời các adapter `correct`, `attn_only`, `qlora` trên một base model duy nhất.
* **Artefacts bắt buộc**:
  - `results/merge_check.json`.
  - Trọng số đã merge: `adapters/merged/`.

### Bonus B5: Đẩy Lên Hugging Face Hub (+2 Điểm)
* **Mục tiêu**: Đẩy thư mục `adapters/correct/` lên Hugging Face Hub công khai.
* **Artefacts bắt buộc**:
  - File `LINKS.md` chứa đường dẫn Hugging Face model adapter và GitHub repo.
  - Đường dẫn được tích hợp vào báo cáo `submission/REPORT.md`.

---

## 4. Tiêu Chuẩn Gatekeeper (`scripts/verify.py`)

Trước khi đóng gói nộp bài, lệnh `python scripts/verify.py` phải trả về mã thoát `0` và in ra `Ready to submit.`:
- [x] Không còn placeholder (`<điền>`, `<paste>`, `<0.xx>`) trong `submission/REPORT.md`.
- [x] Số lượng từ trong `REPORT.md` $\ge 400$ từ; phần kết luận $\ge 150$ từ.
- [x] Cả hai assert trong `mask_proof.json` đều xanh (`True`), `supervised_fraction < 0.95`.
- [x] `baselines_frozen.json` không bật cờ `smoke_mode` (đo trên toàn bộ 50 mẫu).
- [x] `OPTIMIZED_PROMPT` không bị làm suy yếu để tâng bốc fine-tune (`(b) > (a)`).
- [x] Checksum tập eval nguyên vẹn.
- [x] Cả 4 cấu hình trong `runs.csv` chia sẻ cùng giá trị `max_steps`.
- [x] Số tham số huấn luyện của `attn_only` và `correct` sai lệch $< 5\%$.
- [x] Tất cả file json trong `results/` đầy đủ và đồng bộ.
