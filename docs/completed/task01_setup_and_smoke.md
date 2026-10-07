# Nghiệm Thu TASK-01: Thiết lập Môi trường ảo CPU & Chạy Smoke Test cục bộ

> **Mã nhiệm vụ**: `TASK-01`  
> **Thời gian hoàn thành**: 2026-10-07  
> **Người thực hiện**: Antigravity & User  
> **Trạng thái**: ✅ **HOÀN THÀNH (DONE)**  

---

## 1. Nội Dung Công Việc Đã Thực Hiện

1. **Khởi tạo môi trường ảo Python**:
   - Sử dụng Python 3.12 trên hệ điều hành Windows để tạo môi trường ảo cục bộ `.venv`:
     ```powershell
     python -m venv .venv
     ```
2. **Cài đặt các gói phụ thuộc CPU**:
   - Nâng cấp `pip` và cài đặt danh sách phụ thuộc từ [requirements-cpu.txt](file:///f:/VinUniversity/Schedule_Study_VinUniversity/Period_Two/Day06/Day21-Track3-2A202602851-NguyenGiaKhanh-Finetuning-Lab/requirements-cpu.txt):
     ```powershell
     .venv\Scripts\python -m pip install --upgrade pip
     .venv\Scripts\pip install -r requirements-cpu.txt
     ```
   - Các gói chính đã cài đặt thành công:
     - `transformers==5.19.0`
     - `tokenizers==0.23.2`
     - `jinja2==3.1.6` (Bắt buộc cho ChatML template processing)
     - `jupytext==1.19.6`
     - `pytest==9.1.1`
     - `huggingface-hub==1.33.0`
     - `numpy==2.5.3`

3. **Chạy kiểm thử khói (Smoke Test)**:
   - Chạy script kiểm tra liêm chính hệ thống ở chế độ `--smoke`:
     ```powershell
     .venv\Scripts\python scripts/verify.py --smoke
     ```

---

## 2. Kết Quả Nghiệm Thu (Evidence & Test Results)

Toàn bộ 7 kiểm tra smoke test của `scripts/verify.py` đều đạt chuẩn tuyệt đối:

```text
[  ok  ] labkit imports                                   
[  ok  ] tier resolves                                    T4 -> unsloth/Qwen3.5-4B
[  ok  ] all tiers respect the <32 effective-batch rule   
[  ok  ] data/train_seed.jsonl                            250 rows
[  ok  ] data/eval_target.jsonl                           50 rows
[  ok  ] data/eval_regression.jsonl                       15 rows
[  ok  ] unit tests                                       116 passed, 3 skipped in 0.68s

7 passed · 0 warnings · 0 failures

Ready to submit.
```

* **Mã thoát (Exit code)**: `0`
* **Số lượng unit test**: 116 tests passed, 3 skipped (các test đòi hỏi GPU/CUDA được skip hợp lệ).
* **Kiểm tra batch size**: Tất cả các compute tiers (`CPU`, `LAPTOP`, `T4`, `BIGGPU`) đều tuân thủ nguyên tắc batch hiệu dụng $< 32$ (Deck §11.4).

---

## 3. Đối Chiếu Tiêu Chuẩn Nghiệm Thu (Acceptance Criteria)

- [x] Thư mục `.venv` được tạo thành công và chứa đầy đủ binaries Python/pip.
- [x] Lệnh `scripts/verify.py --smoke` thoát với code 0, toàn bộ unit tests đều `PASS`.
