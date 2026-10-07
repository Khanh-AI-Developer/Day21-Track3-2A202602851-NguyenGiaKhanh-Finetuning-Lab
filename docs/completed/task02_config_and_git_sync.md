# Nghiệm Thu TASK-02: Khởi tạo Cấu hình `.env` & Đồng bộ GitHub cá nhân

> **Mã nhiệm vụ**: `TASK-02`  
> **Thời gian hoàn thành**: 2026-10-07  
> **Người thực hiện**: Antigravity & User  
> **Trạng thái**: ✅ **HOÀN THÀNH (DONE)**  

---

## 1. Nội Dung Công Việc Đã Thực Hiện

1. **Khởi tạo và cấu hình file `.env`**:
   - Tạo file `.env` tại thư mục gốc của repository với các biến môi trường chuẩn xác cho tier T4:
     ```ini
     # Hardware tier to run. CPU | LAPTOP | T4 | BIGGPU
     COMPUTE_TIER=T4

     # Loss-mask mode for NB3. assistant-only | masked-think | response-only | everything
     MASK_MODE=assistant-only

     # Epochs for NB3 and NB4 (both share this budget)
     EPOCHS=2

     # NOTE: EVAL_LIMIT is left unset for a full submittable run (50 eval items)
     ```
   - **Lưu ý liêm chính thực nghiệm**: Cờ `EVAL_LIMIT` được chủ ý bỏ trống (unset) để hệ thống tự động sử dụng toàn bộ 50 mẫu đánh giá (`n_target = 50`), đảm bảo không bị gatekeeper `scripts/verify.py` đánh dấu `smoke_mode` và từ chối khi nộp bài (khắc phục lỗi F-27).
   - Xác nhận file `.env` được bảo vệ bởi `.gitignore` và không bị lộ secrets lên git.

2. **Đồng bộ hóa toàn bộ tài liệu & mã nguồn lên GitHub cá nhân**:
   - Thêm các file tài liệu và quản lý task mới vào git staging:
     - `Rule.md`
     - `huongdan.md`
     - `docs/specs/lab21_finetuning_spec.md`
     - `docs/plans/lab21_execution_plan.md`
     - `docs/plans/backlog.md`
     - `docs/tasks/lab21_task_breakdown.md`
     - `docs/completed/README.md`
     - `docs/completed/task01_setup_and_smoke.md`
   - Tạo commit:
     ```bash
     git commit -m "docs: add specs, plans, backlog, tasks and complete task01"
     ```
   - Đẩy commit lên remote repository:
     ```bash
     git push origin main
     ```

---

## 2. Kết Quả Nghiệm Thu (Evidence & Output)

* **Commit Hash**: `577a265`
* **Remote Repository**: `https://github.com/Khanh-AI-Developer/Day21-Track3-2A202602851-NguyenGiaKhanh-Finetuning-Lab`
* **Output từ lệnh git push**:
  ```text
  To https://github.com/Khanh-AI-Developer/Day21-Track3-2A202602851-NguyenGiaKhanh-Finetuning-Lab
     d27c1c0..577a265  main -> main
  ```
* **Trạng thái Git**: `working tree clean`, `branch is up to date with 'origin/main'`.

---

## 3. Đối Chiếu Tiêu Chuẩn Nghiệm Thu (Acceptance Criteria)

- [x] File `.env` tồn tại tại thư mục gốc và chứa cấu hình chuẩn (`COMPUTE_TIER=T4`, `MASK_MODE=assistant-only`, `EPOCHS=2`, không có `EVAL_LIMIT`).
- [x] GitHub repository cá nhân đã cập nhật commit mới nhất (`577a265`), sẵn sàng để clone/pull từ Google Colab.
