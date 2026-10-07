# Bảng Theo Dõi Tiến Độ (Backlog) — Lab 21 Fine-tuning LLMs

> **Mục tiêu điểm số**: Phương án 2 (105/100 Điểm)  
> **Cập nhật lần cuối**: 2026-10-07  

---

## 📌 Tổng Quan Tiến Độ

* **Tổng số task**: 12
* **Đã hoàn thành (Done)**: 1 / 12
* **Đang thực hiện (In Progress)**: 0 / 12
* **Chờ thực hiện (To Do)**: 11 / 12

---

## 📋 Danh Sách Task Theo Trạng Thái

### ⏳ Chờ Thực Hiện (To Do)

| Mã Task | Tên Task | Môi trường | Ước tính | Phụ thuộc |
|---|---|---|---|---|
| **TASK-03** | Khởi động môi trường Google Colab GPU T4 & Xác minh Smoke | Colab T4 | 5 phút | TASK-02 |
| **TASK-04** | Thực thi NB1: Dữ liệu, Chat Template & Tạo Bằng chứng Loss Mask | Colab T4 | 2 phút | TASK-03 |
| **TASK-05** | Thực thi NB2: Đo & Đóng băng Ba Baseline (Full 50 Items) | Colab T4 | 20 phút | TASK-04 |
| **TASK-06** | Thực thi NB3: Huấn luyện Cấu hình Chuẩn `correct` (LoRA Without Regret) | Colab T4 | 20 phút | TASK-05 |
| **TASK-07** | Thực thi NB4: Giải phẫu Ba Cấu hình Sai (`attn_only`, `wrong_lr`, `qlora`) | Colab T4 | 50 phút | TASK-06 |
| **TASK-08** | Thực thi NB5: Đánh giá 4 Nhóm, Chấm 3 Cấu hình Sai & Phán quyết | Colab T4 | 20 phút | TASK-07 |
| **TASK-09** | Thực thi NB6: Merge Checkpoint & Kiểm thử Hoán đổi Adapter (Bonus B1: +3) | Colab T4 | 10 phút | TASK-08 |
| **TASK-10** | Đẩy Adapter lên Hugging Face Hub Công Khai (Bonus B5: +2) | Colab T4 | 5 phút | TASK-09 |
| **TASK-11** | Chạy Gatekeeper `verify.py` & Tải Toàn bộ Artefacts về Máy | Colab / Local | 5 phút | TASK-10 |
| **TASK-12** | Soạn thảo Báo cáo Học thuật `REPORT.md` Chuẩn Rubric & Đóng gói Nộp | Local | 30 phút | TASK-11 |

---

### 🔄 Đang Thực Hiện (In Progress)

| Mã Task | Tên Task | Bắt đầu lúc | Mục tiêu chính |
|---|---|---|---|
| **TASK-02** | Khởi tạo cấu hình `.env` & Đồng bộ GitHub cá nhân | 2026-10-07 | Tạo file .env chuẩn, commit & push git repo |

---

### ✅ Đã Hoàn Thành (Done)

| Mã Task | Tên Task | Ngày hoàn thành | Tài liệu nghiệm thu |
|---|---|---|---|
| **TASK-01** | Thiết lập môi trường ảo CPU & Chạy Smoke Test cục bộ | 2026-10-07 | [docs/completed/task01_setup_and_smoke.md](task01_setup_and_smoke.md) |
