# 🏥 Nhận dạng Thực thể trong Văn bản Y tế tiếng Việt (Medical NER)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Dự án nghiên cứu xây dựng hệ thống Trích xuất Thực thể Y khoa (Medical Named Entity Recognition - NER) chuyên sâu cho ngôn ngữ tiếng Việt bằng cách tinh chỉnh Mô hình Ngôn ngữ Lớn (LLMs). Dự án được thực hiện tại **Trường Đại học Khoa học Tự nhiên, Đại học Quốc gia Hà Nội (HUS)**.

## 🌟 Điểm nổi bật (Key Contributions)
- **Tinh chỉnh Hiệu quả Tham số (PEFT):** Áp dụng kỹ thuật **QLoRA** kết hợp lượng hóa 4-bit (NF4) để huấn luyện mô hình `Qwen2.5-3B-Instruct`. Hệ thống vận hành mượt mà trên phần cứng giới hạn (GPU 15GB VRAM - NVIDIA T4) mà vẫn bảo toàn năng lực biểu diễn tri thức.
- **Prompt Engineering & Hậu xử lý:** Ép LLM sinh đầu ra theo định dạng JSON có cấu trúc nghiêm ngặt. Phát triển thuật toán cắt tỉa biên dấu câu, tối ưu tọa độ không gian văn bản để loại bỏ triệt để hiện tượng sinh văn bản ảo giác (hallucination).
- **Khung đánh giá LLM-as-a-Judge:** Vượt qua giới hạn cứng nhắc của tiêu chuẩn đối khớp chuỗi truyền thống (`seqeval`), hệ thống tích hợp mô hình `Qwen2.5-7B-Instruct` chạy cục bộ làm Giám khảo tự động, giúp phân rã 5 nhóm lỗi hạt mịn và đánh giá chính xác độ hiểu ngữ nghĩa y khoa lâm sàng.

## 🛠 Công nghệ sử dụng (Tech Stack)
- **Ngôn ngữ:** Python
- **AI/Deep Learning:** PyTorch
- **Ecosystem:** Hugging Face (`transformers`, `peft`, `datasets`, `bitsandbytes`)
- **Đánh giá:** `seqeval` (Đánh giá khắt khe), LLM Local Judge (Đánh giá nới lỏng)

## 📂 Dữ liệu Thực nghiệm
Hệ thống được thiết kế Adapter độc lập để trích xuất trên 2 miền văn phong y tế khác nhau:
1. **ViMQ:** Dữ liệu hỏi đáp sức khỏe cộng đồng. Nhãn mục tiêu: `SYMPTOM_AND_DISEASE`, `MEDICAL_PROCEDURE`, `DRUG`.
2. **ViMedNER:** Bệnh án lâm sàng và y văn hàn lâm. Nhãn mục tiêu: `DISEASE`, `SYMPTOM`, `TREATMENT`, `CAUSE`, `DIAGNOSTIC`.

## 📊 Kết quả Đánh giá
Hệ thống đạt hiệu năng xuất sắc, đặc biệt khi được thẩm định dưới lăng kính ngữ nghĩa chẩn đoán thực tiễn (Relaxed Metrics):

| Bộ dữ liệu | Độ phức tạp | F1-Score (Khắt khe / Strict) | F1-Score (Nới lỏng / Relaxed) |
| :--- | :--- | :--- | :--- |
| **ViMQ** | Cơ bản | 75.83% | **88.53%** |
| **ViMedNER** | Chuyên sâu | 60.96% | **70.92%** |

*(Tham khảo thêm chi tiết phân bố lỗi Missing, Spurious, Type Error trong Báo cáo thực nghiệm).*

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Khởi tạo môi trường
Clone repository và cài đặt các thư viện nền tảng:
```bash
git clone [https://github.com/your-username/medical-ner-llm.git](https://github.com/your-username/medical-ner-llm.git)
cd medical-ner-llm
pip install -q -U transformers accelerate peft bitsandbytes datasets seqeval torch
