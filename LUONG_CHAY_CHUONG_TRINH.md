# 📋 MÔ TẢ LUỒNG CHẠY CHƯƠNG TRÌNH

## 🎯 Tổng quan
Chương trình chuyển đổi ảnh thành phác thảo bút chì (Pencil Sketch) sử dụng giao diện đồ họa Tkinter và xử lý ảnh với NumPy.

---

## 📁 Cấu trúc dự án

```
Xu_Ly_Anh-master/
├── main.py                    # File chính - Giao diện GUI
├── xu_ly_anh_toi_uu.py       # Module xử lý ảnh (thuật toán tối ưu)
├── requirements.txt           # Thư viện phụ thuộc
└── __pycache__/              # Cache Python
```

---

## 🚀 Luồng chạy chính

### **1. Khởi động ứng dụng**
```
main.py → main() → Khởi tạo cửa sổ Tkinter → UngDungPhacThao.__init__()
```

**Chi tiết:**
- Tạo cửa sổ chính kích thước 1400x800
- Khởi tạo các biến lưu trữ ảnh:
  - `anh_goc`: Ảnh gốc người dùng tải lên
  - `anh_phac_thao`: Ảnh kết quả sau xử lý
  - `anh_xam`: Ảnh xám trung gian
  - `anh_min`: Ảnh đã làm mịn (cho cập nhật nhanh)
- Khởi tạo tham số mặc định:
  - `sigma = 1.0`: Độ mịn
  - `nguong_thap = 15`: Ngưỡng thấp (nhiều nét nhỏ)
  - `nguong_cao = 50`: Ngưỡng cao (nét chính)
  - `blend = 0.85`: Độ mềm
  - `thickness = 1`: Độ dày nét

### **2. Tạo giao diện người dùng**
```
tao_giao_dien() → Tạo các thành phần GUI
```

**Các thành phần:**

#### **A. Khung nút điều khiển (khung_nut)**
- 🟦 **Nút "Tải ảnh lên"** → `tai_anh_len()`
- 🟥 **Nút "Chuyển thành phác thảo"** → `xu_ly_anh()`
- 🟩 **Nút "Lưu kết quả"** → `luu_ket_qua()`
- 🟨 **Nút "Cập nhật nhanh"** → `cap_nhat_nhanh()`

#### **B. Khung tham số (khung_tham_so)**
5 thanh trượt (slider):
- **Sigma (0.5-2.0)**: Độ mịn của ảnh
- **Ngưỡng thấp (5-50)**: Điều chỉnh chi tiết nhỏ
- **Ngưỡng cao (30-150)**: Điều chỉnh nét chính
- **Độ mềm/Blend (0.5-1.0)**: Độ mềm mại của nét
- **Độ dày nét (0-3)**: Độ đậm của nét vẽ

#### **C. Khung hiển thị ảnh (khung_anh)**
- **Bên trái**: Ảnh gốc (`nhan_anh_goc`)
- **Bên phải**: Ảnh phác thảo (`nhan_ket_qua`)

#### **D. Thanh trạng thái**
Hiển thị trạng thái xử lý hiện tại

---

## 🔄 Luồng xử lý chính

### **BƯỚC 1: Tải ảnh lên** 📁

```
Người dùng nhấn "Tải ảnh lên" 
    ↓
tai_anh_len()
    ↓
Mở hộp thoại chọn file (filedialog)
    ↓
Đọc ảnh bằng PIL.Image.open()
    ↓
Chuyển sang RGB nếu cần
    ↓
Chuyển sang NumPy array (dtype=float64)
    ↓
Hiển thị ảnh gốc lên GUI (hien_thi_anh())
    ↓
Reset ảnh kết quả
    ↓
Cập nhật thanh trạng thái: "Đã tải ảnh: [đường dẫn]"
```

**Định dạng ảnh hỗ trợ:**
- JPG/JPEG
- PNG
- BMP
- GIF

---

### **BƯỚC 2: Xử lý chuyển đổi ảnh** 🎨

```
Người dùng nhấn "Chuyển thành phác thảo"
    ↓
xu_ly_anh()
    ↓
Kiểm tra ảnh gốc có tồn tại không
    ↓
Chuyển RGB → BGR (OpenCV format)
    ↓
Gọi xu_ly_anh_toi_uu.convert_to_sketch()
    ↓
[9 BƯỚC XỬ LÝ ẢNH - Chi tiết bên dưới]
    ↓
Nhận kết quả ảnh phác thảo + thời gian xử lý
    ↓
Hiển thị kết quả lên GUI
    ↓
Cập nhật thanh trạng thái: "⚡ Hoàn thành!"
```

---

### **🔬 9 BƯỚC XỬ LÝ ẢNH (Module xu_ly_anh_toi_uu.py)**

#### **Bước 1: Chuyển ảnh xám (Grayscale Conversion)**
```python
rgb_to_grayscale(image_bgr)
```
- Áp dụng công thức: `gray = 0.114*B + 0.587*G + 0.299*R`
- Kết quả: Ảnh xám 1 kênh

---

#### **Bước 2: Đảo ngược ảnh xám (Inversion)**
```python
invert_image(gray_image)
```
- Công thức: `inverted = 255 - gray`
- Tạo hiệu ứng âm bản (negative)

---

#### **Bước 3: Gaussian Blur (Làm mờ Gaussian)**
```python
gaussian_blur(inverted_gray, kernel=15, sigma=3)
```
**Thuật toán:**
1. Tạo kernel Gaussian:
   - Kích thước: 15x15
   - Công thức: `G(x,y) = exp(-(x² + y²)/(2σ²))`
2. Áp dụng convolution 2D:
   - Padding: reflect mode
   - Convolution từng pixel

**Mục đích:** Làm mờ ảnh để giảm nhiễu

---

#### **Bước 4: Bilateral Filter (QUAN TRỌNG - CHẬM NHẤT)** ⚡
```python
bilateral_filter_optimized(blurred, d=5, sigma_color=50, sigma_space=50)
```

**Các tối ưu hóa:**
1. **Downsampling tự động:**
   - Nếu ảnh > 500px → scale xuống 500px
   - Tăng tốc lên nhiều lần

2. **Pre-compute spatial weights:**
   - Tính trọng số không gian 1 lần duy nhất
   - Tránh tính toán lặp lại

3. **Vectorization với NumPy:**
   - Xử lý toàn bộ vùng cùng lúc
   - Tận dụng tối đa NumPy

4. **Batch processing:**
   - Xử lý 50 dòng/lần
   - Hiển thị tiến độ real-time

5. **Upsampling:**
   - Phóng to về kích thước ban đầu

**Công thức Bilateral:**
```
BF[I]_p = (1/W_p) × Σ_q∈S G_σs(‖p-q‖) × G_σr(|I_p - I_q|) × I_q

Trong đó:
- G_σs: Gaussian không gian (spatial)
- G_σr: Gaussian màu sắc (range)
- W_p: Tổng trọng số chuẩn hóa
```

**Mục đích:** Làm mịn ảnh nhưng giữ nguyên cạnh

---

#### **Bước 5: Đảo ngược ảnh đã làm mờ**
```python
invert_image(blurred)
```
- Đảo ngược lại để chuẩn bị cho blending

---

#### **Bước 6: Phát hiện cạnh (Edge Detection)**
```python
detect_edges(gray_image)
```
**Thuật toán Sobel:**
1. Kernel Sobel X (gradient ngang):
   ```
   [-1  0  1]
   [-2  0  2]
   [-1  0  1]
   ```

2. Kernel Sobel Y (gradient dọc):
   ```
   [-1 -2 -1]
   [ 0  0  0]
   [ 1  2  1]
   ```

3. Tính gradient:
   ```
   Edge = √(Gx² + Gy²) × 2.5
   ```

4. Đảo ngược: `edges_inv = 255 - edges`

**Mục đích:** Tạo nét vẽ sắc nét

---

#### **Bước 7: Color Dodge Blending**
```python
color_dodge(gray_image, inverted_blurred)
```
**Công thức:**
```
Result = (Base / (255 - Blend)) × 255
```

**Đặc điểm:**
- Tạo hiệu ứng phác thảo mềm mại
- Làm sáng vùng có giá trị cao
- Tránh chia cho 0

**Mục đích:** Tạo hiệu ứng bút chì cơ bản

---

#### **Bước 8: Kết hợp nét vẽ cạnh**
```python
multiply_images(sketch, edges_inv_normalized)
```
**Quy trình:**
1. Chuẩn hóa edges: `edges_norm = edges / 255`
2. Làm đậm: `edges_norm = edges_norm^0.6`
3. Nhân element-wise: `result = sketch × edges_norm`

**Mục đích:** Thêm độ sắc nét cho nét vẽ

---

#### **Bước 9: Điều chỉnh Contrast & Brightness**
```python
adjust_contrast(sketch, contrast=1.1)
sketch = sketch + brightness (50)
```
**Quy trình:**
1. Điều chỉnh contrast:
   ```
   adjusted = (image - 128) × 1.1 + 128
   ```

2. Tăng độ sáng:
   ```
   adjusted = adjusted + 50
   ```

3. Thêm noise nhẹ:
   ```
   noise = random.normal(0, 2, shape)
   result = adjusted + noise
   ```

**Mục đích:** Tạo hiệu ứng tự nhiên, giống vẽ tay

---

### **BƯỚC 3: Cập nhật nhanh** ⚡

```
Người dùng thay đổi slider và nhấn "Cập nhật nhanh"
    ↓
cap_nhat_nhanh()
    ↓
Lấy giá trị từ các slider
    ↓
Áp dụng điều chỉnh KHÔNG cần xử lý lại từ đầu:
    - Điều chỉnh contrast dựa trên blend
    - Điều chỉnh brightness dựa trên sigma
    - Làm đậm/mờ nét dựa trên thickness
    - Thêm noise dựa trên sigma
    ↓
Hiển thị kết quả ngay lập tức
    ↓
Cập nhật thanh trạng thái với các tham số
```

**Ưu điểm:**
- Xử lý cực nhanh (< 1 giây)
- Không cần xử lý lại 9 bước
- Cho phép thử nghiệm real-time

---

### **BƯỚC 4: Lưu kết quả** 💾

```
Người dùng nhấn "Lưu kết quả"
    ↓
luu_ket_qua()
    ↓
Kiểm tra ảnh kết quả có tồn tại không
    ↓
Mở hộp thoại lưu file (filedialog)
    ↓
Chuyển NumPy array → PIL Image
    ↓
Lưu file (PNG/JPG/BMP)
    ↓
Hiển thị thông báo thành công
```

---

## 🔧 Các hàm tiện ích

### **1. hien_thi_anh(anh_numpy, nhan)**
**Chức năng:** Hiển thị ảnh NumPy lên Tkinter Label

**Quy trình:**
1. Chuẩn hóa: `clip(0, 255)` → `uint8`
2. Chuyển NumPy → PIL Image
3. Resize về tối đa 550x550 (giữ tỷ lệ)
4. Chuyển PIL → ImageTk
5. Cập nhật Label
6. Giữ reference tránh garbage collection

---

### **2. tao_slider(parent, ten, bien, min_val, max_val, resolution, mo_ta, hang)**
**Chức năng:** Tạo slider với nhãn và mô tả

**Thành phần:**
- Label tên (bên trái)
- Scale slider (giữa)
- Label mô tả (bên phải)

---

## 📊 Sơ đồ luồng dữ liệu

```
┌─────────────────┐
│  Ảnh gốc (RGB)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chuyển BGR     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        XỬ LÝ 9 BƯỚC                     │
│  ┌────────────────────────────────┐    │
│  │ 1. RGB → Grayscale             │    │
│  └──────────────┬─────────────────┘    │
│                 ▼                       │
│  ┌────────────────────────────────┐    │
│  │ 2. Invert                      │    │
│  └──────────────┬─────────────────┘    │
│                 ▼                       │
│  ┌────────────────────────────────┐    │
│  │ 3. Gaussian Blur               │    │
│  └──────────────┬─────────────────┘    │
│                 ▼                       │
│  ┌────────────────────────────────┐    │
│  │ 4. Bilateral Filter (CHẬM)     │    │
│  └──────────────┬─────────────────┘    │
│                 ▼                       │
│  ┌────────────────────────────────┐    │
│  │ 5. Invert                      │    │
│  └──────────────┬─────────────────┘    │
│                 ▼                       │
│  ┌────────────────────────────────┐    │
│  │ 6. Edge Detection (Sobel)      │    │
│  └──────────────┬─────────────────┘    │
│                 ▼                       │
│  ┌────────────────────────────────┐    │
│  │ 7. Color Dodge Blending        │    │
│  └──────────────┬─────────────────┘    │
│                 ▼                       │
│  ┌────────────────────────────────┐    │
│  │ 8. Multiply with Edges         │    │
│  └──────────────┬─────────────────┘    │
│                 ▼                       │
│  ┌────────────────────────────────┐    │
│  │ 9. Contrast + Brightness       │    │
│  └──────────────┬─────────────────┘    │
└─────────────────┼───────────────────────┘
                  ▼
         ┌────────────────┐
         │ Ảnh phác thảo  │
         └────────────────┘
```

---

## ⚙️ Các tham số điều chỉnh

| Tham số | Mặc định | Phạm vi | Ảnh hưởng |
|---------|----------|---------|-----------|
| **Sigma** | 1.0 | 0.5-2.0 | Độ mịn (thấp=chi tiết, cao=mịn) |
| **Ngưỡng thấp** | 15 | 5-50 | Nhiều nét nhỏ (thấp=nhiều, cao=ít) |
| **Ngưỡng cao** | 50 | 30-150 | Độ mạnh nét chính |
| **Blend** | 0.85 | 0.5-1.0 | Độ mềm (cao=mềm, thấp=sắc) |
| **Thickness** | 1 | 0-3 | Độ dày nét (0=mỏng, 3=dày) |

---

## 🎯 Các tính năng chính

### ✅ Điểm mạnh:
1. **Xử lý tối ưu:** Bilateral Filter được tối ưu với 6 kỹ thuật
2. **Cập nhật nhanh:** Điều chỉnh tham số không cần xử lý lại
3. **Giao diện thân thiện:** Tkinter với các nút trực quan
4. **Không phụ thuộc OpenCV:** Chỉ dùng NumPy + PIL
5. **Hiển thị tiến độ:** Console log chi tiết từng bước
6. **Hỗ trợ nhiều định dạng:** JPG, PNG, BMP, GIF

### ⚠️ Hạn chế:
1. **Bilateral Filter vẫn chậm:** Với ảnh lớn > 1000px
2. **Chỉ xử lý tuần tự:** Không dùng đa luồng/GPU
3. **Resize thủ công chậm:** Khi không có scipy

---

## 📦 Thư viện sử dụng

### **Bắt buộc:**
- **NumPy** (>= 1.21.0): Xử lý ma trận ảnh
- **Pillow** (>= 9.0.0): Đọc/ghi/hiển thị ảnh
- **Tkinter**: Giao diện GUI (built-in Python)

### **Tùy chọn (Tối ưu):**
- **SciPy**: Resize nhanh hơn với `zoom()`

---

## 🚀 Hướng dẫn chạy

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy chương trình
python main.py

# 3. Sử dụng:
#    - Nhấn "Tải ảnh lên" → Chọn ảnh
#    - Nhấn "Chuyển thành phác thảo" → Đợi xử lý
#    - Điều chỉnh slider → Nhấn "Cập nhật nhanh"
#    - Nhấn "Lưu kết quả" → Chọn nơi lưu
```

---

## 🔍 Debug & Log

### **Console output mẫu:**
```
============================================================
BẮT ĐẦU XỬ LÝ (LOGIC TỐI ƯU)
============================================================

[1/9] Chuyển ảnh xám...
  ✓ Hoàn thành (0.05s)

[2/9] Đảo ngược ảnh xám...
  ✓ Hoàn thành (0.02s)

[3/9] Gaussian Blur (kernel=15, sigma=3)...
  ✓ Hoàn thành (1.23s)

[4/9] Bilateral Filter (d=5)...
  📉 Downsampling: 1920x1080 → 500x281 (tăng tốc x3.8)
    Tiến độ: 100%
  ✓ Hoàn thành (5.67s)
  📈 Upsampling: 281x500 → 1080x1920

[5/9] Đảo ngược ảnh đã làm mờ...
  ✓ Hoàn thành (0.02s)

[6/9] Phát hiện cạnh (tạo nét vẽ)...
  ✓ Hoàn thành (0.45s)

[7/9] Color Dodge Blending...
  ✓ Hoàn thành (0.18s)

[8/9] Kết hợp nét vẽ cạnh...
  ✓ Hoàn thành (0.12s)

[9/9] Điều chỉnh Contrast & Brightness...
  ✓ Hoàn thành (0.15s)

============================================================
⚡ HOÀN THÀNH - Thời gian xử lý: 7.89 giây
============================================================
```

---

## 📝 Ghi chú kỹ thuật

### **1. Định dạng ảnh trong chương trình:**
- **PIL Image (RGB)** → NumPy array (float64) → **BGR** (OpenCV format)
- Xử lý trong BGR
- Hiển thị GUI: Grayscale hoặc RGB (tự động phát hiện)

### **2. Tối ưu hóa bộ nhớ:**
- Dùng `dtype=float32` thay vì `float64` trong xử lý
- Downsampling ảnh lớn
- Batch processing để giảm overhead

### **3. Xử lý lỗi:**
- Try-catch cho tất cả file I/O
- Kiểm tra ảnh tồn tại trước xử lý
- Hiển thị lỗi chi tiết với traceback

---

## 🎓 Thuật toán nền tảng

### **Computer Vision:**
1. **Grayscale Conversion** - Chuyển màu sang xám
2. **Image Inversion** - Đảo ngược màu
3. **Gaussian Blur** - Làm mờ Gaussian
4. **Bilateral Filtering** - Lọc song phương
5. **Edge Detection (Sobel)** - Phát hiện cạnh
6. **Color Dodge** - Blending mode
7. **Image Multiplication** - Nhân ảnh
8. **Contrast Adjustment** - Điều chỉnh tương phản

### **Image Processing:**
- Convolution 2D
- Bilinear Interpolation
- Padding (reflect mode)
- Normalization & Clipping

---

## 📧 Liên hệ & Đóng góp
Nếu bạn muốn cải thiện chương trình hoặc báo lỗi, vui lòng mở issue hoặc pull request trên GitHub.

---

**📅 Ngày tạo:** 2025  
**✍️ Tác giả:** [Tên tác giả]  
**📌 Phiên bản:** 1.0  
**📜 Giấy phép:** [Loại giấy phép]

---

**🎨 Chúc bạn tạo ra những bức phác thảo tuyệt vời!** ✨
