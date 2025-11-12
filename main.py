import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import xu_ly_anh_toi_uu


class UngDungPhacThao:
    
    def __init__(self, cua_so_chinh):
        self.cua_so_chinh = cua_so_chinh
        self.cua_so_chinh.title("Chuyển đổi ảnh thành phác thảo bút chì")
        self.cua_so_chinh.geometry("1400x800")
        
        # Biến lưu trữ ảnh
        self.anh_goc = None
        self.anh_phac_thao = None
        self.duong_dan_anh_goc = None
        
        # Biến lưu trữ ảnh đã xử lý trung gian (để không phải xử lý lại từ đầu)
        self.anh_xam = None
        self.anh_min = None
        
        # Tham số mặc định cho Canny
        self.sigma = tk.DoubleVar(value=1.0)
        self.nguong_thap = tk.IntVar(value=15)
        self.nguong_cao = tk.IntVar(value=50)
        self.blend = tk.DoubleVar(value=0.85)
        self.thickness = tk.IntVar(value=1)
        
        # Tạo giao diện
        self.tao_giao_dien()
    
    def tao_giao_dien(self):
        """Tạo các thành phần giao diện người dùng"""
        
        # Khung chứa các nút điều khiển
        khung_nut = tk.Frame(self.cua_so_chinh, bg="#2c3e50", pady=10)
        khung_nut.pack(side=tk.TOP, fill=tk.X)
        
        # Nút tải ảnh lên
        nut_tai_anh = tk.Button(
            khung_nut,
            text="📁 Tải ảnh lên",
            command=self.tai_anh_len,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        nut_tai_anh.pack(side=tk.LEFT, padx=20)
        
        # Nút xử lý ảnh
        nut_xu_ly = tk.Button(
            khung_nut,
            text="🎨 Chuyển thành phác thảo",
            command=self.xu_ly_anh,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        nut_xu_ly.pack(side=tk.LEFT, padx=20)
        
        # Nút lưu kết quả
        nut_luu = tk.Button(
            khung_nut,
            text="💾 Lưu kết quả",
            command=self.luu_ket_qua,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        nut_luu.pack(side=tk.LEFT, padx=20)
        
        # Nút cập nhật nhanh (áp dụng tham số mới không cần xử lý lại từ đầu)
        nut_cap_nhat = tk.Button(
            khung_nut,
            text="⚡ Cập nhật nhanh",
            command=self.cap_nhat_nhanh,
            font=("Arial", 12, "bold"),
            bg="#f39c12",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        nut_cap_nhat.pack(side=tk.LEFT, padx=20)
        
        # Khung chứa thanh trượt điều chỉnh tham số
        khung_tham_so = tk.Frame(self.cua_so_chinh, bg="#34495e", pady=10)
        khung_tham_so.pack(side=tk.TOP, fill=tk.X)
        
        # Tiêu đề
        tk.Label(
            khung_tham_so,
            text="⚙️ Tham số phác thảo (Thay đổi và nhấn 'Cập nhật nhanh')",
            font=("Arial", 10, "bold"),
            bg="#34495e",
            fg="white"
        ).pack(pady=5)
        
        # Container cho các slider
        khung_sliders = tk.Frame(khung_tham_so, bg="#34495e")
        khung_sliders.pack(padx=20, pady=5)
        
        # Sigma (độ mịn)
        self.tao_slider(
            khung_sliders, 
            "Sigma (độ mịn)", 
            self.sigma, 
            0.5, 2.0, 0.1,
            "Thấp = nhiều chi tiết, Cao = mịn hơn",
            0
        )
        
        # Ngưỡng thấp
        self.tao_slider(
            khung_sliders,
            "Ngưỡng thấp",
            self.nguong_thap,
            5, 50, 1,
            "Thấp = nhiều nét nhỏ, Cao = ít nhiễu",
            1
        )
        
        # Ngưỡng cao
        self.tao_slider(
            khung_sliders,
            "Ngưỡng cao",
            self.nguong_cao,
            30, 150, 1,
            "Điều chỉnh độ mạnh của các nét chính",
            2
        )
        
        # Blend (trộn độ đậm nhạt)
        self.tao_slider(
            khung_sliders,
            "Độ mềm (Blend)",
            self.blend,
            0.5, 1.0, 0.05,
            "Cao = mềm mại, Thấp = nét sắc",
            3
        )
        
        # Thickness (độ dày nét)
        self.tao_slider(
            khung_sliders,
            "Độ dày nét",
            self.thickness,
            0, 3, 1,
            "0 = mỏng, 3 = dày",
            4
        )
        
        # Khung chứa các ảnh
        khung_anh = tk.Frame(self.cua_so_chinh, bg="#ecf0f1")
        khung_anh.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Khung ảnh gốc
        khung_anh_goc = tk.LabelFrame(
            khung_anh,
            text="Ảnh gốc",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        khung_anh_goc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.nhan_anh_goc = tk.Label(
            khung_anh_goc,
            text="Chưa có ảnh\nNhấn 'Tải ảnh lên' để bắt đầu",
            bg="#bdc3c7",
            font=("Arial", 10),
            fg="#7f8c8d"
        )
        self.nhan_anh_goc.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Khung ảnh kết quả
        khung_ket_qua = tk.LabelFrame(
            khung_anh,
            text="Ảnh phác thảo",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        khung_ket_qua.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.nhan_ket_qua = tk.Label(
            khung_ket_qua,
            text="Kết quả sẽ hiển thị ở đây",
            bg="#bdc3c7",
            font=("Arial", 10),
            fg="#7f8c8d"
        )
        self.nhan_ket_qua.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Thanh trạng thái
        self.thanh_trang_thai = tk.Label(
            self.cua_so_chinh,
            text="Sẵn sàng",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 9),
            bg="#34495e",
            fg="white"
        )
        self.thanh_trang_thai.pack(side=tk.BOTTOM, fill=tk.X)
    
    def tao_slider(self, parent, ten, bien, min_val, max_val, resolution, mo_ta, hang):
        """Tạo một slider với nhãn và giá trị hiển thị"""
        frame = tk.Frame(parent, bg="#34495e")
        frame.grid(row=hang, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        
        # Nhãn tên
        label_ten = tk.Label(
            frame,
            text=f"{ten}:",
            font=("Arial", 9, "bold"),
            bg="#34495e",
            fg="white",
            width=15,
            anchor="w"
        )
        label_ten.pack(side=tk.LEFT, padx=5)
        
        # Slider
        slider = tk.Scale(
            frame,
            variable=bien,
            from_=min_val,
            to=max_val,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            length=200,
            bg="#2c3e50",
            fg="white",
            highlightthickness=0,
            troughcolor="#1abc9c",
            font=("Arial", 8)
        )
        slider.pack(side=tk.LEFT, padx=5)
        
        # Nhãn mô tả
        label_mota = tk.Label(
            frame,
            text=mo_ta,
            font=("Arial", 8, "italic"),
            bg="#34495e",
            fg="#bdc3c7"
        )
        label_mota.pack(side=tk.LEFT, padx=10)
    
    def tai_anh_len(self):
        """Xử lý sự kiện tải ảnh lên"""
        # Mở hộp thoại chọn file
        duong_dan = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[
                ("Tất cả file ảnh", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("Tất cả file", "*.*")
            ]
        )
        
        if duong_dan:
            try:
                # Cập nhật thanh trạng thái
                self.thanh_trang_thai.config(text="Đang tải ảnh...")
                self.cua_so_chinh.update()
                
                # Đọc ảnh bằng PIL và chuyển sang numpy
                self.duong_dan_anh_goc = duong_dan
                anh_pil = Image.open(duong_dan)
                if anh_pil.mode != 'RGB':
                    anh_pil = anh_pil.convert('RGB')
                self.anh_goc = np.array(anh_pil, dtype=np.float64)
                
                # Hiển thị ảnh gốc
                self.hien_thi_anh(self.anh_goc, self.nhan_anh_goc)
                
                # Reset ảnh kết quả
                self.anh_phac_thao = None
                self.nhan_ket_qua.config(
                    image='',
                    text="Nhấn 'Chuyển thành phác thảo' để xử lý"
                )
                
                self.thanh_trang_thai.config(text=f"Đã tải ảnh: {duong_dan}")
                messagebox.showinfo("Thành công", "Đã tải ảnh thành công!")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh:\n{str(e)}")
                self.thanh_trang_thai.config(text="Lỗi khi tải ảnh")
    
    def xu_ly_anh(self):
        """Xử lý chuyển đổi ảnh thành phác thảo - SỬ DỤNG MODULE TỐI ƯU"""
        if self.anh_goc is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh lên trước!")
            return
        
        try:
            # Cập nhật thanh trạng thái
            self.thanh_trang_thai.config(text="Đang xử lý ảnh (9 bước)...")
            self.cua_so_chinh.update()
            
            # Chuyển từ PIL array (RGB) sang BGR (OpenCV format)
            if len(self.anh_goc.shape) == 3:
                # RGB -> BGR
                anh_bgr = self.anh_goc[:, :, ::-1].astype(np.uint8)
            else:
                anh_bgr = self.anh_goc.astype(np.uint8)
            
            # Gọi hàm xử lý chính từ module tối ưu
            sketch, tong_thoi_gian = xu_ly_anh_toi_uu.convert_to_sketch(
                anh_bgr,
                gaussian_kernel=15,
                gaussian_sigma=3,
                bilateral_kernel=5,
                sigma_color=50,
                sigma_space=50,
                contrast=1.1,
                brightness=50
            )
            
            # Lưu kết quả
            self.anh_phac_thao = sketch
            self.anh_min = sketch  # Cho tính năng "Cập nhật nhanh" (nếu cần)
            
            # Hiển thị kết quả
            self.hien_thi_anh(self.anh_phac_thao, self.nhan_ket_qua)
            
            # Cập nhật thanh trạng thái
            self.thanh_trang_thai.config(text=f"⚡ Hoàn thành! ")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý ảnh:\n{str(e)}")
            self.thanh_trang_thai.config(text="Lỗi khi xử lý ảnh")
    
    def cap_nhat_nhanh(self):
        """Cập nhật nhanh - điều chỉnh độ sáng và độ tương phản của ảnh đã xử lý"""
        if self.anh_phac_thao is None:
            messagebox.showwarning(
                "Cảnh báo", 
                "Vui lòng nhấn 'Chuyển thành phác thảo' ít nhất 1 lần trước!"
            )
            return
        
        try:
            self.thanh_trang_thai.config(text="⚡ Đang cập nhật nhanh...")
            self.cua_so_chinh.update()
            
            # Lấy giá trị từ các slider
            sigma_val = self.sigma.get()
            blend_val = self.blend.get()
            thickness_val = self.thickness.get()
            
            # Áp dụng điều chỉnh độ tương phản dựa trên blend
            contrast_factor = 0.8 + (blend_val * 0.4)  # 0.8 - 1.2
            adjusted = xu_ly_anh_toi_uu.adjust_contrast(self.anh_min, contrast_factor)
            
            # Áp dụng brightness dựa trên sigma
            brightness_add = int((sigma_val - 1.0) * 30)  # -15 đến +30
            adjusted = np.clip(adjusted.astype(np.int16) + brightness_add, 0, 255).astype(np.uint8)
            
            # Áp dụng độ dày nét bằng cách làm đậm/mờ các cạnh
            if thickness_val > 0:
                # Làm đậm bằng cách giảm giá trị pixel (tối hơn)
                adjusted = np.clip(adjusted.astype(np.int16) - (thickness_val * 10), 0, 255).astype(np.uint8)
            
            # Thêm noise nhẹ dựa trên các tham số
            noise_strength = 1 + sigma_val
            noise = np.random.normal(0, noise_strength, adjusted.shape).astype(np.int16)
            adjusted = np.clip(adjusted.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Cập nhật ảnh kết quả
            self.anh_phac_thao = adjusted
            
            # Hiển thị kết quả
            self.hien_thi_anh(self.anh_phac_thao, self.nhan_ket_qua)
            
            self.thanh_trang_thai.config(text=f"⚡ Cập nhật nhanh hoàn thành! (Sigma={sigma_val:.1f}, Blend={blend_val:.2f}, Độ dày={thickness_val})")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi", f"Lỗi khi cập nhật:\n{str(e)}")
            self.thanh_trang_thai.config(text="Lỗi khi cập nhật")
    
    def luu_ket_qua(self):
        """Lưu ảnh kết quả"""
        if self.anh_phac_thao is None:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để lưu!")
            return
        
        # Mở hộp thoại lưu file
        duong_dan = filedialog.asksaveasfilename(
            title="Lưu ảnh phác thảo",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("Tất cả file", "*.*")
            ]
        )
        
        if duong_dan:
            try:
                # Lưu ảnh
                self.thanh_trang_thai.config(text="Đang lưu ảnh...")
                self.cua_so_chinh.update()
                
                # Lưu ảnh bằng PIL
                anh_luu = Image.fromarray(self.anh_phac_thao)
                anh_luu.save(duong_dan)
                
                self.thanh_trang_thai.config(text=f"Đã lưu ảnh: {duong_dan}")
                messagebox.showinfo("Thành công", f"Đã lưu ảnh thành công!\n{duong_dan}")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu ảnh:\n{str(e)}")
                self.thanh_trang_thai.config(text="Lỗi khi lưu ảnh")
    
    def hien_thi_anh(self, anh_numpy, nhan):
        """
        Hiển thị ảnh NumPy lên label Tkinter
        
        Tham số:
            anh_numpy: Mảng NumPy chứa ảnh (2D hoặc 3D)
            nhan: Label Tkinter để hiển thị
        """
        # Chuẩn hóa và chuyển sang uint8
        anh_hien_thi = np.clip(anh_numpy, 0, 255).astype(np.uint8)
        
        # Pillow tự động phát hiện mode từ shape của mảng NumPy
        # 2D array -> grayscale, 3D array -> RGB
        anh_pil = Image.fromarray(anh_hien_thi)
        
        # Resize ảnh để vừa với khung hiển thị (tối đa 550x550)
        kich_thuoc_toi_da = (550, 550)
        anh_pil.thumbnail(kich_thuoc_toi_da, Image.Resampling.LANCZOS)
        
        # Chuyển sang ImageTk để hiển thị trong Tkinter
        anh_tk = ImageTk.PhotoImage(anh_pil)
        
        # Cập nhật label
        nhan.config(image=anh_tk, text='')
        nhan.image = anh_tk  # Giữ reference để tránh bị garbage collected


def main():
    """Hàm chính chạy ứng dụng"""
    cua_so = tk.Tk()
    ung_dung = UngDungPhacThao(cua_so)
    cua_so.mainloop()


if __name__ == "__main__":
    main()
