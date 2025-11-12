
import numpy as np
try:
    from scipy.ndimage import zoom
    CO_SCIPY = True
except ImportError:
    CO_SCIPY = False
    print(" scipy không có sẵn - sẽ dùng resize thủ công (chậm hơn)")


def thay_doi_kich_thuoc_anh(anh, chieu_rong_moi, chieu_cao_moi):
    
    # Resize ảnh bằng phương pháp bilinear interpolation
    
    # Tham số:
    #     anh: Ảnh đầu vào (numpy array)
    #     chieu_rong_moi: Chiều rộng mới
    #     chieu_cao_moi: Chiều cao mới
    
    # Trả về:
    #     Ảnh đã resize
    
    # Lấy kích thước ảnh gốc
    chieu_cao, chieu_rong = anh.shape[:2]
    
    # Tính tỷ lệ scale để map từ ảnh mới về ảnh gốc
    # Ví dụ: ảnh gốc 1000x1000 -> ảnh mới 500x500 thì ty_le = 2.0
    ty_le_y = chieu_cao / chieu_cao_moi
    ty_le_x = chieu_rong / chieu_rong_moi
    
    # Tạo ảnh output với kích thước mới
    # Kiểm tra xem ảnh có màu (3 channels) hay ảnh xám (2D)
    if len(anh.shape) == 3:
        anh_da_resize = np.zeros((chieu_cao_moi, chieu_rong_moi, anh.shape[2]), dtype=anh.dtype)
    else:
        anh_da_resize = np.zeros((chieu_cao_moi, chieu_rong_moi), dtype=anh.dtype)
    
    # Bilinear interpolation - nội suy tuyến tính 2 chiều
    # Duyệt qua từng pixel của ảnh mới
    for i in range(chieu_cao_moi):
        for j in range(chieu_rong_moi):
            # Tìm vị trí tương ứng trong ảnh gốc (có thể là số thập phân)
            nguon_y = i * ty_le_y
            nguon_x = j * ty_le_x
            
            # Tìm 4 điểm lân cận trong ảnh gốc để nội suy
            # y0,x0: góc trên trái; y1,x1: góc dưới phải
            y0 = int(np.floor(nguon_y))
            y1 = min(y0 + 1, chieu_cao - 1)
            x0 = int(np.floor(nguon_x))
            x1 = min(x0 + 1, chieu_rong - 1)
            
            # Tính trọng số dựa trên khoảng cách đến 4 điểm
            # dy, dx trong khoảng [0, 1]
            dy = nguon_y - y0
            dx = nguon_x - x0
            
            # Nội suy bilinear: tính giá trị pixel mới từ 4 pixel lân cận
            # Công thức: f(x,y) = f(0,0)(1-dx)(1-dy) + f(1,0)dx(1-dy) + f(0,1)(1-dx)dy + f(1,1)dxdy
            if len(anh.shape) == 3:
                # Xử lý từng channel (BGR) riêng biệt
                for c in range(anh.shape[2]):
                    gia_tri = (anh[y0, x0, c] * (1 - dx) * (1 - dy) +
                           anh[y0, x1, c] * dx * (1 - dy) +
                           anh[y1, x0, c] * (1 - dx) * dy +
                           anh[y1, x1, c] * dx * dy)
                    anh_da_resize[i, j, c] = gia_tri
            else:
                # Ảnh xám (grayscale)
                gia_tri = (anh[y0, x0] * (1 - dx) * (1 - dy) +
                       anh[y0, x1] * dx * (1 - dy) +
                       anh[y1, x0] * (1 - dx) * dy +
                       anh[y1, x1] * dx * dy)
                anh_da_resize[i, j] = gia_tri
    
    return anh_da_resize


def nhan_hai_anh(anh1, anh2):
    
    # Nhân hai ảnh với nhau (element-wise multiplication)
    
    # Tham số:
    #     anh1: Ảnh thứ nhất (numpy array)
    #     anh2: Ảnh thứ hai (numpy array)
    
    # Trả về:
    #     Kết quả nhân hai ảnh
    
    return anh1 * anh2


def chuyen_rgb_sang_xam(anh):
    
    # Chuyển RGB sang grayscale
    
    # Tham số:
    #     anh: Ảnh BGR (OpenCV format)
    
    # Trả về:
    #     Ảnh xám (numpy array)
    
    if len(anh.shape) == 3:
        # Tách 3 kênh màu BGR
        b = anh[:, :, 0].astype(np.float32)
        g = anh[:, :, 1].astype(np.float32)
        r = anh[:, :, 2].astype(np.float32)
        # Công thức chuẩn ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B
        # Mắt người nhạy cảm với màu xanh lá (Green) nhất
        anh_xam = 0.114 * b + 0.587 * g + 0.299 * r
        return anh_xam.astype(np.uint8)
    return anh


def dao_nguoc_anh(anh):
    # Đảo ngược giá trị pixel: trắng thành đen, đen thành trắng
    # Dùng cho hiệu ứng negative hoặc chuẩn bị cho color dodge
    return 255 - anh


def tao_kernel_gaussian(kich_thuoc, sigma):
    
    # Tạo kernel Gaussian
    
    # Tham số:
    #     kich_thuoc: Kích thước kernel (số lẻ)
    #     sigma: Độ lệch chuẩn
    
    # Trả về:
    #     Kernel Gaussian đã chuẩn hóa
    
    # Đảm bảo kích thước kernel là số lẻ để có tâm đối xứng
    kich_thuoc = kich_thuoc if kich_thuoc % 2 == 1 else kich_thuoc + 1
    tam = kich_thuoc // 2
    kernel = np.zeros((kich_thuoc, kich_thuoc), dtype=np.float32)
    
    # Tạo kernel Gaussian 2D theo công thức: G(x,y) = e^(-(x²+y²)/(2σ²))
    for i in range(kich_thuoc):
        for j in range(kich_thuoc):
            # Tính khoảng cách từ điểm hiện tại đến tâm kernel
            x = i - tam
            y = j - tam
            # Áp dụng hàm Gaussian
            kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    
    # Chuẩn hóa kernel để tổng = 1 (bảo toàn độ sáng)
    kernel = kernel / np.sum(kernel)
    return kernel


def ap_dung_tich_chap(anh, kernel):
    
    # Áp dụng convolution 2D
    
    # Tham số:
    #     anh: Ảnh đầu vào
    #     kernel: Kernel convolution
    
    # Trả về:
    #     Ảnh sau khi convolution
    
    chieu_cao, chieu_rong = anh.shape
    chieu_cao_k, chieu_rong_k = kernel.shape
    # Tính padding cần thiết để giữ nguyên kích thước ảnh
    pad_h = chieu_cao_k // 2
    pad_w = chieu_rong_k // 2
    
    # Thêm padding cho ảnh (mode='reflect': lấy đối xứng qua biên)
    # Giúp xử lý tốt các pixel ở biên ảnh
    anh_padding = np.pad(anh, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    ket_qua = np.zeros((chieu_cao, chieu_rong), dtype=np.float32)
    
    # Sliding window: trượt kernel qua từng vị trí của ảnh
    for i in range(chieu_cao):
        for j in range(chieu_rong):
            # Lấy vùng ảnh có kích thước bằng kernel
            vung = anh_padding[i:i+chieu_cao_k, j:j+chieu_rong_k]
            # Convolution: nhân element-wise rồi tính tổng
            # Đây là phép toán cốt lõi của CNN và các bộ lọc ảnh
            ket_qua[i, j] = np.sum(vung * kernel)
    
    # Clip giá trị về khoảng [0, 255] và chuyển về uint8
    return np.clip(ket_qua, 0, 255).astype(np.uint8)


def lam_mo_gaussian(anh, kich_thuoc_kernel, sigma):
    
    # Làm mờ Gaussian
    
    # Tham số:
    #     anh: Ảnh đầu vào
    #     kich_thuoc_kernel: Kích thước kernel
    #     sigma: Độ lệch chuẩn
    
    # Trả về:
    #     Ảnh đã làm mờ
    
    kernel = tao_kernel_gaussian(kich_thuoc_kernel, sigma)
    return ap_dung_tich_chap(anh, kernel)


def bo_loc_song_phuong_toi_uu(anh, d, sigma_mau, sigma_khong_gian):
    
    #  Downsampling tự động (ảnh > 500px)
    #  Pre-compute spatial weights (tính 1 lần)
    #  Vectorization (NumPy operations)
    #  Batch processing (50 dòng/lần)
    
    # Tham số:
    #     anh: Ảnh xám đầu vào
    #     d: Đường kính vùng lân cận
    #     sigma_mau: Độ lệch chuẩn màu
    #     sigma_khong_gian: Độ lệch chuẩn không gian
    
    # Trả về:
    #     Ảnh đã làm mịn
    
    chieu_cao, chieu_rong = anh.shape
    
    # TỐI ƯU 1: Downsampling - giảm kích thước ảnh để tăng tốc xử lý
    # Bilateral filter rất chậm với ảnh lớn (O(n²) cho mỗi pixel)
    ty_le_scale = 1.0
    if max(chieu_cao, chieu_rong) > 500:
        # Giảm kích thước xuống tối đa 500px
        ty_le_scale = 500.0 / max(chieu_cao, chieu_rong)
        chieu_cao_moi = int(chieu_cao * ty_le_scale)
        chieu_rong_moi = int(chieu_rong * ty_le_scale)
        
        if CO_SCIPY:
            anh_nho = zoom(anh, ty_le_scale, order=1)
        else:
            # Resize thủ công
            buoc_h = max(1, int(1 / ty_le_scale))
            buoc_w = max(1, int(1 / ty_le_scale))
            anh_nho = anh[::buoc_h, ::buoc_w]
        
        print(f"  📉 Downsampling: {chieu_cao}x{chieu_rong} → {chieu_cao_moi}x{chieu_rong_moi} (tăng tốc x{1/ty_le_scale:.1f})")
    else:
        anh_nho = anh
    
    cao_nho, rong_nho = anh_nho.shape
    ban_kinh = d // 2
    
    # TỐI ƯU 2: Pre-compute spatial weights - tính trước trọng số không gian
    # Trọng số này chỉ phụ thuộc vị trí, không đổi cho mọi pixel
    # Tính 1 lần thay vì tính lại cho mỗi pixel -> tiết kiệm thời gian
    trong_so_khong_gian = np.zeros((d, d), dtype=np.float32)
    for ki in range(-ban_kinh, ban_kinh + 1):
        for kj in range(-ban_kinh, ban_kinh + 1):
            # Khoảng cách Euclidean bình phương
            khoang_cach_khong_gian = ki*ki + kj*kj
            # Công thức Gaussian: e^(-d²/(2σ²))
            trong_so_khong_gian[ki + ban_kinh, kj + ban_kinh] = np.exp(
                -khoang_cach_khong_gian / (2 * sigma_khong_gian * sigma_khong_gian)
            )
    
    anh_padding = np.pad(anh_nho, ban_kinh, mode='reflect')
    ket_qua = np.zeros_like(anh_nho, dtype=np.float32)
    
    
    # TỐI ƯU 3: Batch processing - xử lý theo lô để tối ưu bộ nhớ cache
    kich_thuoc_batch = 50
    
    for bat_dau_batch in range(0, cao_nho, kich_thuoc_batch):
        ket_thuc_batch = min(bat_dau_batch + kich_thuoc_batch, cao_nho)
        
        # TỐI ƯU 4: Vectorization - sử dụng NumPy operations thay vì vòng lặp
        for i in range(bat_dau_batch, ket_thuc_batch):
            for j in range(rong_nho):
                # Giá trị pixel tâm (pixel đang xử lý)
                gia_tri_tam = anh_padding[i + ban_kinh, j + ban_kinh]
                
                # TỐI ƯU 5: Lấy toàn bộ vùng lân cận một lần
                # Thay vì truy cập từng pixel riêng lẻ
                vung = anh_padding[i:i+d, j:j+d].astype(np.float32)
                
                # Vectorized computation - tính toán song song trên toàn bộ vùng
                # Tính trọng số màu dựa trên sự khác biệt giá trị pixel
                chenh_lech_gia_tri = vung - float(gia_tri_tam)
                trong_so_mau = np.exp(
                    -(chenh_lech_gia_tri * chenh_lech_gia_tri) / (2 * sigma_mau * sigma_mau)
                )
                
                # Kết hợp trọng số không gian và trọng số màu
                # Bilateral filter = spatial weight × color weight
                trong_so_ket_hop = trong_so_khong_gian * trong_so_mau
                tong_trong_so = np.sum(trong_so_ket_hop)
                
                # Tính giá trị pixel mới bằng trung bình có trọng số
                if tong_trong_so > 0:
                    ket_qua[i, j] = np.sum(vung * trong_so_ket_hop) / tong_trong_so
                else:
                    ket_qua[i, j] = gia_tri_tam
        
        tien_do = int((ket_thuc_batch) / cao_nho * 100)
        print(f"    Tiến độ: {tien_do}%", end='\r')
    
    print()
    
    # TỐI ƯU 6: Upsampling - phóng to ảnh về kích thước gốc nếu đã downsampling
    if ty_le_scale < 1.0:
        if CO_SCIPY:
            # Dùng scipy.ndimage.zoom (nhanh hơn)
            ket_qua = zoom(ket_qua, 1.0/ty_le_scale, order=1)
            ket_qua = ket_qua[:chieu_cao, :chieu_rong]
        else:
            # Upsampling thủ công - nearest neighbor interpolation
            ket_qua_day_du = np.zeros((chieu_cao, chieu_rong), dtype=np.float32)
            for i in range(chieu_cao):
                for j in range(chieu_rong):
                    # Map pixel ảnh lớn về ảnh nhỏ
                    i_nho = int(i * ty_le_scale)
                    j_nho = int(j * ty_le_scale)
                    ket_qua_day_du[i, j] = ket_qua[min(i_nho, cao_nho-1), min(j_nho, rong_nho-1)]
            ket_qua = ket_qua_day_du
        
        print(f"  📈 Upsampling: {cao_nho}x{rong_nho} → {chieu_cao}x{chieu_rong}")
    
    return np.clip(ket_qua, 0, 255).astype(np.uint8)


def phat_hien_canh(anh):
    # Phát hiện cạnh bằng Sobel operator
    
    # Tham số:
    #     anh: Ảnh xám
    
    # Trả về:
    #     Ảnh cạnh
    
    # Sobel kernels - phát hiện gradient theo hướng x và y
    # Sobel X: phát hiện cạnh dọc (thay đổi theo chiều ngang)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    # Sobel Y: phát hiện cạnh ngang (thay đổi theo chiều dọc)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    chieu_cao, chieu_rong = anh.shape
    anh_padding = np.pad(anh, 1, mode='reflect')
    
    canh_x = np.zeros_like(anh, dtype=np.float32)
    canh_y = np.zeros_like(anh, dtype=np.float32)
    
    # Áp dụng Sobel filter theo cả 2 hướng
    for i in range(chieu_cao):
        for j in range(chieu_rong):
            vung = anh_padding[i:i+3, j:j+3].astype(np.float32)
            # Tính gradient theo trục x
            canh_x[i, j] = np.sum(vung * sobel_x)
            # Tính gradient theo trục y
            canh_y[i, j] = np.sum(vung * sobel_y)
    
    # Tính độ lớn gradient: |G| = sqrt(Gx² + Gy²)
    # Đây là cường độ cạnh tại mỗi pixel
    canh = np.sqrt(canh_x**2 + canh_y**2)
    
    # Tăng cường độ đậm của edges để nét vẽ rõ hơn
    canh = canh * 2.5
    canh = np.clip(canh, 0, 255).astype(np.uint8)
    
    return canh


def tron_mau_color_dodge(nen, tron):
    # Color dodge blending mode
    
    # Tham số:
    #     nen: Ảnh nền
    #     tron: Ảnh blend
    
    # Trả về:
    #     Ảnh sau blending
    
    nen_float = nen.astype(np.float32)
    tron_float = tron.astype(np.float32)
    
    # Đảo ngược ảnh blend
    tron_dao = 255.0 - tron_float
    # Tránh chia cho 0
    tron_dao = np.where(tron_dao == 0, 1, tron_dao)
    
    # Công thức Color Dodge: Result = Base / (1 - Blend) × 255
    # Tạo hiệu ứng làm sáng, giống như chiếu sáng lên ảnh
    # Dùng để tạo hiệu ứng phác thảo/vẽ chì
    ket_qua = (nen_float / tron_dao) * 255.0
    ket_qua = np.clip(ket_qua, 0, 255)
    
    return ket_qua.astype(np.uint8)


def dieu_chinh_tuong_phan(anh, he_so_tuong_phan):
    
    # Điều chỉnh contrast
    
    # Tham số:
    #     anh: Ảnh đầu vào
    #     he_so_tuong_phan: Hệ số contrast (1.0 = không thay đổi)
    
    # Trả về:
    #     Ảnh đã điều chỉnh contrast
    
    anh_float = anh.astype(np.float32)
    # Công thức điều chỉnh contrast: output = (input - 128) × factor + 128
    # 128 là giá trị giữa (anchor point)
    # factor > 1: tăng contrast, factor < 1: giảm contrast
    da_dieu_chinh = (anh_float - 128.0) * he_so_tuong_phan + 128.0
    da_dieu_chinh = np.clip(da_dieu_chinh, 0, 255)
    return da_dieu_chinh.astype(np.uint8)




def chuyen_thanh_phac_thao(anh_bgr, kernel_gaussian=15, sigma_gaussian=3,
                     kernel_song_phuong=5, sigma_mau=50, sigma_khong_gian=50,
                     tuong_phan=1.1, do_sang=50):
    
    # Pipeline chính: Chuyển ảnh màu thành phác thảo
    
    # Tham số:
    #     anh_bgr: Ảnh BGR (OpenCV format)
    #     kernel_gaussian: Kích thước kernel Gaussian
    #     sigma_gaussian: Sigma cho Gaussian
    #     kernel_song_phuong: Kích thước kernel Bilateral
    #     sigma_mau: Sigma màu cho Bilateral
    #     sigma_khong_gian: Sigma không gian cho Bilateral
    #     tuong_phan: Hệ số tương phản (1.0 = không đổi)
    #     do_sang: Độ sáng thêm vào (0-100)
    
    # Trả về:
    #     Ảnh phác thảo
    
    import time
    
    thoi_gian_bat_dau = time.time()
    
    # Bước 1: Chuyển sang ảnh xám
    print("\n[1/9] Chuyển ảnh xám...")
    t1 = time.time()
    anh_xam = chuyen_rgb_sang_xam(anh_bgr)
    print(f"  ✓ Hoàn thành ({time.time()-t1:.2f}s)")
    


    # Bước 2: Đảo ngược ảnh xám
    print("\n[2/9] Đảo ngược ảnh xám...")
    t2 = time.time()
    anh_xam_dao = dao_nguoc_anh(anh_xam)
    print(f"  ✓ Hoàn thành ({time.time()-t2:.2f}s)")
    


    # Bước 3: Gaussian Blur
    print(f"\n[3/9] Gaussian Blur (kernel={kernel_gaussian}, sigma={sigma_gaussian})...")
    t3 = time.time()
    anh_mo = lam_mo_gaussian(anh_xam_dao, kernel_gaussian, sigma_gaussian)
    print(f"  ✓ Hoàn thành ({time.time()-t3:.2f}s)")
    



    # Bước 4: Bilateral Filter (CHẬM NHẤT - đã tối ưu)
    print(f"\n[4/9] Bilateral Filter (d={kernel_song_phuong})...")
    t4 = time.time()
    anh_mo = bo_loc_song_phuong_toi_uu(anh_mo, kernel_song_phuong, sigma_mau, sigma_khong_gian)
    print(f"  ✓ Hoàn thành ({time.time()-t4:.2f}s)")
    



    # Bước 5: Đảo ngược ảnh đã làm mờ
    print("\n[5/9] Đảo ngược ảnh đã làm mờ...")
    t5 = time.time()
    anh_mo_dao = dao_nguoc_anh(anh_mo)
    print(f"  ✓ Hoàn thành ({time.time()-t5:.2f}s)")
    



    # Bước 6: Phát hiện cạnh
    print("\n[6/9] Phát hiện cạnh (tạo nét vẽ)...")
    t6 = time.time()
    canh = phat_hien_canh(anh_xam)
    canh_dao = 255 - canh
    print(f"  ✓ Hoàn thành ({time.time()-t6:.2f}s)")
    
    

    # Bước 7: Color Dodge Blending
    print("\n[7/9] Color Dodge Blending...")
    t7 = time.time()
    # Đảm bảo cùng kích thước trước khi blend
    if anh_mo_dao.shape != anh_xam.shape:
        anh_mo_dao = thay_doi_kich_thuoc_anh(anh_mo_dao, anh_xam.shape[1], anh_xam.shape[0])
    # Color dodge: tạo hiệu ứng phác thảo bằng cách blend ảnh gốc với ảnh mờ
    phac_thao = tron_mau_color_dodge(anh_xam, anh_mo_dao)
    print(f"  ✓ Hoàn thành ({time.time()-t7:.2f}s)")
    



    # Bước 8: Kết hợp nét vẽ cạnh
    print("\n[8/9] Kết hợp nét vẽ cạnh...")
    t8 = time.time()

    # Đảm bảo cùng kích thước
    if canh_dao.shape != phac_thao.shape:
        canh_dao = thay_doi_kich_thuoc_anh(canh_dao, phac_thao.shape[1], phac_thao.shape[0])
    
    # Làm đậm edges bằng gamma correction (power function)
    # Chuẩn hóa về [0, 1]
    canh_dao_chuan_hoa = canh_dao.astype(np.float32) / 255.0
    # Áp dụng gamma < 1 để làm nổi bật vùng tối (edges)
    canh_dao_chuan_hoa = np.power(canh_dao_chuan_hoa, 0.6)
    
    # Nhân edges với phác thảo để tạo nét vẽ rõ nét
    phac_thao = nhan_hai_anh(phac_thao.astype(np.float32) / 255.0, canh_dao_chuan_hoa)
    phac_thao = (phac_thao * 255).astype(np.uint8)
    print(f"  ✓ Hoàn thành ({time.time()-t8:.2f}s)")
    



    # Bước 9: Điều chỉnh Contrast & Brightness
    print("\n[9/9] Điều chỉnh Contrast & Brightness...")
    t9 = time.time()
    # Tăng contrast để nét vẽ rõ ràng hơn
    phac_thao = dieu_chinh_tuong_phan(phac_thao, tuong_phan)
    # Tăng brightness để ảnh sáng hơn
    phac_thao = np.clip(phac_thao.astype(np.int16) + do_sang, 0, 255).astype(np.uint8)
    
    # Thêm noise nhẹ để tạo texture giống bút chì thật
    nhieu = np.random.normal(0, 2, phac_thao.shape).astype(np.int16)
    phac_thao = np.clip(phac_thao.astype(np.int16) + nhieu, 0, 255).astype(np.uint8)
    print(f"  ✓ Hoàn thành ({time.time()-t9:.2f}s)")
    
    tong_thoi_gian = time.time() - thoi_gian_bat_dau
    
    print("\n" + "="*60)
    print(f"⚡ HOÀN THÀNH - Thời gian xử lý: {tong_thoi_gian:.2f} giây")
    print("="*60 + "\n")
    
    return phac_thao, tong_thoi_gian