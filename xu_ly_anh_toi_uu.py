
import numpy as np
try:
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy không có sẵn - sẽ dùng resize thủ công (chậm hơn)")


def resize_image(image, new_width, new_height):
    """
    Resize ảnh bằng phương pháp bilinear interpolation
    
    Tham số:
        image: Ảnh đầu vào (numpy array)
        new_width: Chiều rộng mới
        new_height: Chiều cao mới
    
    Trả về:
        Ảnh đã resize
    """
    height, width = image.shape[:2]
    
    # Tính tỷ lệ scale
    scale_y = height / new_height
    scale_x = width / new_width
    
    # Tạo ảnh output
    if len(image.shape) == 3:
        resized = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        resized = np.zeros((new_height, new_width), dtype=image.dtype)
    
    # Bilinear interpolation
    for i in range(new_height):
        for j in range(new_width):
            # Tìm vị trí trong ảnh gốc
            src_y = i * scale_y
            src_x = j * scale_x
            
            # Tìm 4 điểm lân cận
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, height - 1)
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, width - 1)
            
            # Tính trọng số
            dy = src_y - y0
            dx = src_x - x0
            
            # Interpolation
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    val = (image[y0, x0, c] * (1 - dx) * (1 - dy) +
                           image[y0, x1, c] * dx * (1 - dy) +
                           image[y1, x0, c] * (1 - dx) * dy +
                           image[y1, x1, c] * dx * dy)
                    resized[i, j, c] = val
            else:
                val = (image[y0, x0] * (1 - dx) * (1 - dy) +
                       image[y0, x1] * dx * (1 - dy) +
                       image[y1, x0] * (1 - dx) * dy +
                       image[y1, x1] * dx * dy)
                resized[i, j] = val
    
    return resized


def multiply_images(image1, image2):
    """
    Nhân hai ảnh với nhau (element-wise multiplication)
    
    Tham số:
        image1: Ảnh thứ nhất (numpy array)
        image2: Ảnh thứ hai (numpy array)
    
    Trả về:
        Kết quả nhân hai ảnh
    """
    return image1 * image2


def rgb_to_grayscale(image):
    """
    Chuyển RGB sang grayscale
    
    Tham số:
        image: Ảnh BGR (OpenCV format)
    
    Trả về:
        Ảnh xám (numpy array)
    """
    if len(image.shape) == 3:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return gray.astype(np.uint8)
    return image


def invert_image(image):
    return 255 - image


def create_gaussian_kernel(size, sigma):
    """
    Tạo kernel Gaussian
    
    Tham số:
        size: Kích thước kernel (số lẻ)
        sigma: Độ lệch chuẩn
    
    Trả về:
        Kernel Gaussian đã chuẩn hóa
    """
    size = size if size % 2 == 1 else size + 1
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    
    kernel = kernel / np.sum(kernel)
    return kernel


def apply_convolution(image, kernel):
    """
    Áp dụng convolution 2D
    
    Tham số:
        image: Ảnh đầu vào
        kernel: Kernel convolution
    
    Trả về:
        Ảnh sau khi convolution
    """
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros((height, width), dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            region = padded[i:i+k_height, j:j+k_width]
            output[i, j] = np.sum(region * kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)


def gaussian_blur(image, kernel_size, sigma):
    """
    Làm mờ Gaussian
    
    Tham số:
        image: Ảnh đầu vào
        kernel_size: Kích thước kernel
        sigma: Độ lệch chuẩn
    
    Trả về:
        Ảnh đã làm mờ
    """
    kernel = create_gaussian_kernel(kernel_size, sigma)
    return apply_convolution(image, kernel)


def bilateral_filter_optimized(image, d, sigma_color, sigma_space):
    """
    - Downsampling tự động (ảnh > 500px)
    - Pre-compute spatial weights (tính 1 lần)
    - Vectorization (NumPy operations)
    - Batch processing (50 dòng/lần)
    
    Tham số:
        image: Ảnh xám đầu vào
        d: Đường kính vùng lân cận
        sigma_color: Độ lệch chuẩn màu
        sigma_space: Độ lệch chuẩn không gian
    
    Trả về:
        Ảnh đã làm mịn
    """
    height, width = image.shape
    
    # TỐI ƯU 1: Downsampling
    scale_factor = 1.0
    if max(height, width) > 500:
        scale_factor = 500.0 / max(height, width)
        new_h = int(height * scale_factor)
        new_w = int(width * scale_factor)
        
        if HAS_SCIPY:
            image_small = zoom(image, scale_factor, order=1)
        else:
            # Resize thủ công
            step_h = max(1, int(1 / scale_factor))
            step_w = max(1, int(1 / scale_factor))
            image_small = image[::step_h, ::step_w]
        
        print(f"  📉 Downsampling: {height}x{width} → {new_h}x{new_w} (tăng tốc x{1/scale_factor:.1f})")
    else:
        image_small = image
    
    h_small, w_small = image_small.shape
    radius = d // 2
    
    # TỐI ƯU 2: Pre-compute spatial weights (chỉ tính 1 lần)
    spatial_weights = np.zeros((d, d), dtype=np.float32)
    for ki in range(-radius, radius + 1):
        for kj in range(-radius, radius + 1):
            spatial_dist = ki*ki + kj*kj
            spatial_weights[ki + radius, kj + radius] = np.exp(
                -spatial_dist / (2 * sigma_space * sigma_space)
            )
    
    padded = np.pad(image_small, radius, mode='reflect')
    output = np.zeros_like(image_small, dtype=np.float32)
    
    
    # TỐI ƯU 3: Batch processing
    batch_size = 50
    
    for batch_start in range(0, h_small, batch_size):
        batch_end = min(batch_start + batch_size, h_small)
        
        # TỐI ƯU 4: Vectorization
        for i in range(batch_start, batch_end):
            for j in range(w_small):
                center_value = padded[i + radius, j + radius]
                
                # TỐI ƯU 5: Lấy toàn bộ vùng một lần
                region = padded[i:i+d, j:j+d].astype(np.float32)
                
                # Vectorized computation
                value_diffs = region - float(center_value)
                range_weights = np.exp(
                    -(value_diffs * value_diffs) / (2 * sigma_color * sigma_color)
                )
                
                combined_weights = spatial_weights * range_weights
                weight_sum = np.sum(combined_weights)
                
                if weight_sum > 0:
                    output[i, j] = np.sum(region * combined_weights) / weight_sum
                else:
                    output[i, j] = center_value
        
        progress = int((batch_end) / h_small * 100)
        print(f"    Tiến độ: {progress}%", end='\r')
    
    print()
    
    # TỐI ƯU 6: Upsampling nếu cần
    if scale_factor < 1.0:
        if HAS_SCIPY:
            output = zoom(output, 1.0/scale_factor, order=1)
            output = output[:height, :width]
        else:
            # Upsampling thủ công
            output_full = np.zeros((height, width), dtype=np.float32)
            for i in range(height):
                for j in range(width):
                    i_small = int(i * scale_factor)
                    j_small = int(j * scale_factor)
                    output_full[i, j] = output[min(i_small, h_small-1), min(j_small, w_small-1)]
            output = output_full
        
        print(f"  📈 Upsampling: {h_small}x{w_small} → {height}x{width}")
    
    return np.clip(output, 0, 255).astype(np.uint8)


def detect_edges(image):
    """
    Phát hiện cạnh bằng Sobel operator
    
    Tham số:
        image: Ảnh xám
    
    Trả về:
        Ảnh cạnh
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    height, width = image.shape
    padded = np.pad(image, 1, mode='reflect')
    
    edges_x = np.zeros_like(image, dtype=np.float32)
    edges_y = np.zeros_like(image, dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            region = padded[i:i+3, j:j+3].astype(np.float32)
            edges_x[i, j] = np.sum(region * sobel_x)
            edges_y[i, j] = np.sum(region * sobel_y)
    
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # Tăng cường độ đậm của edges
    edges = edges * 2.5
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return edges


def color_dodge(base, blend):
    """
    Color dodge blending mode
    
    Tham số:
        base: Ảnh nền
        blend: Ảnh blend
    
    Trả về:
        Ảnh sau blending
    """
    base_float = base.astype(np.float32)
    blend_float = blend.astype(np.float32)
    
    inverted_blend = 255.0 - blend_float
    inverted_blend = np.where(inverted_blend == 0, 1, inverted_blend)
    
    result = (base_float / inverted_blend) * 255.0
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def adjust_contrast(image, contrast_factor):
    """
    Điều chỉnh contrast
    
    Tham số:
        image: Ảnh đầu vào
        contrast_factor: Hệ số contrast (1.0 = không thay đổi)
    
    Trả về:
        Ảnh đã điều chỉnh contrast
    """
    img_float = image.astype(np.float32)
    adjusted = (img_float - 128.0) * contrast_factor + 128.0
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)


def convert_to_sketch(image_bgr, gaussian_kernel=15, gaussian_sigma=3,
                     bilateral_kernel=5, sigma_color=50, sigma_space=50,
                     contrast=1.1, brightness=50):
    """
    Pipeline chính: Chuyển ảnh màu thành phác thảo
    
    Tham số:
        image_bgr: Ảnh BGR (OpenCV format)
        gaussian_kernel: Kích thước kernel Gaussian
        gaussian_sigma: Sigma cho Gaussian
        bilateral_kernel: Kích thước kernel Bilateral
        sigma_color: Sigma màu cho Bilateral
        sigma_space: Sigma không gian cho Bilateral
        contrast: Hệ số tương phản (1.0 = không đổi)
        brightness: Độ sáng thêm vào (0-100)
    
    Trả về:
        Ảnh phác thảo
    """
    import time
    
    print("\n" + "="*60)
    print("BẮT ĐẦU XỬ LÝ (LOGIC TỐI ƯU)")
    print("="*60)
    
    total_start = time.time()
    
    # Bước 1: Chuyển sang ảnh xám
    print("\n[1/9] Chuyển ảnh xám...")
    t1 = time.time()
    gray_image = rgb_to_grayscale(image_bgr)
    print(f"  ✓ Hoàn thành ({time.time()-t1:.2f}s)")
    
    # Bước 2: Đảo ngược ảnh xám
    print("\n[2/9] Đảo ngược ảnh xám...")
    t2 = time.time()
    inverted_gray = invert_image(gray_image)
    print(f"  ✓ Hoàn thành ({time.time()-t2:.2f}s)")
    
    # Bước 3: Gaussian Blur
    print(f"\n[3/9] Gaussian Blur (kernel={gaussian_kernel}, sigma={gaussian_sigma})...")
    t3 = time.time()
    blurred = gaussian_blur(inverted_gray, gaussian_kernel, gaussian_sigma)
    print(f"  ✓ Hoàn thành ({time.time()-t3:.2f}s)")
    
    # Bước 4: Bilateral Filter (CHẬM NHẤT - đã tối ưu)
    print(f"\n[4/9] Bilateral Filter (d={bilateral_kernel})...")
    t4 = time.time()
    blurred = bilateral_filter_optimized(blurred, bilateral_kernel, sigma_color, sigma_space)
    print(f"  ✓ Hoàn thành ({time.time()-t4:.2f}s)")
    
    # Bước 5: Đảo ngược ảnh đã làm mờ
    print("\n[5/9] Đảo ngược ảnh đã làm mờ...")
    t5 = time.time()
    inverted_blurred = invert_image(blurred)
    print(f"  ✓ Hoàn thành ({time.time()-t5:.2f}s)")
    
    # Bước 6: Phát hiện cạnh
    print("\n[6/9] Phát hiện cạnh (tạo nét vẽ)...")
    t6 = time.time()
    edges = detect_edges(gray_image)
    edges_inv = 255 - edges
    print(f"  ✓ Hoàn thành ({time.time()-t6:.2f}s)")
    
    
    # Bước 7: Color Dodge Blending
    print("\n[7/9] Color Dodge Blending...")
    t7 = time.time()
    # Đảm bảo cùng kích thước
    if inverted_blurred.shape != gray_image.shape:
        inverted_blurred = resize_image(inverted_blurred, gray_image.shape[1], gray_image.shape[0])
    sketch = color_dodge(gray_image, inverted_blurred)
    print(f"  ✓ Hoàn thành ({time.time()-t7:.2f}s)")
    
    # Bước 8: Kết hợp nét vẽ cạnh
    print("\n[8/9] Kết hợp nét vẽ cạnh...")
    t8 = time.time()
    # Đảm bảo cùng kích thước
    if edges_inv.shape != sketch.shape:
        edges_inv = resize_image(edges_inv, sketch.shape[1], sketch.shape[0])
    
    # Làm đậm edges
    edges_inv_normalized = edges_inv.astype(np.float32) / 255.0
    edges_inv_normalized = np.power(edges_inv_normalized, 0.6)
    
    sketch = multiply_images(sketch.astype(np.float32) / 255.0, edges_inv_normalized)
    sketch = (sketch * 255).astype(np.uint8)
    print(f"  ✓ Hoàn thành ({time.time()-t8:.2f}s)")
    
    # Bước 9: Điều chỉnh Contrast & Brightness
    print("\n[9/9] Điều chỉnh Contrast & Brightness...")
    t9 = time.time()
    sketch = adjust_contrast(sketch, contrast)
    sketch = np.clip(sketch.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    
    # Thêm noise nhẹ
    noise = np.random.normal(0, 2, sketch.shape).astype(np.int16)
    sketch = np.clip(sketch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    print(f"  ✓ Hoàn thành ({time.time()-t9:.2f}s)")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print(f"⚡ HOÀN THÀNH - Thời gian xử lý: {total_time:.2f} giây")
    print("="*60 + "\n")
    
    return sketch, total_time
