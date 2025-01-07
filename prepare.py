import nbformat
import io
import tempfile
import base64
from nbconvert import PDFExporter
from PIL import Image, ImageDraw
from docx import Document
from pptx import Presentation
from io import BytesIO
from pdf2image import convert_from_path
def compress_image(image, max_size_mb=2):
    """Compress image to a maximum size of 2MB for API compatibility."""
    buffered = io.BytesIO()
    # Convert to RGB if image is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Initial save with high quality
    image.save(buffered, format="JPEG", quality=95)
    size_kb = len(buffered.getvalue()) / 1024
    
    if size_kb > max_size_mb * 1024:
        # Calculate compression ratio
        compression_ratio = (max_size_mb * 1024) / size_kb
        quality = int(95 * compression_ratio)
        
        # Resize if quality would be too low
        if quality < 30:
            scale_factor = (max_size_mb * 1024 * 30) / (size_kb * 95)
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            quality = 30
        
        # Save with new quality/size
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        
    print(f"Compressed image size: {len(buffered.getvalue()) / 1024:.2f}KB")
    return buffered
# 定義函數：將 ipynb 轉換為 PDF
def convert_ipynb_to_pdf(ipynb_path):
    try:
        nb = nbformat.read(ipynb_path, as_version=4)
        pdf_exporter = PDFExporter()
        pdf_exporter.template_name = 'classic'
        pdf_data, _ = pdf_exporter.from_notebook_node(nb)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tf:
            tf.write(pdf_data)
            pdf_path = tf.name
        return pdf_path
    except Exception as e:
        print(f"無法讀取的檔案: {str(e)}")


# 將 PDF 轉換為 Base64 圖片
def convert_pdf_to_base64_images(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        base64_images = []
        for image in images:
            buffered = compress_image(image)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_images.append(img_base64)
        return base64_images
    except Exception as e:
        print(f"無法轉換的檔案: {str(e)}")


# 將 .docx 轉換為 Base64 圖片
def convert_docx_to_base64_images(docx_path):
    try:
        doc = Document(docx_path)
        base64_images = []
        for paragraph in doc.paragraphs:
            # 將每段文字轉換成圖片的邏輯
            img = Image.new('RGB', (800, 100), color='white')
            d = ImageDraw.Draw(img)
            d.text((10, 10), paragraph.text, fill='black')
            buffered = compress_image(img)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_images.append(img_base64)
        return base64_images
    except Exception as e:
        print(f"無法讀取檔案: {docx_path}, 錯誤: {str(e)}")


# 將 .pptx 轉換為 Base64 圖片
def convert_pptx_to_base64_images(pptx_path):
    try:
        prs = Presentation(pptx_path)
        base64_images = []
        for slide in prs.slides:
            img = Image.new('RGB', (800, 600), color='white')
            d = ImageDraw.Draw(img)
            slide_text = '\n'.join([shape.text for shape in slide.shapes if hasattr(shape, "text")])
            d.text((10, 10), slide_text, fill='black')
            buffered = compress_image(img)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_images.append(img_base64)
        return base64_images
    except Exception as e:
        print(f"無法讀取的檔案: {str(e)}")


# 將圖片轉換為 Base64 字符串
def image_to_base64(image):
    buffered = compress_image(image)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
