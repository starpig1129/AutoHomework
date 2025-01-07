# AutoHomework
## 專案簡介
此專案包含以下檔案與功能：

### 專案檔案
- **prepare.py**: 
  - 提供各種檔案格式的轉換功能，將 `.py`, `.java`, `.cpp`, `.txt`,`.ipynb`, `.pdf`, `.docx`, 和 `.pptx` 檔案轉換為 Base64 格式的圖片。
  - 包含圖片壓縮功能，用於確保圖片大小符合最大限制。

- **main.ipynb**: 
  - 實現整合功能的 Jupyter Notebook，可快速測試和執行 `prepare.py` 中定義的功能。

- **.gitignore**: 
  - 定義 Git 需要忽略的檔案和資料夾，避免上傳臨時檔案或敏感資訊。

### 功能介紹
1. **Notebook 轉 PDF**:
   - 使用 `convert_ipynb_to_pdf` 函數將 `.ipynb` 檔案轉換為 `.pdf` 格式。

2. **PDF 轉圖片**:
   - 使用 `convert_pdf_to_base64_images` 函數將 PDF 內容轉換為 Base64 圖片。

3. **Word (DOCX) 文件轉圖片**:
   - 使用 `convert_docx_to_base64_images` 函數將 `.docx` 檔案的段落內容生成 Base64 圖片。

4. **PowerPoint (PPTX) 文件轉圖片**:
   - 使用 `convert_pptx_to_base64_images` 函數將 `.pptx` 文件的幻燈片內容生成 Base64 圖片。

5. **圖片壓縮**:
   - 提供 `compress_image` 函數，用於壓縮圖片以符合設定的大小限制。

6. **圖片 Base64 編碼**:
   - 使用 `image_to_base64` 將圖片轉換為 Base64 字串。

## 安裝與運行

### 環境需求
- Python 3.8 或以上版本。
- 必須安裝以下套件：
  - `nbformat`
  - `nbconvert`
  - `Pillow`
  - `python-docx`
  - `python-pptx`
  - `pdf2image`
  - `langchain_openai`
  - `openai`
  - `dotenv-python`
  
### 安裝依賴
```bash
pip install -r requirements.txt
```


## 貢獻
歡迎提出 Issue 或發送 Pull Request，以改進此專案功能。

## 授權
此專案依據 MIT License 授權。
