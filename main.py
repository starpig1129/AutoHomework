import os
import csv
import base64
import openai
import logging
import anthropic
import magic
import zipfile
import shutil
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from prepare import convert_ipynb_to_pdf,convert_pdf_to_base64_images,convert_docx_to_base64_images,convert_pptx_to_base64_images,image_to_base64
# 設置日誌
logging.basicConfig(filename='grading.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
HOMEWORK_DIR = os.path.abspath(os.getenv('HOMEWORK_DIR'))
# 確認路徑是否存在
if not os.path.exists(HOMEWORK_DIR):
    raise FileNotFoundError(f"The directory {HOMEWORK_DIR} does not exist. Current path: {HOMEWORK_DIR}")
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
use_anthropic = False
with open('assignment_requirements.txt', 'r', encoding='utf-8') as f:
    assignment_requirements = f.read()

# 讀取學生檔案並分類
def read_files(student):
    image_base64_list = []
    text_contents = []
    
    logging.info(f"處理學生作業: {student['id']}")
    
    for text_file in student['texts']:
        text_path = os.path.join(student['path'], text_file)
        try:
            with open(text_path, 'r', encoding='utf-8') as file_content:
                content = file_content.read()
                text_contents.append(content)
        except Exception as e:
            logging.error(f"無法讀取文件 {text_path}: {str(e)}")
    
    for image_file in student['images']:
        image_path = os.path.join(student['path'], image_file)
        try:
            if not os.path.exists(image_path):
                logging.error(f"文件不存在: {image_path}")
                continue
                
            # Log file size
            file_size = os.path.getsize(image_path)
            logging.info(f"處理文件: {image_path}, 大小: {file_size/1024:.2f}KB")
                
            try:
                mime_type = magic.from_file(image_path, mime=True)
                
                if mime_type == 'application/x-ipynb+json' or image_file.endswith('.ipynb'):
                    try:
                        pdf_path = convert_ipynb_to_pdf(image_path)
                        if pdf_path:
                            base64_images = convert_pdf_to_base64_images(pdf_path)
                            if base64_images:
                                image_base64_list.extend(base64_images)
                            else:
                                logging.error(f"PDF轉換失敗: {image_path}")
                        else:
                            logging.error(f"Notebook轉換失敗: {image_path}")
                    except Exception as e:
                        logging.error(f"Notebook處理錯誤 {image_path}: {str(e)}, 類型: {type(e).__name__}")
                
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    try:
                        base64_images = convert_docx_to_base64_images(image_path)
                        if base64_images:
                            image_base64_list.extend(base64_images)
                        else:
                            logging.error(f"DOCX轉換失敗: {image_path}")
                    except Exception as e:
                        logging.error(f"DOCX處理錯誤 {image_path}: {str(e)}, 類型: {type(e).__name__}")
                
                elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
                    try:
                        base64_images = convert_pptx_to_base64_images(image_path)
                        if base64_images:
                            image_base64_list.extend(base64_images)
                        else:
                            logging.error(f"PPTX轉換失敗: {image_path}")
                    except Exception as e:
                        logging.error(f"PPTX處理錯誤 {image_path}: {str(e)}, 類型: {type(e).__name__}")
                
                elif mime_type == 'application/pdf':
                    try:
                        base64_images = convert_pdf_to_base64_images(image_path)
                        if base64_images:
                            image_base64_list.extend(base64_images)
                        else:
                            logging.error(f"PDF轉換失敗: {image_path}")
                    except Exception as e:
                        logging.error(f"PDF處理錯誤 {image_path}: {str(e)}, 類型: {type(e).__name__}")
                
                elif mime_type.startswith('image/'):
                    try:
                        with Image.open(image_path) as img:
                            img_base64 = image_to_base64(img)
                            if img_base64:
                                image_base64_list.append(img_base64)
                            else:
                                logging.error(f"圖片轉換失敗: {image_path}")
                    except Exception as e:
                        logging.error(f"圖片處理錯誤 {image_path}: {str(e)}, 類型: {type(e).__name__}")
                
                else:
                    logging.warning(f"未知的檔案類型 {image_path}: {mime_type}")
                    
            except Exception as e:
                logging.error(f"檔案類型檢測失敗 {image_path}: {str(e)}")
                    
        except Exception as e:
            logging.error(f"處理文件時發生未知錯誤 {image_path}: {str(e)}")
    
    logging.info(f"完成處理 {student['id']}: {len(text_contents)}文本, {len(image_base64_list)}圖片")
    return image_base64_list, text_contents

SystemPrompt = '''
你是一位公正的教育者，負責根據指定的評分標準評估學生的作業。
對於每位學生，請根據其作業質量和對作業要求的遵守程度，給出100分制的分數。
請確保分數稍微偏高一些。同時，請用30個字以內的繁體中文提供簡短評語，說明評分理由。

請嚴格按照以下格式輸出，每位學生一行：
學號,分數,評語

注意事項：
1. 不要添加任何額外的文字或信息
2. 每行必須包含且僅包含：學號、分數、評語，用逗號分隔
3. 分數必須是0-100的整數
4. 評語必須是繁體中文，不超過30字
5. 不要使用引號或其他標點符號包裹內容

範例輸出格式：
411234567,85,作業內容完整，理解深入
411234568,92,解釋清晰，舉例恰當
'''

from langchain_openai import ChatOpenAI

def grade_single_student_anthropic(student, assignment_requirements):
    """Grade a single student's assignment using Anthropic's Claude"""
    try:
        image_base64_list, text_contents = read_files(student)
        
        if not image_base64_list and not text_contents:
            logging.warning(f"{student['id']}: 無作業內容")
            return None
            
        client = anthropic.Anthropic()
        
        # Prepare message content
        message_content = []
        
        # Add images first
        for image_base64 in image_base64_list:
            message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64
                }
            })
        
        # Add text content
        text_content = f"{SystemPrompt}\n\n作業要求：\n{assignment_requirements}\n\n"
        text_content += f"學生資訊：{student['id']} - {student['name']}\n\n"
        
        if text_contents:
            text_content += "作業內容：\n"
            for text in text_contents:
                text_content += text + "\n"
        
        text_content += "\n請評分這位學生的作業。"
        
        message_content.append({
            "type": "text",
            "text": text_content
        })
        
        # Make API call
        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": message_content
            }]
        )
        
        # Log the raw response for debugging
        logging.info(f"學生 {student['id']} Anthropic API Response: {message.content}")
        
        # Parse response
        line = message.content[0].text.strip().strip('"\'')
        if not line:
            return None
            
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            return {
                'id': parts[0].strip(),
                'score': parts[1].strip(),
                'comment': parts[2].strip()
            }
        else:
            logging.error(f"學生 {student['id']} 回應格式無效: {line}")
            return None
            
    except Exception as e:
        logging.error(f"評分學生 {student['id']} 時發生錯誤: {str(e)}")
        return None

def grade_single_student(student, assignment_requirements):
    """Grade a single student's assignment"""
    messages = [
        {"role": "system", "content": SystemPrompt},
        {"role": "user", "content": f"作業內容：\n{assignment_requirements}"}
    ]
    
    try:
        image_base64_list, text_contents = read_files(student)
        
        if not image_base64_list and not text_contents:
            logging.warning(f"{student['id']}: 無作業內容")
            return None
        
        messages.append({"role": "user", "content": f"學生資訊：{student['id']} - {student['name']}"})
        
        for image_base64 in image_base64_list:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        }
                    },
                ],
            })
        
        content = f"學生資訊：{student['id']} - {student['name']}\n作業內容如下：\n"
        for content_part in text_contents:
            content += content_part + "\n"
        messages.append({"role": "user", "content": content})
        
        messages.append({"role": "user", "content": "請評分這位學生的作業。"})
        
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0, max_tokens=300)
        response = llm.invoke(messages)
        
        # Log the raw response for debugging
        logging.info(f"學生 {student['id']} API Response: {response.content}")
        
        # Parse response
        line = response.content.strip().strip('"\'')
        if not line:
            return None
            
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            return {
                'id': parts[0].strip(),
                'score': parts[1].strip(),
                'comment': parts[2].strip()
            }
        else:
            logging.error(f"學生 {student['id']} 回應格式無效: {line}")
            return None
            
    except Exception as e:
        logging.error(f"評分學生 {student['id']} 時發生錯誤: {str(e)}")
        return None

def grade_batch_assignments(students_batch, use_anthropic=False):
    """Grade a batch of students' assignments"""
    grades = []
    logging.info(f"評分批次: {len(students_batch)}位學生")
    
    for student in students_batch:
        if not student['has_attachments']:
            continue
            
        try:
            if use_anthropic:
                grade = grade_single_student_anthropic(student, assignment_requirements)
            else:
                grade = grade_single_student(student, assignment_requirements)
            if grade:
                grades.append(grade)
        except Exception as e:
            logging.error(f"批次處理學生 {student['id']} 時發生錯誤: {str(e)}")
    
    logging.info(f"完成評分: {len(grades)}份")
    return grades
    
def process_file(file_path, images, texts, depth=0, max_depth=10):
    """Process a single file and categorize it as either image or text"""
    try:
        # 先檢查是否為目錄
        if os.path.isdir(file_path):
            # 如果是目錄，遍歷其中的檔案（目錄遍歷不計入遞迴深度）
            for f in os.listdir(file_path):
                try:
                    sub_path = os.path.join(file_path, f)
                    process_file(sub_path, images, texts, depth, max_depth)
                except Exception as e:
                    logging.error(f"處理子檔案失敗 {f}: {str(e)}")
            return

        # Normalize file path and handle encoding
        file_path = os.path.normpath(file_path)
        file_path = os.path.abspath(file_path)
        
        # Try different encodings if file not found
        if not os.path.exists(file_path):
            encodings = ['utf-8', 'big5', 'gbk', 'latin1']
            found = False
            for encoding in encodings:
                try:
                    encoded_path = file_path.encode(encoding).decode('utf-8')
                    if os.path.exists(encoded_path):
                        file_path = encoded_path
                        found = True
                        break
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
            
            if not found:
                logging.error(f"文件不存在或編碼問題: {file_path}")
                return
            
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            logging.error(f"文件無法讀取: {file_path}")
            return
            
        # Get file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logging.error(f"文件大小為0: {file_path}")
                return
        except OSError as e:
            logging.error(f"無法獲取文件大小 {file_path}: {str(e)}")
            return
            
        # Get file extension and mime type
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            mime_type = magic.from_file(file_path, mime=True)
        except Exception as e:
            logging.error(f"無法確定文件類型 {file_path}: {str(e)}")
            return
        
        # Skip processing if it's an Office file
        if file_ext in ['.docx', '.pptx', '.xlsx']:
            if file_ext == '.pptx':
                images.append(os.path.basename(file_path))
            return
            
        # Handle regular zip files
        if mime_type == 'application/zip' or file_ext == '.zip':
            try:
                extract_dir = os.path.dirname(file_path)
                
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    try:
                        # Get list of files before extraction
                        file_list = zip_ref.namelist()
                        
                        # Check for potential path traversal and file size
                        total_size = 0
                        max_size = 500 * 1024 * 1024  # 500MB limit
                        valid_files = []
                        
                        for fname in file_list:
                            if fname.startswith('/') or '..' in fname:
                                logging.warning(f"跳過不安全的路徑: {fname}")
                                continue
                                
                            try:
                                info = zip_ref.getinfo(fname)
                                total_size += info.file_size
                                if total_size > max_size:
                                    logging.error(f"壓縮檔太大: {file_path}")
                                    return
                                valid_files.append(fname)
                            except Exception as e:
                                logging.error(f"無法獲取檔案資訊 {fname}: {str(e)}")
                                continue
                        
                        # 創建臨時解壓目錄
                        temp_extract_dir = os.path.join(extract_dir, 'temp_extract')
                        os.makedirs(temp_extract_dir, exist_ok=True)
                        
                        # Extract only valid files to temp directory
                        for fname in valid_files:
                            try:
                                zip_ref.extract(fname, temp_extract_dir)
                            except Exception as e:
                                logging.error(f"解壓縮檔案失敗 {fname}: {str(e)}")
                                continue
                        
                        # 處理解壓縮後的檔案並複製到原始目錄
                        for fname in valid_files:
                            try:
                                # 處理可能的編碼問題
                                try:
                                    temp_file = os.path.join(temp_extract_dir, fname)
                                    target_file = os.path.join(extract_dir, fname)
                                except (UnicodeEncodeError, UnicodeDecodeError):
                                    # 嘗試不同編碼
                                    for encoding in ['utf-8', 'big5', 'gbk', 'latin1']:
                                        try:
                                            encoded_fname = fname.encode('latin1').decode(encoding)
                                            temp_file = os.path.join(temp_extract_dir, encoded_fname)
                                            target_file = os.path.join(extract_dir, encoded_fname)
                                            break
                                        except (UnicodeEncodeError, UnicodeDecodeError):
                                            continue
                                    else:
                                        logging.error(f"無法處理檔案名稱編碼: {fname}")
                                        continue
                                
                                if os.path.isfile(temp_file) and os.path.basename(temp_file) != os.path.basename(file_path):
                                    try:
                                        # 創建目標文件的目錄（如果需要）
                                        target_dir = os.path.dirname(target_file)
                                        if target_dir:
                                            os.makedirs(target_dir, exist_ok=True)
                                        # 複製文件到原始目錄
                                        shutil.copy2(temp_file, target_file)
                                    except Exception as e:
                                        logging.error(f"複製檔案失敗 {temp_file} -> {target_file}: {str(e)}")
                                        continue
                                    
                                    if depth < max_depth:
                                        process_file(target_file, images, texts, depth + 1, max_depth)
                                    else:
                                        logging.warning(f"達到最大遞迴深度，跳過處理: {target_file}")
                            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                                # 嘗試不同編碼
                                for encoding in ['utf-8', 'big5', 'gbk', 'latin1']:
                                    try:
                                        encoded_path = fname.encode('latin1').decode(encoding)
                                        extracted_file = os.path.join(temp_extract_dir, encoded_path)
                                        if os.path.isfile(extracted_file):
                                            if depth < max_depth:
                                                process_file(extracted_file, images, texts, depth + 1, max_depth)
                                            break
                                    except (UnicodeEncodeError, UnicodeDecodeError):
                                        continue
                                
                    except Exception as e:
                        logging.error(f"處理壓縮檔時發生錯誤 {file_path}: {str(e)}")
                            
            except zipfile.BadZipFile:
                logging.error(f"Bad zip file: {file_path}")
            except Exception as e:
                logging.error(f"Error processing zip file {file_path}: {str(e)}")
            finally:
                # 清理臨時目錄
                if os.path.exists(temp_extract_dir):
                    try:
                        shutil.rmtree(temp_extract_dir)
                    except Exception as e:
                        logging.error(f"清理臨時目錄失敗 {temp_extract_dir}: {str(e)}")
            return
        
        # Get base filename without any additional extensions
        base_name = os.path.basename(file_path)
        
        # Map MIME types to appropriate categories
        if any(mime_type.startswith(t) for t in ['image/', 'application/pdf']):
            images.append(base_name)
        elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
            images.append(base_name)
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            images.append(base_name)
        elif mime_type == 'application/x-ipynb+json' or base_name.endswith('.ipynb'):  # Special case for Jupyter
            images.append(base_name)
        elif any(mime_type.startswith(t) for t in ['text/', 'application/x-']):
            # Common programming and text file types
            if not base_name.lower().endswith(('.exe', '.dll', '.so', '.dylib')):  # Exclude binaries
                texts.append(base_name)
    except Exception as e:
        logging.error(f"檔案處理失敗 {file_path}: {str(e)}")

# 遍歷學生資料夾
students = []
for student_folder in os.listdir(HOMEWORK_DIR):
    student_path = os.path.join(HOMEWORK_DIR, student_folder)
    if os.path.isdir(student_path):
        try:
            # 處理資料夾名稱編碼問題
            try:
                folder_name = student_folder
                if not isinstance(folder_name, str):
                    # 如果不是字符串，嘗試解碼
                    encodings = ['utf-8', 'big5', 'gbk', 'latin1']
                    for encoding in encodings:
                        try:
                            folder_name = folder_name.decode(encoding)
                            break
                        except (UnicodeDecodeError, AttributeError):
                            continue
                    else:
                        logging.error(f"無法解碼資料夾名稱: {student_folder}")
                        continue
                
                # 分割學號和姓名
                parts = folder_name.split('_', 1)
                if len(parts) != 2:
                    logging.warning(f"無效的資料夾名稱格式: {folder_name}")
                    continue
                student_id, student_name = parts
            except Exception as e:
                logging.error(f"處理資料夾名稱時發生錯誤: {student_folder}, 錯誤: {str(e)}")
                continue
        except ValueError:
            logging.warning(f"無效的學生文件夾名稱: {student_folder}")
            continue
        
        if student_folder.endswith('無附件'):
            students.append({
                'id': student_id,
                'name': student_name.replace('無附件',''),
                'images': [],
                'texts': [],
                'path': student_path,
                'has_attachments': False
            })
        else:
            # Process student files
            try:
                if not os.path.exists(student_path):
                    logging.error(f"學生目錄不存在: {student_path}")
                    students.append({
                        'id': student_id,
                        'name': student_name,
                        'images': [],
                        'texts': [],
                        'path': student_path,
                        'has_attachments': False
                    })
                    continue

                files = os.listdir(student_path)
                if not files:
                    logging.warning(f"學生目錄為空: {student_path}")
                    students.append({
                        'id': student_id,
                        'name': student_name,
                        'images': [],
                        'texts': [],
                        'path': student_path,
                        'has_attachments': False
                    })
                    continue

                images = []
                texts = []
                
                for f in files:
                    try:
                        file_path = os.path.join(student_path, f)
                        process_file(file_path, images, texts, depth=0, max_depth=10)
                    except Exception as e:
                        logging.error(f"處理文件失敗 {file_path}: {str(e)}")
                        continue
                
                students.append({
                    'id': student_id,
                    'name': student_name,
                    'images': images,
                    'texts': texts,
                    'path': student_path,
                    'has_attachments': len(images) > 0 or len(texts) > 0
                })
            except Exception as e:
                logging.error(f"處理學生目錄失敗 {student_path}: {str(e)}")
                students.append({
                    'id': student_id,
                    'name': student_name,
                    'images': [],
                    'texts': [],
                    'path': student_path,
                    'has_attachments': False
                })

logging.info(f"總共找到 {len(students)} 位學生的作業")

# 開始評分
output_file_name = HOMEWORK_DIR.split('/')[-1].split('_')[1]
with open(f'{output_file_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['學號', '姓名', '分數', '評語']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for i in range(0, len(students), 10):
        batch = students[i:i+10]
        grades = grade_batch_assignments([s for s in batch if s['has_attachments']], use_anthropic=use_anthropic)
        
        for student in batch:
            if not student['has_attachments']:
                writer.writerow({'學號': student['id'], '姓名': student['name'], '分數': 0, '評語': '未繳交'})
                print({'學號': student['id'], '姓名': student['name'], '分數': 0, '評語': '未繳交'})
            else:
                grade = next((g for g in grades if g['id'] == student['id']), None)
                if grade:
                    writer.writerow({'學號': student['id'], '姓名': student['name'], '分數': grade['score'], '評語': grade['comment']})
                    print({'學號': student['id'], '姓名': student['name'], '分數': grade['score'], '評語': grade['comment']})
                else:
                    writer.writerow({'學號': student['id'], '姓名': student['name'], '分數': 0, '評語': '讀取異常'})
                    print({'學號': student['id'], '姓名': student['name'], '分數': 0, '評語': '讀取異常'})
                    logging.warning(f"{student['id']}: 讀取異常")

print("評分完成。請檢查 grading.log 文件以獲取詳細的錯誤信息。")
