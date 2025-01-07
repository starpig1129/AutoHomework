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
    
# 遍歷學生資料夾
students = []
for student_folder in os.listdir(HOMEWORK_DIR):
    student_path = os.path.join(HOMEWORK_DIR, student_folder)
    if os.path.isdir(student_path):
        try:
            student_id, student_name = student_folder.split('_', 1)
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
            def process_file(file_path, images, texts):
                try:
                    # Get file extension and mime type
                    file_ext = os.path.splitext(file_path)[1].lower()
                    mime_type = magic.from_file(file_path, mime=True)
                    
                    # Skip processing if it's an Office file
                    if file_ext in ['.docx', '.pptx', '.xlsx']:
                        if file_ext == '.pptx':
                            images.append(os.path.basename(file_path))
                        return
                        
                    # Handle regular zip files
                    if mime_type == 'application/zip':
                        temp_dir = os.path.join(os.path.dirname(file_path), 'temp_unzip')
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                            
                        # Process extracted files
                        for root, _, files in os.walk(temp_dir):
                            for f in files:
                                extracted_file = os.path.join(root, f)
                                process_file(extracted_file, images, texts)
                                
                        # Clean up
                        shutil.rmtree(temp_dir)
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

            files = os.listdir(student_path)
            images = []
            texts = []
            
            for f in files:
                file_path = os.path.join(student_path, f)
                process_file(file_path, images, texts)
            students.append({
                'id': student_id,
                'name': student_name,
                'images': images,
                'texts': texts,
                'path': student_path,
                'has_attachments': True
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
