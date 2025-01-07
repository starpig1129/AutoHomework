import os
import csv
import base64
import openai
import logging
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
#os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

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
                
            if image_file.lower().endswith('.ipynb'):
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
                    logging.error(f"Notebook處理錯誤 {image_path}: {str(e)}")
                    
            elif image_file.lower().endswith('.docx'):
                try:
                    base64_images = convert_docx_to_base64_images(image_path)
                    if base64_images:
                        image_base64_list.extend(base64_images)
                    else:
                        logging.error(f"DOCX轉換失敗: {image_path}")
                except Exception as e:
                    logging.error(f"DOCX處理錯誤 {image_path}: {str(e)}")
                    
            elif image_file.lower().endswith('.pptx'):
                try:
                    base64_images = convert_pptx_to_base64_images(image_path)
                    if base64_images:
                        image_base64_list.extend(base64_images)
                    else:
                        logging.error(f"PPTX轉換失敗: {image_path}")
                except Exception as e:
                    logging.error(f"PPTX處理錯誤 {image_path}: {str(e)}")
                    
            elif image_file.lower().endswith('.pdf'):
                try:
                    base64_images = convert_pdf_to_base64_images(image_path)
                    if base64_images:
                        image_base64_list.extend(base64_images)
                    else:
                        logging.error(f"PDF轉換失敗: {image_path}")
                except Exception as e:
                    logging.error(f"PDF處理錯誤 {image_path}: {str(e)}")
                    
            else:
                try:
                    with Image.open(image_path) as img:
                        img_base64 = image_to_base64(img)
                        if img_base64:
                            image_base64_list.append(img_base64)
                        else:
                            logging.error(f"圖片轉換失敗: {image_path}")
                except Exception as e:
                    logging.error(f"圖片處理錯誤 {image_path}: {str(e)}")
                    
        except Exception as e:
            logging.error(f"處理文件時發生未知錯誤 {image_path}: {str(e)}")
    
    logging.info(f"完成處理 {student['id']}: {len(text_contents)}文本, {len(image_base64_list)}圖片")
    return image_base64_list, text_contents

SystemPrompt = '''
As an impartial and neutral educator, 
your objective is to evaluate students' submitted assignments according to the specified grading criteria. 
For each student, please assign a score out of 100, 
based on the quality of their work and their adherence to the assignment requirements. 
Ensure that the scores are slightly higher than usual. 
Additionally, provide a brief comment in Traditional Chinese, 
limited to 30 characters, explaining the reason for the assigned score.

Ensure your output follows this structured format:
"Student ID, score, comment."
Refrain from including any additional text or information.
'''

from langchain_openai import ChatOpenAI

def grade_batch_assignments(students_batch):
    messages = [
        {"role": "system", "content": SystemPrompt},
        {"role": "user", "content": f"作業內容：\n{assignment_requirements}"}
    ]
    
    for student in students_batch:
        try:
            image_base64_list, text_contents = read_files(student)
            
            if not image_base64_list and not text_contents:
                logging.warning(f"{student['id']}: 無作業內容")
                continue
            
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
            
            for content in text_contents:
                messages.append({"role": "user", "content": content})
            
            messages.append({"role": "user", "content": "請評分這位學生的作業。"})
        except Exception as e:
            logging.error(f"處理學生 {student['id']} 的作業時發生錯誤: {str(e)}")
    
    try:
        logging.info(f"評分批次: {len(students_batch)}位學生")
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0, max_tokens=300)
        response = llm.invoke(messages)
        
        grades = []
        for line in response.content.strip().split('\n'):
            parts = line.strip().split(',')
            if len(parts) == 3:
                grades.append({
                    'id': parts[0].strip(),
                    'score': parts[1].strip(),
                    'comment': parts[2].strip()
                })
        logging.info(f"完成評分: {len(grades)}份")
        return grades
    except Exception as e:
        logging.error(f"調用 API 時發生錯誤: {str(e)}")
        return []
    
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
            files = os.listdir(student_path)
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg','.pdf','.ipynb','.docx', '.pptx'))]
            texts = [f for f in files if f.lower().endswith(('.py', '.java', '.cpp',  '.txt',))]
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
        grades = grade_batch_assignments([s for s in batch if s['has_attachments']])
        
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
