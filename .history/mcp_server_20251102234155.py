# 1. 환경 설정 및 라이브러리 임포트
import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# 문서 파싱을 위한 라이브러리 (천우성 팀원 담당)
from pypdf import PdfReader
from docx import Document as DocxDocument

# Google Gemini API 라이브러리
from google import genai
from google.genai.errors import APIError

app = FastAPI(title="LLM 기반 MCP 자동화 서버", version="1.0")

# 2. Model 정의 및 클라이언트 초기화
# API 키 설정 확인 (환경 변수에서 불러옴)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # API 키가 설정되지 않으면 서버를 시작하지 않고 오류를 발생시킵니다.
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

# Gemini 클라이언트 초기화
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    raise RuntimeError(f"Gemini Client 초기화 오류: {e}")

# 3. Controller 정의 (지휘관 로직)
def determine_routing(target_tag: str):
    """
    Controller 역할: 요청 태그에 따라 사용할 LLM과 프롬프트를 결정합니다.
    """
    if target_tag == "분석":
        # 복잡한 논리 분석 및 초안 작성에는 Pro 모델 (GPT 역할)
        model_name = "gemini-2.5-pro"
        system_prompt = "당신은 최고의 금융 분석가입니다. 입력된 데이터를 기반으로 논리적인 구조의 심층 분석 보고서 초안을 작성하십시오. 핵심 수치를 명확히 제시하십시오."
    
    elif target_tag == "문체":
        # 문체 다듬기 및 효율성이 중요한 작업은 Flash 모델 (Claude 역할)
        model_name = "gemini-2.5-flash"
        system_prompt = "당신은 문장력을 최고로 다듬는 전문 편집자입니다. 입력된 텍스트를 전문적이고 자연스러운 문체로 다듬고, 핵심만 간결하게 요약하십시오."
    
    else:
        # 기타 요청 (기본값)
        model_name = "gemini-2.5-flash"
        system_prompt = "요청에 따라 내용을 전문적으로 처리하고 응답하십시오."
        
    return model_name, system_prompt

# 4. 파일에서 텍스트를 추출하는 보조 함수 (천우성 팀원 담당)
def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """PDF 및 DOCX 파일에서 텍스트를 추출합니다."""
    
    if filename.lower().endswith('.pdf'):
        try:
            reader = PdfReader(BytesIO(file_content))
            # 텍스트 추출 시도. 실패하면 빈 문자열 반환
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text if text else "[텍스트 추출 실패: PDF는 이미지로 분석을 시도합니다.]"
        except Exception:
            return "[PDF 파싱 중 오류 발생: 이미지만 분석을 시도합니다.]"
            
    elif filename.lower().endswith(('.docx', '.doc')):
        try:
            doc = DocxDocument(BytesIO(file_content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
            return text if text else "DOCX 파일에서 텍스트를 추출하지 못했습니다."
        except Exception:
            return "DOCX 파싱 중 오류가 발생했습니다."
            
    elif filename.lower().endswith('.txt'):
        try:
            return file_content.decode('utf-8')
        except Exception:
            return "TXT 파일 디코딩 오류."
            
    return "지원되는 문서 파일 형식이 아닙니다."


# 5. Pipeline 정의 (멀티 모달 처리 및 실행 흐름)
async def run_multimodal_pipeline(file_content: bytes, file_name: str, file_mime: str, user_request: str, target_tag: str):
    """
    Pipeline 역할: Vision 분석 → Controller 판단 → 최종 LLM 실행까지의 흐름을 정의합니다.
    """
    
    # 5-1. Step 1: 문서 텍스트 추출 및 이미지 객체 준비
    text_data = extract_text_from_file(file_content, file_name)
    image_data = None
    
    is_image_or_pdf = file_mime.startswith('image/') or file_name.lower().endswith('.pdf')

    if is_image_or_pdf:
        # 이미지 파일 또는 PDF 파일인 경우 PIL Image 객체로 변환 시도 (PDF는 이미지로 간주)
        try:
            image_data = Image.open(BytesIO(file_content))
        except Exception:
            # 이미지나 PDF를 이미지로 여는 데 실패할 경우
            if file_mime.startswith('image/'):
                raise HTTPException(status_code=400, detail="이미지 파일이 손상되었거나 형식을 읽을 수 없습니다.")
            # PDF는 텍스트 추출이라도 성공했으면 진행
            image_data = None
            
    # 멀티모달 LLM 호출을 위한 contents 리스트 준비
    contents_list = []
    
    # 텍스트 추출에 실패했으나 이미지 파일인 경우를 위해, 텍스트 데이터가 없어도 이미지 데이터를 넣을 수 있도록 로직 보강
    if image_data:
        contents_list.append(image_data)
        
    # 추출된 텍스트가 있다면 프롬프트에 추가 (실패 메시지라도 LLM에게 전달하여 상황 인지하도록 함)
    if text_data and text_data != "지원되는 문서 파일 형식이 아닙니다.":
        contents_list.append(f"[문서 텍스트]: {text_data}")
    
    # 텍스트 데이터도 없고, 이미지 데이터도 없으면 오류 발생
    if not contents_list:
        raise HTTPException(status_code=400, detail="파일에서 추출할 수 있는 내용(텍스트 또는 이미지)이 없습니다.")


    # 5-2. Step 2: Controller 판단 및 LLM 선택
    model_name, system_prompt = determine_routing(target_tag)
    
    # 최종 프롬프트 구성
    final_prompt = f"""
    {system_prompt}
    
    [사용자 요청]: "{user_request}"
    
    위 요청을 처리하기 위해 이미 전달된 문서 텍스트와 이미지를 종합적으로 분석하여 보고서를 작성하십시오.
    """
    
    contents_list.append(final_prompt)


    # 5-3. Step 3: 최종 LLM 실행 (Controller 지시에 따름)
    try:
        final_response = client.models.generate_content(
            model=model_name,
            contents=contents_list,
            config={"temperature": 0.3}
        )
        final_result = final_response.text
    except APIError as e:
        return f"최종 LLM 실행 오류: API 호출 실패 - {e}", ""

    process_log = f"Controller가 '{target_tag}' 태그에 따라 {model_name} 모델을 선택하여 최종 실행했습니다. (문서 파싱 포함)"
    
    return final_result, process_log


# 6. FastAPI 엔드포인트 (프론트와 연결되는 단일 API)
@app.post("/api/process_document")
async def process_document(
    file: UploadFile = File(..., description="문서 또는 이미지 파일 (PDF, DOCX, TXT, PNG, JPG)"),
    user_request: str = Form(..., description="사용자의 구체적인 요청 내용"),
    target_tag: str = Form(..., description="Controller 라우팅 태그: '분석' 또는 '문체'")
):
    """
    문서 및 사진을 받아 복합 LLM 라우팅 파이프라인을 실행합니다.
    """
    # 파일 내용과 메타데이터 추출
    contents = await file.read()
    
    # Pipeline 실행
    final_result, process_log = await run_multimodal_pipeline(
        file_content=contents,
        file_name=file.filename,
        file_mime=file.content_type,
        user_request=user_request,
        target_tag=target_tag
    )

    # 결과 반환
    return JSONResponse(content={
        "status": "success",
        "final_report": final_result,
        "process_log": process_log,
        "model_used": process_log.split(" ")[-3]
    })