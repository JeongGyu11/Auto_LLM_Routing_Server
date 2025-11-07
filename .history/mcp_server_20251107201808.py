import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any

# 문서 파싱 라이브러리
from pypdf import PdfReader
from docx import Document

# 이미지 처리 라이브러리
from PIL import Image

# Gemini 연동 라이브러리
from google import genai
from google.genai.errors import APIError

# 1. API 키 설정 확인 및 클라이언트 연결
# 코드를 실행하기 전에 터미널에서 $env:GEMINI_API_KEY="YOUR_KEY"를 반드시 실행해야 합니다.
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = genai.Client()

# FastAPI 앱 인스턴스 생성 (MCP 서버의 몸통)
app = FastAPI(title="LLM 기반 MCP 자동화 서버", version="1.0")

# ==============================================================================
# 1.1. CORS 설정 (프론트엔드 연결 허용) - [Vercel HTTPS 주소 추가]
# ==============================================================================
# Vercel 배포 주소 및 로컬 개발 환경 모두 허용
origins = [
    "http://localhost:3000",        # 로컬 개발 환경
    "http://127.0.0.1:3000",        # 로컬 테스트용
    "https://auto-llm-routing.vercel.app", # Vercel의 주 도메인 (정확한 주소로 변경 필요)
    "https://*.vercel.app",         # 모든 Vercel 서브 도메인 허용
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (POST, GET 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# ==============================================================================
# 2. 문서 파싱 및 멀티 모달 처리 (천우성 팀원 담당 기능)
# ==============================================================================

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """PDF, DOCX 파일에서 텍스트를 추출합니다. 텍스트가 부족할 경우 LLM에게 알립니다."""
    file_extension = os.path.splitext(filename)[1].lower()
    
    # 1. PDF 파일 처리 로직
    if file_extension == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            if len(text.strip()) > 50: 
                return text.strip()
            else:
                return "[텍스트 추출 성공, 하지만 내용이 너무 적어 분석이 어려움]" 
                
        except Exception:
            return "[PDF 텍스트 추출 실패: OCR(이미지 파싱) 기능 필요]"


    # 2. DOCX 파일 처리 로직
    elif file_extension == ".docx":
        try:
            document = Document(io.BytesIO(file_content))
            text = "\n".join([paragraph.text for paragraph in document.paragraphs])
            return text.strip()
        except Exception:
            return "Error: DOCX 텍스트 추출 실패."

    # 3. TXT 파일 처리 로직
    elif file_extension == ".txt":
        try:
            return file_content.decode('utf-8', errors='ignore').strip()
        except Exception:
            return "TXT 파일 디코딩 오류."

    return "지원되는 문서 파일 형식이 아닙니다."

def create_multimodal_parts(file_content: bytes, file_mime_type: str, extracted_text: str) -> List[Any]:
    """Gemini API에 전달할 content parts 생성 (텍스트와 이미지)"""
    parts = []
    
    # 1. 일반 이미지 파일 처리 (JPEG, PNG 등)
    if file_mime_type.startswith('image/'):
        try:
            img = Image.open(io.BytesIO(file_content))
            parts.append(img)
            parts.append("제공된 이미지를 분석하여 보고서를 작성하십시오.")
        except Exception:
            parts.append(f"주의: 이미지 파일 로드 실패. 파일명: {file_mime_type}")
    
    # 2. 텍스트 파트 (성공적으로 추출된 텍스트)
    if extracted_text and extracted_text != "지원되는 문서 파일 형식이 아닙니다.":
        parts.append(f"--- 추출된 문서 텍스트 ---\n{extracted_text}")
        
    return parts

# ==============================================================================
# 3. Controller 정의 (지능적인 모델 라우팅)
# ==============================================================================

def determine_routing(target_tag: str) -> Tuple[str, str]:
    """요청 태그에 따라 사용할 LLM 모델을 결정하고 프롬프트 스타일을 반환"""
    
    # Controller 사용 강제: 태그가 없거나 유효하지 않으면 ValueError 발생
    tag = target_tag.lower().strip()

    if "분석" in tag:
        # 논리 분석 및 구조화에 강한 모델 (Pro 모델 지정)
        return "gemini-2.5-pro", "당신은 세계적인 분석가입니다. 제공된 문서 및 시각 데이터를 **최우선으로 분석**하여 보고서 초안을 작성하고, 명확한 근거와 시사점을 제시하십시오."
    
    elif "문체" in tag:
        # 빠르고 저렴하며 문맥 이해가 좋은 모델 (Flash 모델 지정)
        return "gemini-2.5-flash", "당신은 전문 편집자입니다. 추출된 텍스트를 검토하여 문법 오류를 수정하고, 읽기 쉽고 전문적인 보고서 문체로 내용을 다듬어 최종본을 완성하십시오."
    
    else:
        # 태그가 없거나 유효하지 않으면 오류 반환 (Controller 사용 강제)
        raise ValueError("Controller 라우팅을 위해 '분석' 또는 '문체' 태그를 반드시 포함해야 합니다.")

# ==============================================================================
# 4. Pipeline 정의 (LLM 실행 흐름)
# ==============================================================================

def run_multimodal_pipeline(
    parts: List[Any],
    model_name: str,
    prompt_style: str,
    user_request: str
) -> Tuple[str, str]:
    """멀티 모달 데이터를 받아 MCP 파이프라인을 실행"""

    # 1. 최종 프롬프트 구성 (사용자 요청, 스타일, 데이터 통합)
    final_prompt = [
        f"--- 최종 지침: {prompt_style} ---",
        f"--- 사용자 요청: {user_request} ---",
        "**최종 결과물은 반드시 사용자 요청({user_request})의 형식과 길이(예: 3줄 요약)를 철저히 준수해야 합니다.** 위 지침과 사용자 요청을 바탕으로 제공된 문서 내용(텍스트 및 시각 자료)을 활용하여 최종 보고서를 작성하십시오. 시각 자료가 제공되었다면 반드시 그 내용을 분석에 통합해야 합니다."
    ]
    
    # final_prompt 리스트를 parts의 시작 부분에 추가하여 컨텍스트 제공
    parts_with_prompt = final_prompt + parts

    # 2. LLM 호출 실행 (Pipeline)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=parts_with_prompt,
        )
        # 3. 로그 생성
        log = f"Controller가 '{prompt_style[:10]}...' 스타일로 요청을 받아 '{model_name}' 모델을 성공적으로 호출했습니다."
        return response.text, log

    except APIError as e:
        error_msg = f"API 호출 중 오류 발생: {e}. API 키 또는 모델 이름 확인 필요."
        return error_msg, f"오류 발생: {model_name} 호출 실패."
    except Exception as e:
        error_msg = f"파이프라인 실행 중 예기치 않은 오류 발생: {e}"
        return error_msg, f"오류 발생: {model_name} 실행 중 예외 처리."


# ==============================================================================
# 5. FastAPI 엔드포인트 (프론트엔드 연결 지점)
# ==============================================================================

class ProcessResponse(BaseModel):
    status: str
    final_report: str
    process_log: str
    model_used: str

@app.post("/api/process_document", response_model=ProcessResponse, tags=["default"])
async def process_document(
    file: UploadFile = File(..., description="문서 및 사진 파일 (PDF, DOCX, TXT, PNG, JPG)"),
    user_request: str = Form(..., description="문서 분석을 위한 구체적인 요청 내용"),
    target_tag: str = Form(..., description="Controller 라우팅 태그: '분석' 또는 '문체'"),
):
    """문서 및 사진을 받아 복합 LLM 라우팅 파이프라인을 실행합니다."""
    
    file_content = await file.read()
    filename = file.filename
    file_mime_type = file.content_type

    try:
        # 1. Controller 실행: 라우팅 모델과 프롬프트 스타일 결정
        model_name, prompt_style = determine_routing(target_tag)

        # 2. 문서 파싱 실행: 텍스트 추출 (천우성 팀원 담당)
        extracted_text = extract_text_from_file(file_content, filename)
        
        # 3. 멀티 모달 파트 준비 (글/이미지 결합)
        parts = create_multimodal_parts(file_content, file_mime_type, extracted_text)

        # 4. Pipeline 실행
        final_report, process_log = run_multimodal_pipeline(
            parts=parts,
            model_name=model_name,
            prompt_style=prompt_style,
            user_request=user_request
        )

        return ProcessResponse(
            status="success",
            final_report=final_report,
            process_log=process_log,
            model_used=model_name
        )
    
    except ValueError as e:
        # Controller에서 발생한 오류 (태그 미입력 등)
        return JSONResponse(
            status_code=400,
            content={"detail": str(e)}
        )
    except Exception as e:
        # 기타 모든 예외 처리
        return JSONResponse(
            status_code=500,
            content={"detail": f"서버 내부 오류 발생: {e}"}
        )