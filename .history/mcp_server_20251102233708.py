import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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
    # API 키가 설정되지 않으면 서버를 실행하지 않도록 오류 발생
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = genai.Client()

# 메모리 캐시 설정 (테스트 중 동일 프롬프트 재사용 방지)
# set_llm_cache(InMemoryCache()) # LangChain 제거로 인해 사용하지 않음

# FastAPI 앱 인스턴스 생성 (MCP 서버의 몸통)
app = FastAPI(title="LLM 기반 MCP 자동화 서버", version="1.0")

# ==============================================================================
# 2. 문서 파싱 및 멀티 모달 처리 (천우성 팀원 담당 기능)
# ==============================================================================

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """PDF, DOCX 파일에서 텍스트를 추출하고, 실패 시 이미지 분석을 위한 Base64 데이터를 반환"""
    file_extension = os.path.splitext(filename)[1].lower()
    
    # 1. PDF 파일 처리 로직
    if file_extension == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_content))
            text = ""
            # 모든 페이지 텍스트 추출
            for page in reader.pages:
                text += page.extract_text() or ""
            
            # 텍스트 추출 성공 시
            if len(text.strip()) > 50: 
                return text.strip()
            
            # 텍스트 추출 실패 시 (이미지 PDF로 간주하고 Base64로 전달)
            else:
                return base64.b64encode(file_content).decode("utf-8")
                
        except Exception as e:
            # PDF 파싱 중 오류 발생 시, Base64 데이터 반환 (이미지 분석 우회)
            print(f"PDF 텍스트 추출 오류: {e}. Base64로 전환.")
            return base64.b64encode(file_content).decode("utf-8")


    # 2. DOCX 파일 처리 로직
    elif file_extension == ".docx":
        try:
            document = Document(io.BytesIO(file_content))
            text = "\n".join([paragraph.text for paragraph in document.paragraphs])
            return text.strip()
        except Exception:
            return f"Error: DOCX 텍스트 추출 실패."

    # 3. TXT 파일 처리 로직
    elif file_extension == ".txt":
        return file_content.decode('utf-8', errors='ignore').strip()

    return ""

def create_multimodal_parts(file_content: bytes, file_mime_type: str, extracted_text: str) -> List[Any]:
    """Gemini API에 전달할 content parts 생성 (텍스트와 이미지)"""
    parts = []
    
    # 1. 멀티 모달 - 이미지 파트 (JPEG, PNG 등)
    if file_mime_type.startswith('image/'):
        try:
            # 이미지 파일을 PIL Image 객체로 변환
            img = Image.open(io.BytesIO(file_content))
            parts.append(img)
        except Exception:
            # 이미지 파일이 유효하지 않을 경우 텍스트로 처리
            parts.append(f"주의: 이미지 파일 로드 실패. 파일명: {file_mime_type}")
    
    # 2. 텍스트 파트 (추출된 텍스트 또는 문서 파일)
    if extracted_text:
        parts.append(f"--- 추출된 문서 텍스트 ---\n{extracted_text}")
    elif file_mime_type.startswith('application/pdf'):
        # PDF 텍스트 추출 실패 시, Base64로 넘어온 경우를 대비하여 텍스트로 추가
        # Base64 데이터는 run_multimodal_pipeline에서 직접 처리되므로 여기서는 텍스트만 추가
        parts.append(f"주의: PDF에서 텍스트 추출이 어렵거나 내용이 매우 적음. Base64 데이터 기반으로 이미지 분석을 시도합니다.")
        
    return parts

# ==============================================================================
# 3. Controller 정의 (지능적인 모델 라우팅)
# ==============================================================================

def determine_routing(target_tag: str) -> Tuple[str, str]:
    """요청 태그에 따라 사용할 LLM 모델을 결정하고 프롬프트 스타일을 반환"""
    
    # 요청 태그를 소문자로 변환하여 비교
    tag = target_tag.lower().strip()

    if "분석" in tag:
        # 논리 분석 및 구조화에 강한 모델 (GPT-4 역할)
        return "gemini-2.5-pro", "당신은 세계적인 분석가입니다. 추출된 문서를 비즈니스 관점에서 깊이 분석하여 보고서 초안을 작성하고, 명확한 근거와 시사점을 제시하십시오."
    
    elif "문체" in tag:
        # 빠르고 저렴하며 문맥 이해가 좋은 모델 (Claude 역할)
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
        "위 지침과 사용자 요청을 바탕으로 제공된 문서 내용(텍스트 및 이미지)을 활용하여 최종 보고서를 작성하십시오."
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
    file: UploadFile = File(...),
    user_request: str = "사용자의 구체적인 요청 내용",
    target_tag: str = "Controller 라우팅 태그: '분석' 또는 '문체'",
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
