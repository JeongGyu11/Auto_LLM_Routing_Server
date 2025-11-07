# 1. 환경 설정 및 라이브러리 임포트
import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# Google Gemini API 라이브러리만 사용
from google import genai
from google.genai.errors import APIError

app = FastAPI(title="LLM 기반 MCP 자동화 서버", version="1.0")

# 2. Model 정의 및 클라이언트 초기화
# API 키 설정 확인 (환경 변수에서 불러옴)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
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
        # 복잡한 논리 분석이 필요하므로, Pro 모델을 사용합니다.
        model_name = "gemini-2.5-pro"
        system_prompt = "당신은 최고의 금융 분석가입니다. 입력된 데이터를 기반으로 논리적인 구조의 심층 분석 보고서 초안을 작성하십시오. 핵심 수치를 명확히 제시해야 합니다."
    
    elif target_tag == "문체":
        # 문체 다듬기 및 효율성이 중요한 작업은 Flash 모델을 사용합니다.
        model_name = "gemini-2.5-flash"
        system_prompt = "당신은 문장력을 최고로 다듬는 전문 편집자입니다. 입력된 텍스트를 전문적이고 자연스러운 문체로 다듬고, 핵심만 간결하게 요약하십시오."
    
    else:
        # 기타 요청 (기본값)
        model_name = "gemini-2.5-flash"
        system_prompt = "요청에 따라 내용을 전문적으로 처리하고 응답하십시오."
        
    return model_name, system_prompt


# 4. Pipeline 정의 (멀티 모달 처리 및 실행 흐름)
async def run_multimodal_pipeline(image_data: Image.Image, text_data: str, target_tag: str, user_request: str):
    """
    Pipeline 역할: Vision 분석 → Controller 판단 → 최종 LLM 실행까지의 흐름을 정의합니다.
    """
    
    # 4-1. Pipeline Step 1: Vision 분석 (이미지 데이터를 텍스트로 전환)
    
    vision_prompt = f"""
    당신은 비정형 문서의 핵심 데이터를 추출하는 AI 전문가입니다.
    [문서 텍스트]: {text_data}
    [사용자 요청]: "{user_request}"
    요청을 처리하는 데 필요한 모든 데이터(표의 수치, 그래프의 추이)를 추출하고, 이미지에서 추출한 내용을 텍스트와 통합하여 최종 보고서 작성에 필요한 정리된 데이터셋을 생성하십시오.
    """
    
    # Vision 모델 호출: 이미지와 텍스트를 함께 전달합니다.
    try:
        vision_response = client.models.generate_content(
            model='gemini-2.5-flash', # Vision 기능이 포함된 Flash 모델 사용
            contents=[image_data, vision_prompt],
            config={"temperature": 0.1}
        )
        analysis_result = vision_response.text
        
    except APIError as e:
        return f"Vision 모델 분석 오류: API 호출 실패 - {e}", ""


    # 4-2. Pipeline Step 2: Controller 판단 및 LLM 선택
    model_name, system_prompt = determine_routing(target_tag)
    
    # 최종 프롬프트 구성
    final_prompt = f"""
    {system_prompt}
    
    이전에 분석된 데이터와 사용자의 최종 요청을 처리하십시오.
    
    [분석된 데이터]:
    ---
    {analysis_result}
    ---
    
    [최종 요청]:
    ---
    {user_request}
    ---
    """
    
    # 4-3. Pipeline Step 3: 최종 LLM 실행 (Controller 지시에 따름)
    try:
        final_response = client.models.generate_content(
            model=model_name,
            contents=[final_prompt],
            config={"temperature": 0.3}
        )
        final_result = final_response.text
    except APIError as e:
        return f"최종 LLM 실행 오류: API 호출 실패 - {e}", ""

    process_log = f"Controller가 '{target_tag}' 태그에 따라 {model_name} 모델을 선택하여 최종 실행했습니다."
    
    return final_result, process_log


# 5. FastAPI 엔드포인트 (프론트와 연결되는 단일 API)

@app.post("/api/process_document")
async def process_document(
    file: UploadFile = File(...),
    user_request: str = Form(..., description="사용자의 구체적인 요청 내용"),
    target_tag: str = Form(..., description="Controller 라우팅 태그: '분석' 또는 '문체'")
):
    """
    문서 및 사진을 받아 복합 LLM 라우팅 파이프라인을 실행합니다.
    """
    
    # 5-1. 파일 처리 및 이미지 추출
    if not (file.content_type.startswith('image/') or file.filename.endswith(('.pdf', '.docx', '.txt', '.png', '.jpg'))):
        raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다. 이미지, PDF, DOCX, TXT 파일을 사용하세요.")

    # 파일 내용을 메모리에 로드
    contents = await file.read()
    
    image_data = None
    # 이미지 파일은 PIL Image 객체로 변환
    if file.content_type.startswith('image/'):
        try:
            image_data = Image.open(BytesIO(contents))
        except Exception:
            # 이미지 파일이 손상된 경우를 대비
            pass

    # 임시 텍스트 데이터 (OCR/파싱 모듈이 없으므로 임시 데이터를 사용합니다 - 천우성 팀원 개발 예정)
    dummy_text_data = f"업로드된 파일명: {file.filename}. (임시 텍스트) 이 텍스트는 OCR을 통해 추출된 문서의 본문입니다. 이미지는 그래프, 표, 차트 등을 포함합니다."
    
    # 멀티 모달 처리를 위해 Image 객체가 필요합니다.
    if image_data is None:
        raise HTTPException(status_code=400, detail="파일이 이미지 형식이 아닙니다. 현재는 이미지 파일 또는 PDF 파일 내 이미지가 필요합니다. (PDF 파싱 모듈 구현 전)")
    
    # 5-2. Pipeline 실행
    final_result, process_log = await run_multimodal_pipeline(
        image_data=image_data,
        text_data=dummy_text_data,
        target_tag=target_tag,
        user_request=user_request
    )

    # 5-3. 결과 반환
    return JSONResponse(content={
        "status": "success",
        "final_report": final_result,
        "process_log": process_log,
        "model_used": process_log.split(" ")[-3]
    })
