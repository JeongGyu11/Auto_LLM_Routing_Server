# 1. 환경 설정 및 라이브러리 임포트
# MCP 서버는 FastAPI, LangChain, Gemini API를 사용합니다.
import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# LangChain과 Gemini 연동
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

# API 키 설정 확인 (환경 변수에서 불러옴)
# 코드를 실행하기 전에 터미널에서 export GEMINI_API_KEY="YOUR_KEY"를 반드시 실행해야 합니다.
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

# 메모리 캐시 설정 (테스트 중 동일 프롬프트 재사용 방지)
set_llm_cache(InMemoryCache())

app = FastAPI(title="LLM 기반 MCP 자동화 서버", version="1.0")

# 2. Model 정의 (LLM 전문가 팀)
# 역할을 분리하여 Model을 정의합니다. (Controller가 이들을 지휘합니다)
# -----------------------------------------------------------
# Vision 모델: 멀티 모달 처리 전문가
vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# Pro 모델: 논리 및 분석 전문가 (GPT 역할)
pro_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

# Flash 모델: 문체 완성 및 효율성 전문가 (Claude 역할)
flash_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
# -----------------------------------------------------------


# 3. Controller 정의 (지휘관 로직)
# 사용자의 요청 태그에 따라 최적의 LLM을 선택하고 파이프라인을 조정합니다.
def determine_routing(target_tag: str):
    """
    Controller 역할: 요청 태그에 따라 사용할 LLM과 프롬프트를 결정합니다.
    """
    if target_tag == "분석":
        # 복잡한 논리 분석이 필요하므로, Pro 모델을 사용합니다.
        llm = pro_model
        system_prompt = "당신은 최고의 금융 분석가입니다. 입력된 데이터를 기반으로 논리적인 구조의 심층 분석 보고서 초안을 작성하십시오. 핵심 수치를 명확히 제시해야 합니다."
    
    elif target_tag == "문체":
        # 문체 다듬기 및 요약 등 효율성이 중요한 작업은 Flash 모델을 사용합니다.
        llm = flash_model
        system_prompt = "당신은 문장력을 최고로 다듬는 전문 편집자입니다. 입력된 텍스트를 전문적이고 자연스러운 문체로 다듬고, 핵심만 간결하게 요약하십시오."
    
    else:
        # 기타 요청 (기본값)
        llm = flash_model
        system_prompt = "요청에 따라 내용을 전문적으로 처리하고 응답하십시오."
        
    return llm, system_prompt


# 4. Pipeline 정의 (멀티 모달 처리 및 실행 흐름)

def run_multimodal_pipeline(image_data: Image.Image, text_data: str, target_tag: str, user_request: str):
    """
    Pipeline 역할: Vision 분석 → Controller 판단 → 최종 LLM 실행까지의 흐름을 정의합니다.
    """
    
    # 4-1. Pipeline Step 1: Vision 분석 (이미지 데이터를 텍스트로 전환)
    # 이미지와 텍스트를 함께 Vision 모델에 넣어 분석합니다.
    vision_prompt = f"""
    당신은 비정형 문서의 핵심 데이터를 추출하는 AI 전문가입니다.
    다음은 문서의 텍스트와 이미지/그래프입니다.
    
    [문서 텍스트]:
    ---
    {text_data}
    ---
    
    [사용자 요청]: "{user_request}"
    
    요청을 처리하는 데 필요한 모든 데이터(표의 수치, 그래프의 추이, 핵심 문장)를 추출하고,
    이미지에서 추출한 내용을 텍스트와 통합하여 최종 보고서 작성에 필요한 정리된 데이터셋을 생성하십시오.
    """
    
    # Vision 모델 호출: 이미지와 텍스트를 함께 전달합니다.
    try:
        vision_response = vision_model.invoke(
            [
                image_data, 
                vision_prompt
            ]
        ).content
        
    except Exception as e:
        # Vision 모델 호출 실패 시, 파이프라인 중단 로그 반환
        return f"Vision 모델 분석 오류: {e}", ""


    # 4-2. Pipeline Step 2: Controller 판단 및 LLM 선택
    llm, system_prompt = determine_routing(target_tag)
    
    # 최종 프롬프트 구성
    final_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "이전에 분석된 데이터와 사용자의 최종 요청을 처리하십시오.\n\n[분석된 데이터]:\n{analysis_result}\n\n[최종 요청]:\n{final_request}")
    ])

    # 4-3. Pipeline Step 3: 최종 LLM 실행 (Controller 지시에 따름)
    final_chain = LLMChain(llm=llm, prompt=final_prompt_template)
    
    final_result = final_chain.invoke({
        "analysis_result": vision_response,
        "final_request": user_request
    })

    process_log = f"Controller가 '{target_tag}' 태그에 따라 {llm.model} 모델을 선택하여 최종 실행했습니다."
    
    return final_result['text'], process_log


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
    if not file.content_type.startswith('image/'):
        # 현재 코드에서는 이미지 파일만 테스트 가능하며,
        # PDF, DOCX 파일 처리는 천우성 팀원이 별도로 OCR/파싱 모듈 구현 필요
        raise HTTPException(status_code=400, detail="현재는 이미지 파일(png, jpg)만 지원합니다. PDF/DOCX 지원을 위해 OCR 모듈이 필요합니다.")

    # 파일 내용을 메모리에 로드하여 이미지 객체 생성
    try:
        contents = await file.read()
        image_data = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일 처리 중 오류가 발생했습니다.")
    
    # (실제 프로젝트에서는 OCR을 통해 텍스트 데이터도 여기서 추출되어야 함)
    # 현재는 OCR 모듈이 없으므로 임시 텍스트 데이터를 사용합니다.
    dummy_text_data = f"업로드된 파일명: {file.filename}. (임시 텍스트) 이 텍스트는 OCR을 통해 추출된 문서의 본문입니다. 이미지는 그래프, 표, 차트 등을 포함합니다."
    
    
    # 5-2. Pipeline 실행
    try:
        final_result, process_log = run_multimodal_pipeline(
            image_data=image_data,
            text_data=dummy_text_data,
            target_tag=target_tag,
            user_request=user_request
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP 파이프라인 실행 중 심각한 오류 발생: {e}")


    # 5-3. 결과 반환
    return JSONResponse(content={
        "status": "success",
        "final_report": final_result,
        "process_log": process_log,
        "model_used": process_log.split(" ")[-3] # 사용된 최종 모델 이름 추출
    })
