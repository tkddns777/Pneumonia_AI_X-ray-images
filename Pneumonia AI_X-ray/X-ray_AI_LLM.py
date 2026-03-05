import json
from google import genai

# ======================================
# API KEY 직접 입력 (테스트용)
# ======================================

client = genai.Client(api_key="AIzaSyCFYI-_ZtwgVa3cnD59sImf_A3q3fPLbmA")

# ======================================
# JSON 읽기
# ======================================

JSON_FILE = r"C:\Users\user\OneDrive\바탕 화면\코딩\Pneumonia_AI_X-ray\Pneumonia AI_X-ray\json_analysis\person172_bacteria_828_analysis.json"

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

prediction = data["prediction"]
confidence = data["confidence"]
region = data.get("activation_region", "unknown")
area_ratio = data.get("activation_area_ratio", 0)

# ======================================
# Prompt
# ======================================

prompt = prompt = f"""
당신은 흉부 X-ray 판독을 보조하는 의료 AI 설명 시스템입니다.

다음은 폐렴 진단 AI 모델의 분석 결과입니다.

예측 결과: {prediction}
예측 확률: {confidence:.3f}

Grad-CAM 활성화 영역: {region}
활성화 영역 비율: {area_ratio:.2f}

위 정보를 바탕으로 다음을 한국어로 설명하세요.

1. 해당 영역에서 의심되는 방사선학적 소견
2. 폐렴과 관련될 수 있는 영상학적 특징
3. 간단한 임상적 해석

설명은 의료 보고서 스타일로 3~5문장 정도로 작성하세요.
"""

# ======================================
# Gemini 호출
# ======================================

response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=prompt
)

explanation = response.text

print("\n===== Gemini Explanation =====\n")
print(explanation)