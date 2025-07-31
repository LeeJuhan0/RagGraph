import requests
import os
import pandas as pd
import re
from datasets import Dataset
from huggingface_hub import login
login(token="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")

# URL 및 파일 저장 경로 지정
base_url = "https://eiec.kdi.re.kr/material/wordDicDetail.do"
output_csv = "word_dic_results_cleaned.csv"

start_idx = 1
end_idx = 2959

# 결과를 저장할 리스트 초기화
results = []

# HTML 태그 제거 및 공백 제거 함수
def clean_text(text):
    # HTML 태그 제거
    text = re.sub(r"<.*?>", "", text)
    # 줄바꿈, 탭, 기타 공백 문자 제거
    text = re.sub(r"[\r\n\t]+", " ", text)
    # 앞뒤 공백 제거
    return text.strip()

# SSL 인증서 검증 생략 (verify=False)
for idx in range(start_idx, end_idx + 1):
    try:
        response = requests.get(base_url, params={"dic_idx": idx}, verify=False)  # SSL 인증서 검증 생략
        response.raise_for_status()

        # 응답 데이터 처리 후 리스트에 추가
        cleaned_content = clean_text(response.text)
        results.append({"content": cleaned_content})  # idx 제거
        print(f"[{idx}/{end_idx}] 저장 완료")

    except requests.exceptions.RequestException as e:
        print(f"[{idx}] 요청 실패: {e}")
        results.append({"content": None})  # 실패한 경우 내용은 None

# DataFrame으로 변환
df = pd.DataFrame(results)
print(df)

# CSV로 저장
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"CSV 파일 저장 완료: {os.path.abspath(output_csv)}")

hf_dataset = Dataset.from_pandas(df)

hf_dataset.push_to_hub(
    repo_id="Juhannn/KDI_finanance",  # 데이터셋 이름 (username/dataset_name 형식)
    token="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",  # Hugging Face API 토큰
    private=False,  # True로 설정하면 데이터를 비공개로 업로드
)