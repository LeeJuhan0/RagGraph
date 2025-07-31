from datasets import load_dataset
import pandas as pd

# Hugging Face 데이터셋 로드
# "Juhannn/practice" 데이터셋을 로드합니다.
dataset = load_dataset("Juhannn/practice")

# 데이터셋을 Pandas DataFrame으로 변환
df = pd.DataFrame(dataset['finance_word'])  # 'train'을 사용 (다른 split 사용 가능)

# 'question' 열과 'answer' 열을 합쳐 새로운 'context' 열 생성
df['context'] = df['question'] + " " + df['answer']

# 결과를 CSV 파일로 저장
output_csv = "koreabank.csv"
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"CSV 파일이 저장되었습니다: {output_csv}")