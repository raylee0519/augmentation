# llama 모델을 불러오기 위한 베이스 코드에서 탬플릿 + 답변만을 추출
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 ID 정의
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 모델 로드 (여러 GPU에 걸쳐 자동 분산)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",  # 모델의 데이터 타입 자동 설정
    device_map="auto"    # 사용 가능한 GPU에 모델 자동 분산
)

# 테스트 입력 문장
input_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 한국에 대한 관광정보를 제공하는 도우미 입니다. 질문에 대해 한국어로 답변하세요.
<|eot_id|><|start_header_id|>user<|end_header_id|>전통 의상에 대해 알고 싶어요
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

# 입력 문장 토크나이즈
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

# 모델을 사용하여 텍스트 생성
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=3,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# 생성된 텍스트 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 'assistant' 이후의 텍스트 추출
start_marker = "assistant"

start_idx = generated_text.find(start_marker)

if start_idx != -1:
    # 'assistant' 텍스트의 끝부분을 찾는 방법으로, 'user' 또는 텍스트의 끝을 기준으로 할 수 있습니다.
    # 여기는 간단하게 start_marker 이후부터 텍스트의 끝까지 추출합니다.
    start_idx += len(start_marker)
    assistant_response = generated_text[start_idx:].strip()
else:
    assistant_response = "응답을 찾을 수 없습니다."

# 결과 출력
print(assistant_response)
