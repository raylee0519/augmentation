# llama 모델을 불러오기 위한 베이스 코드
from transformers import AutoTokenizer, AutoModelForCausalLM

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
input_text = "Hello, how are you?"

# 입력 문장 토크나이즈
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

# 모델을 사용하여 텍스트 생성
outputs = model.generate(**inputs)

# 생성된 텍스트 디코딩 및 출력
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
