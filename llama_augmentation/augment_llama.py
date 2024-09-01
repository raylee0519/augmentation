# 참조할 수 있는 데이터 추가 후 데이터 증강
# 분산작업은 acceletor로 자동 진행
import json
import random
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

# JSONL 파일 읽기 및 처리
input_file = 'nikluge-gtps-2023-train.jsonl'
output_file = 'output_data.jsonl'

def generate_description(table, title, table_title, highlighted_cells):
    highlighted_cells = set(tuple(cell) for cell in highlighted_cells)
    valid_cells = [cell for cell in table if (cell["row"], cell["col"]) not in highlighted_cells]
    selected_cells = random.sample(valid_cells, min(len(valid_cells), 4))  # 선택할 수 있는 셀이 4개보다 적을 수 있음
    
    descriptions = []
    for cell in selected_cells:
        input_text = (
            "아래는 참고할 수 있는 입력 데이터 형태입니다.\n\n"
            '참조 데이터 1: {"id": "nikluge-gtps-2023-train-000001", "input": {"metadata": {"title": "4차 산업혁명에 따른 조세환경 변화와 정책 과제", "table_title": "주요 토픽별 핵심 키워드", "date": "2020-06-09", "publisher": "국회예산정책처", "url": "https://www.nabo.go.kr/Sub/01Report/01_01_Board.jsp", "highlighted_cells": [[0, 0], [1, 0]]}, "table": [{"value": "4차 산업혁명 경제패러다임 변화", "is_header": true, "col": 0, "colspan": 1, "row": 0, "rowspan": 1}, {"value": "디지털경제, 공유경제, 가상화폐, 플랫폼, 유튜브, 무체물(무형자산), 빅데이터", "is_header": true, "col": 1, "colspan": 1, "row": 0, "rowspan": 1}, {"value": "조세환경 변화", "is_header": false, "col": 0, "colspan": 1, "row": 1, "rowspan": 1}, {"value": "과세권, 법인(법인세), 고정사업장, 국제화, 다국적, 디지털세, 소득이전, BEPS, 소득파악, 조세조약, 조세회피(탈세)", "is_header": false, "col": 1, "colspan": 1, "row": 1, "rowspan": 1}, {"value": "기업환경 변화", "is_header": false, "col": 0, "colspan": 1, "row": 2, "rowspan": 1}, {"value": "해외(현지진출), 연구개발, 조세지원, 경쟁, 부과, 게임, 사용(사용자)", "is_header": false, "col": 1, "colspan": 1, "row": 2, "rowspan": 1}]}, "output": ["4차 산업혁명에 따른 경제 패러다임 변화에 대한 키워드로는 디지털경제, 공유경제, 가상화폐, 플랫폼, 유튜브, 무체물(무형자산), 빅데이터를 꼽을 수 있다.", "디지털경제, 공유경제, 가상화폐, 플랫폼, 유튜브, 무체물(무형자산), 빅데이터와 같은 키워드를 가지고 4차 산업혁명에 따른 경제 패러다임의 변화는 설명된다.", "디지털경제, 공유경제, 가상화폐, 플랫폼, 유튜브, 무체물(무형자산), 빅데이터 등이 4차 산업혁명에 따른 경제 패러다임 변화에 대한 키워드로 꼽힌다.", "키워드 중 4차 산업혁명에 따른 경제 패러다임 변화를 설명할 수 있는 것은 디지털경제, 공유경제, 가상화폐, 플랫폼, 유튜브, 무체물(무형자산), 빅데이터 등이다.", "디지털경제, 공유경제, 가상화폐, 플랫폼, 유튜브, 무체물(무형자산), 빅데이터라는 키워드로 4차 산업혁명이 가져온 경제 패러다임의 변화가 이루어졌다."]}\n\n'
            '참조 데이터 2: {"id": "nikluge-gtps-2023-train-000011", "input": {"metadata": {"title": "2019회계연도 결산 위원회별 분석-행정안전부", "table_title": "2019회계연도 행정안전부 소관 총수입 결산", "date": "2020-08-07", "publisher": "국회예산정책처", "url": "https://www.nabo.go.kr/Sub/01Report/01_01_Board.jsp", "highlighted_cells": [[4, 3], [6, 3]]}, "table": [{"value": "구분", "is_header": true, "col": 0, "colspan": 1, "row": 0, "rowspan": 3}, {"value": "2018결산(A)", "is_header": true, "col": 1, "colspan": 1, "row": 0, "rowspan": 3}, {"value": "2019", "is_header": true, "col": 2, "colspan": 4, "row": 0, "rowspan": 1}, {"value": "전년대비(C-A)", "is_header": true, "col": 6, "colspan": 1, "row": 0, "rowspan": 3}, {"value": "예산", "is_header": false, "col": 2, "colspan": 2, "row": 1, "rowspan": 1}, {"value": "결산(C)", "is_header": false, "col": 4, "colspan": 1, "row": 1, "rowspan": 2}, {"value": "예산대비(C-B)", "is_header": false, "col": 5, "colspan": 1, "row": 1, "rowspan": 2}, {"value": "본예산", "is_header": false, "col": 2, "colspan": 1, "row": 2, "rowspan": 1}, {"value": "추경(B)", "is_header": false, "col": 3, "colspan": 1, "row": 2, "rowspan": 1}, {"value": "총수입", "is_header": false, "col": 0, "colspan": 1, "row": 3, "rowspan": 1}, {"value": "48,999", "is_header": false, "col": 1, "colspan": 1, "row": 3, "rowspan": 1}, {"value": "58,877", "is_header": false, "col": 2, "colspan": 1, "row": 3, "rowspan": 1}, {"value": "58,877", "is_header": false, "col": 3, "colspan": 1, "row": 3, "rowspan": 1}, {"value": "61,152", "is_header": false, "col": 4, "colspan": 1, "row": 3, "rowspan": 1}, {"value": "2,275", "is_header": false, "col": 5, "colspan": 1, "row": 3, "rowspan": 1}, {"value": "12,153", "is_header": false, "col": 6, "colspan": 1, "row": 3, "rowspan": 1}]}, "output": ["2019회계연도 행정안전부 소관 총수입 결산은 전년 대비 121억 5,300만 원이 증가한 611억 5,200만 원으로 나타났다.", "전년도보다 121억 5,300만 원이 더해진 2019회계연도 행정안전부 소관 총수입은 611억 5,200만 원으로 집계되었다.", "2019회계연도 행정안전부 소관 총수입 결산은 611억 5,200만 원을 기록하여 전년 대비 121억 5,300만 원 증가했다.", "2019년 회계연도 행정안전부 소관 총수입 결산은 2018년 대비 121억 5,300만 원 증가했으며, 611억 5,200만 원이다.", "611억 5,200만 원으로 나타난 2019회계연도 행정안전부 소관 총수입 결산은 2020년과 비교하여 121억 5,300만 원이 증가한 금액이다."]}\n\n'
            f"제목: {title}\n"
            f"표 제목: {table_title}\n"
            f"하이라이트된 셀: {highlighted_cells}\n"
            f"셀 내용: {cell['value']}\n"
            "이 정보를 기반으로 참조 데이터와 같은 형식으로 출력을 생성해 보세요."
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=50, num_beams=5, early_stopping=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        descriptions.append(generated_text)
    
    return descriptions

# 결과를 저장할 파일 열기
with open(output_file, 'w', encoding='utf-8') as output_f:
    # 입력 JSONL 파일 열기
    with open(input_file, 'r', encoding='utf-8') as input_f:
        for line in input_f:
            data = json.loads(line)
            table = data["input"]["table"]
            title = data["input"]["metadata"]["title"]
            table_title = data["input"]["metadata"]["table_title"]
            highlighted_cells = data["input"]["metadata"]["highlighted_cells"]
            
            # 설명 생성
            descriptions = generate_description(table, title, table_title, highlighted_cells)
            
            # 결과를 JSON 데이터에 추가
            data["output"] = descriptions
            
            # 결과를 파일에 기록
            output_f.write(json.dumps(data, ensure_ascii=False) + '\n')

print("처리가 완료되었습니다.")
