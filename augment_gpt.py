import openai
import json
import random
from dotenv import load_dotenv
import os

# .env 파일에서 OpenAI API 키 로드
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# jsonl 파일을 읽어오는 함수 (상위 5줄만 가져옴)
def load_jsonl(file_path, limit=5):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f][:limit]

# GPT API를 사용하여 새로운 highlighted_cells와 output을 생성하는 함수
def generate_augmented_data(item):
    input_data = item['input']
    table = input_data['table']
    
    # 모든 셀을 (row, col)로 저장
    all_cells = [(cell['row'], cell['col']) for cell in table]
    
    # 기존 highlighted_cells를 제외한 셀들만 남기기
    highlighted_cells = input_data['metadata']['highlighted_cells']
    remaining_cells = list(set(all_cells) - set(tuple(cell) for cell in highlighted_cells))
    
    # 임의로 4개의 셀 선택
    new_highlighted_cells = random.sample(remaining_cells, 4)
    
    # 새로 선택한 highlighted_cells를 input_data에 업데이트
    input_data['metadata']['highlighted_cells'] = new_highlighted_cells
    input_data_str = json.dumps(input_data, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that processes table data."},
        {"role": "user", "content": f"""
        여기 JSON 데이터가 있습니다. 이 데이터에서 highlighted_cells로 지정된 4개의 셀을 기반으로 
        5줄의 추론 및 설명을 생성해 주세요. 결과는 내가 주었던 JSON 데이터 형식으로만 반환해 주세요.
        
        데이터:
        {input_data_str}
        """}
    ]

    # GPT API 호출
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=200,
        temperature=0.7
    )

    # 응답 처리
    try:
        new_output = json.loads(response.choices[0].message['content'].strip())['output']
        return {
            "id": item['id'] + '-generated',
            "input": {
                "metadata": {
                    "title": input_data['metadata']['title'],
                    "table_title": input_data['metadata']['table_title'],
                    "date": input_data['metadata']['date'],
                    "publisher": input_data['metadata']['publisher'],
                    "url": input_data['metadata']['url'],
                    "highlighted_cells": new_highlighted_cells
                },
                "table": table
            },
            "output": new_output
        }
    except json.JSONDecodeError:
        print("JSON 디코딩 오류 발생. 응답 내용을 확인하세요.")
        return None

# jsonl 데이터를 처리하는 함수
def process_jsonl_data(jsonl_data):
    augmented_data = []
    for item in jsonl_data:
        augmented_item = generate_augmented_data(item)
        if augmented_item:
            augmented_data.append(augmented_item)
    return augmented_data

# 결과를 jsonl 형식으로 저장하는 함수
def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 파일 경로 설정
input_file = 'nikluge-gtps-2023-train.jsonl'
output_file = 'augmented_output.jsonl'

# jsonl 데이터 로드 (상위 5줄만)
jsonl_data = load_jsonl(input_file, limit=5)

# 데이터 처리
augmented_data = process_jsonl_data(jsonl_data)

# 결과 저장
save_jsonl(augmented_data, output_file)

print("초반 5줄의 데이터 증강이 완료되었습니다.")