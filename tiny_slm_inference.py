import json
import tflite_runtime.interpreter as tflite
import numpy as np
import os

# 모델 파일 경로
model_path = "tiny_slm.tflite"  # 상대 경로

# 현재 작업 디렉토리 확인
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# 파일 존재 확인
if not os.path.exists(model_path):
    print("Error: tiny_slm.tflite not found. Please check the file path.")
    exit()

# 어휘 사전 로드
with open('vocab.json', 'r') as f:
    vocab_data = json.load(f)
    word_to_index = vocab_data['word_to_index']
    index_to_word = vocab_data['index_to_word']
    # index_to_word 딕셔너리의 키를 정수형으로 변환
    index_to_word = {int(k): v for k, v in index_to_word.items() if k.isdigit()}

# 모델 로드
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 초기 컨텍스트 단어 선택
context_word = "Hello"  # 시작 단어, 어휘 사전에 있는 단어로 변경
context_index = word_to_index.get(context_word)

if context_index is None:
    print(f"Error: Context word '{context_word}' not found in vocabulary.")
    exit()

print(f"Initial Context: {context_word}")

# 다음 단어 예측
input_data = np.array([[context_index]], dtype=np.int32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output_data[0])
predicted_word = index_to_word.get(predicted_index) #  정수형 index로 검색

if predicted_word is None:
    print(f"Error: Predicted index '{predicted_index}' not found in vocabulary.")
    exit()

print(f"Predicted Word: {predicted_word}")