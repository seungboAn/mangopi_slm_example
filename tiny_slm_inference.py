import tflite_runtime.interpreter as tflite
import numpy as np
import json

# 어휘 사전 로드
with open('vocab.json', 'r') as f:
    vocab_data = json.load(f)
    word_to_index = vocab_data['word_to_index']
    index_to_word = vocab_data['index_to_word']
    index_to_word = {int(k): v for k, v in index_to_word.items()} # JSON 로드 시 key가 문자열로 변경되므로 int로 변환

# 모델 로드
interpreter = tflite.Interpreter(model_path="tiny_slm.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 초기 컨텍스트 단어 선택
context_word = "today"  # 시작 단어
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
predicted_word = index_to_word.get(str(predicted_index)) # index_to_word의 키가 문자열이므로 str로 변환

if predicted_word is None:
    print(f"Error: Predicted index '{predicted_index}' not found in vocabulary.")
    exit()

print(f"Predicted Word: {predicted_word}")