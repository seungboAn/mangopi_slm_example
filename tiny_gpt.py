from transformers import pipeline

# 모델 및 토크나이저 로드
generator = pipeline('text-generation', model='distilgpt2')

# 초기 프롬프트 설정
prompt = "Hello, how are you?"

# 대화 루프
while True:
    # 사용자 입력 받기
    user_input = input("User: ")

    # 프롬프트 업데이트
    prompt = prompt + " " + user_input

    # 텍스트 생성
    generated_text = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)[0]['generated_text']

    # 응답 출력
    print("Model:", generated_text)

    # 다음 턴을 위해 프롬프트 업데이트
    prompt = generated_text