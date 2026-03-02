# Tokenizer Input Set

`data/prompts.txt`는 토크나이저 입력용 샘플 문장/질문 모음입니다.

## 형식

- 각 줄: `<TYPE>|<TEXT>`
- `TYPE`
- `S`: statement(평서문)
- `Q`: question(질문문)

## 목적

- 길이가 다양한 입력 테스트
- 기술/일반/한영 혼합 입력 테스트
- 문장부호/숫자/질문형 문장 토큰화 테스트

이 파일을 그대로 읽어서 tokenizer encode에 넣으면 됩니다.
