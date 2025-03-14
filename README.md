# 기업 메시징 중계 서버 스팸 필터 시스템

## 개요
본 프로젝트는 기업 사용자와 이통3사(SKT, KT, LG U+) 간의 SMS 및 RCS 메시징 서비스에서 스팸 메시지를 효과적으로 필터링하기 위한 시스템입니다. 메시징 중계 서버에 스팸 필터 기능을 구축하여 불법 광고, 피싱, 스미싱 등의 유해 메시지로부터 최종 사용자를 보호합니다.

## 주요 기능
- SMS 및 RCS 메시지 실시간 스팸 필터링
- 머신러닝 기반 스팸 탐지 알고리즘
- AWS Bedrock Claude 3.7 Sonnet LLM을 활용한 고급 스팸 판정
- 키워드 및 패턴 기반 필터링
- 발신번호 블랙리스트 관리
- 스팸 탐지 로그 및 통계 대시보드
- 관리자 설정 인터페이스
- API 연동 지원

## 시스템 구성
- 스팸 필터 엔진
- 관리자 웹 인터페이스
- 데이터베이스
- API 서버
- 로깅 및 모니터링 시스템
- AWS Bedrock 연동 모듈

## 설치 및 설정

### 요구 사항
- Python 3.8 이상
- 필요한 패키지: requirements.txt 참조
- AWS 계정 및 Bedrock 서비스 접근 권한 (Claude 3.7 Sonnet 사용 시)

### 설치 방법
1. 저장소 클론
```bash
git clone https://github.com/medialog-AX-TF/spam-filter.git
cd spam-filter
```

2. 가상 환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

4. AWS 자격 증명 설정 (Claude 3.7 Sonnet 사용 시)
```bash
# aws_config.py 파일 생성 및 자격 증명 설정
# 예시:
# AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID"
# AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY"
# AWS_REGION = "us-east-1"
```

## 사용 방법

### 데이터 분석
엑셀 파일에 저장된 SMS 메시지 데이터를 분석합니다.

```bash
python src/main.py --mode analyze --data_path 데이터경로 --text_column 메시지컬럼명 --output_dir 결과저장경로
```

주요 옵션:
- `--data_path`: 분석할 엑셀 파일 또는 디렉토리 경로
- `--text_column`: SMS 메시지가 포함된 컬럼 이름
- `--label_column`: 스팸 레이블이 포함된 컬럼 이름 (있는 경우)
- `--output_dir`: 분석 결과를 저장할 디렉토리 경로

### 모델 학습
SMS 스팸 분류 모델을 학습합니다.

```bash
python src/main.py --mode train --data_path 데이터경로 --text_column 메시지컬럼명 --label_column 스팸여부컬럼명 --model_path 모델저장경로
```

주요 옵션:
- `--data_path`: 학습 데이터가 포함된 엑셀 파일 또는 디렉토리 경로
- `--text_column`: SMS 메시지가 포함된 컬럼 이름
- `--label_column`: 스팸 레이블이 포함된 컬럼 이름 (0: 정상, 1: 스팸)
- `--model_path`: 학습된 모델을 저장할 경로
- `--test_size`: 테스트 세트 비율 (기본값: 0.2)
- `--val_size`: 검증 세트 비율 (기본값: 0.1)
- `--grid_search`: 그리드 서치를 통한 하이퍼파라미터 최적화 수행 여부
- `--remove_urls`: URL 제거 여부
- `--remove_numbers`: 숫자 제거 여부
- `--use_konlpy`: KoNLPy 사용 여부 (한국어 형태소 분석)

### 스팸 예측
학습된 모델을 사용하여 SMS 메시지의 스팸 여부를 예측합니다.

```bash
python src/main.py --mode predict --data_path 데이터경로 --model_path 모델경로 --text_column 메시지컬럼명 --output_dir 결과저장경로
```

주요 옵션:
- `--data_path`: 예측할 SMS 메시지가 포함된 엑셀 파일 또는 디렉토리 경로
- `--model_path`: 학습된 모델 파일 경로
- `--text_column`: SMS 메시지가 포함된 컬럼 이름
- `--output_dir`: 예측 결과를 저장할 디렉토리 경로

### 모델 평가
학습된 모델의 성능을 평가합니다.

```bash
python src/main.py --mode evaluate --data_path 데이터경로 --model_path 모델경로 --text_column 메시지컬럼명 --label_column 스팸여부컬럼명 --output_dir 결과저장경로
```

주요 옵션:
- `--data_path`: 평가 데이터가 포함된 엑셀 파일 또는 디렉토리 경로
- `--model_path`: 평가할 모델 파일 경로
- `--text_column`: SMS 메시지가 포함된 컬럼 이름
- `--label_column`: 실제 스팸 레이블이 포함된 컬럼 이름
- `--output_dir`: 평가 결과를 저장할 디렉토리 경로

### AWS Bedrock Claude 3.7 Sonnet을 사용한 스팸 분석
AWS Bedrock의 Claude 3.7 Sonnet LLM을 사용하여 스팸 메시지를 분석하고 판정합니다.

```bash
python analyze_spam_data.py [옵션]
```

주요 옵션:
- `--analyze_only`: 데이터 분석만 수행 (Claude 판정 없음)
- `--claude_sample_size`: Claude로 판정할 샘플 크기 (기본값: 10)
- `--input_file`: 입력 엑셀 파일 경로 (기본값: 'data/스팸리스트_20250310_개인정보_삭제.xlsx')
- `--process_all`: 전체 데이터셋에 Claude 판정 결과 추가

사용 예시:
```bash
# 데이터 분석만 수행
python analyze_spam_data.py --analyze_only

# 20개 샘플 메시지에 대해 Claude 판정 수행
python analyze_spam_data.py --claude_sample_size 20

# 전체 데이터셋에 Claude 판정 결과 추가
python analyze_spam_data.py --process_all --input_file data/스팸리스트_20250310_개인정보_삭제.xlsx

# 다른 입력 파일 사용
python analyze_spam_data.py --input_file data/my_spam_data.xlsx
```

### 추가 기능
- `--save_prompt`: 프롬프트 히스토리 저장 여부
- `--prompt_file`: 프롬프트 히스토리 파일 경로
- `--repo_dir`: 깃허브 저장소 디렉토리 경로 (자동 푸시 기능)
- `--commit_message`: 커밋 메시지

## 예제

### 데이터 분석 예제
```bash
python src/main.py --mode analyze --data_path data/sms_messages.xlsx --text_column message --output_dir results/analysis
```

### 모델 학습 예제
```bash
python src/main.py --mode train --data_path data/labeled_sms.xlsx --text_column message --label_column is_spam --model_path models/spam_classifier.joblib --remove_urls --use_konlpy
```

### 스팸 예측 예제
```bash
python src/main.py --mode predict --data_path data/new_messages.xlsx --model_path models/spam_classifier.joblib --text_column message --output_dir results/predictions
```

### 모델 평가 예제
```bash
python src/main.py --mode evaluate --data_path data/test_messages.xlsx --model_path models/spam_classifier.joblib --text_column message --label_column is_spam --output_dir results/evaluation
```

### Claude 3.7 Sonnet 스팸 판정 예제
```bash
# 샘플 데이터 Claude 판정
python analyze_spam_data.py --claude_sample_size 10

# 전체 데이터셋 Claude 판정
python analyze_spam_data.py --process_all --input_file data/스팸리스트_20250310_개인정보_삭제.xlsx
```

## 결과 해석

### 예측 결과
예측 결과는 다음 컬럼을 포함하는 엑셀 파일로 저장됩니다:
- 원본 메시지 컬럼
- `is_spam`: 스팸 여부 (0: 정상, 1: 스팸)
- `spam_probability`: 스팸일 확률 (0.0 ~ 1.0)
- 기타 추출된 특성 컬럼 (URL 포함 여부, 전화번호 포함 여부 등)

### 평가 결과
평가 결과는 다음 파일들을 포함합니다:
- 혼동 행렬 (Confusion Matrix) 시각화
- ROC 곡선 시각화
- 정밀도-재현율 곡선 시각화
- 임계값에 따른 성능 시각화
- 오분류 샘플 목록
- 평가 메트릭 (정확도, 정밀도, 재현율, F1 점수 등)

### Claude 3.7 Sonnet 판정 결과
Claude 판정 결과는 다음 정보를 포함합니다:
- `Claude_판정`: Claude의 스팸/비스팸 판정 결과
- `Claude_판정_근거`: 판정 이유 설명
- `판정_시간`: 판정에 소요된 시간
- 판정 결과 시각화 (일치율, 혼동 행렬 등)
- 엑셀 업로드 일시 컬럼에 Claude 판정 결과 및 근거 요약

## AWS Bedrock 설정

### AWS 자격 증명 설정
1. AWS 계정 생성 및 Bedrock 서비스 접근 권한 설정
2. IAM 사용자 생성 및 액세스 키 발급
3. `aws_config.py` 파일 생성 및 자격 증명 정보 입력:
```python
# AWS 자격 증명 설정
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY"
AWS_REGION = "us-east-1"  # 사용하는 리전으로 변경

# Claude 모델 설정
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
MAX_TOKENS = 1000
TEMPERATURE = 0.1
```

## 라이센스
본 프로젝트는 [라이센스 정보]에 따라 배포됩니다.

## 연락처
문의사항이 있으시면 [연락처 정보]로 연락주시기 바랍니다. 