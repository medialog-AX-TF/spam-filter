# LightGBM 기반 SMS 스팸 필터 성능 검증 및 학습 방법

## 1. 성능 요구사항 검증

### 1.1 처리 성능 요구사항
- 초당 처리량: 1,000 TPS (Transaction Per Second) 이상
- 메시지당 처리 시간: 10ms 이내
- 정확도: 90% 이상
- 오탐률: 5% 이하

### 1.2 LightGBM 모델의 성능 특성
- 추론 속도: 평균 3-5ms/메시지
- 배치 처리 시 2,000-3,000 TPS 달성 가능
- 메모리 사용량: 모델 크기 50-100MB
- 정확도: 92-96% (벤치마크 데이터셋 기준)
- 오탐률: 3-5%

### 1.3 성능 달성 방안
1. **모델 최적화**
   - 특성 수 최적화 (100-200개 이내)
   - 트리 깊이 제한 (최대 8-10)
   - 리프 노드 수 제한 (최대 32-64)
   
2. **배치 처리 최적화**
   - 최적 배치 크기: 50-100 메시지
   - 배치 처리로 2,000-3,000 TPS 달성
   - 병렬 처리로 4,000-5,000 TPS까지 확장 가능

3. **시스템 최적화**
   - 모델 서빙 최적화 (TensorRT, ONNX Runtime)
   - CPU/GPU 병렬 처리
   - 메모리 캐싱 전략

## 2. 학습 데이터 준비

### 2.1 데이터 요구사항
- 최소 100,000건의 레이블된 SMS 메시지
- 스팸:정상 비율 = 30:70 (불균형 데이터 처리)
- 한국어/영어 혼합 텍스트 지원
- 최신 스팸 패턴 포함

### 2.2 데이터 수집 방안
1. **공개 데이터셋**
   - UCI SMS Spam Collection
   - 한국정보화진흥원 스팸 데이터셋
   - GitHub 공개 SMS 데이터셋

2. **자체 수집 데이터**
   - 기업 메시징 서비스 로그
   - 사용자 신고 데이터
   - 스팸 트랩 계정 수집 데이터

3. **데이터 증강**
   - 백 트랜슬레이션
   - 동의어 치환
   - 규칙 기반 변형

### 2.3 데이터 전처리
1. **텍스트 정규화**
   ```python
   def normalize_text(text):
       # 소문자 변환
       text = text.lower()
       # 특수문자 처리
       text = re.sub(r'[^\w\s]', ' ', text)
       # 숫자 정규화
       text = re.sub(r'\d+', 'NUM', text)
       # 공백 정규화
       text = ' '.join(text.split())
       return text
   ```

2. **토큰화 및 임베딩**
   - 형태소 분석 (Mecab/KoNLPy)
   - N-gram 특성 추출 (N=1,2,3)
   - TF-IDF 벡터화

3. **특성 엔지니어링**
   - 텍스트 길이
   - URL 개수
   - 특수문자 비율
   - 대문자 비율
   - 이모티콘 사용
   - 연속된 문자 반복

## 3. 모델 학습 방법

### 3.1 LightGBM 하이퍼파라미터
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 32,
    'max_depth': 8,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 8
}
```

### 3.2 학습 파이프라인
```python
def train_spam_filter():
    # 데이터 로드
    X_train, X_test, y_train, y_test = load_data()
    
    # 특성 추출
    vectorizer = TfidfVectorizer(max_features=200)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # 추가 특성 생성
    X_train_features = create_additional_features(X_train)
    
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_train_tfidf, label=y_train)
    
    # 모델 학습
    model = lgb.train(params,
                     train_data,
                     num_boost_round=100,
                     valid_sets=[valid_data],
                     early_stopping_rounds=10)
    
    return model, vectorizer
```

### 3.3 교차 검증 및 성능 평가
```python
def evaluate_model(model, X_test, y_test):
    # 예측 수행
    y_pred = model.predict(X_test)
    
    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred > 0.5)
    precision = precision_score(y_test, y_pred > 0.5)
    recall = recall_score(y_test, y_pred > 0.5)
    f1 = f1_score(y_test, y_pred > 0.5)
    
    # 처리 시간 측정
    start_time = time.time()
    model.predict(X_test[:1000])
    inference_time = (time.time() - start_time) / 1000
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'inference_time': inference_time
    }
```

## 4. 모델 최적화 및 서빙

### 4.1 모델 경량화
1. **특성 선택**
   - Feature Importance 기반 선택
   - L1 정규화
   - 상관관계 분석

2. **모델 압축**
   - 가지치기 (Pruning)
   - 양자화 (Quantization)
   - 모델 증류 (Distillation)

### 4.2 배치 처리 최적화
```python
def batch_predict(model, texts, batch_size=100):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_features = prepare_features(batch)
        batch_predictions = model.predict(batch_features)
        predictions.extend(batch_predictions)
    return predictions
```

### 4.3 서빙 시스템 구성
1. **모델 서빙 아키텍처**
   - 로드 밸런서
   - 모델 서버 풀
   - 캐시 레이어
   - 모니터링 시스템

2. **성능 모니터링**
   - 처리량 (TPS)
   - 지연 시간 (Latency)
   - 자원 사용률
   - 정확도 변화

## 5. 성능 테스트 계획

### 5.1 단위 테스트
- 개별 메시지 처리 시간 측정
- 정확도 및 오탐률 검증
- 메모리 사용량 측정

### 5.2 부하 테스트
- 점진적 부하 증가 (500 → 1000 → 2000 TPS)
- 지속적 최대 부하 (1000 TPS, 1시간)
- 스파이크 부하 (순간 3000 TPS)

### 5.3 안정성 테스트
- 24시간 연속 운영 테스트
- 장애 복구 테스트
- 메모리 누수 테스트

## 6. 결론

LightGBM 모델은 다음과 같은 이유로 SMS 스팸 필터의 1000 TPS 처리 요구사항을 충족할 수 있습니다:

1. **처리 성능**
   - 단일 메시지 처리 시간: 3-5ms
   - 배치 처리 시 2,000-3,000 TPS 달성
   - 병렬 처리로 4,000-5,000 TPS까지 확장 가능

2. **정확도 및 안정성**
   - 92-96%의 높은 정확도
   - 3-5%의 낮은 오탐률
   - 안정적인 처리 성능

3. **리소스 효율성**
   - 50-100MB의 합리적인 모델 크기
   - 효율적인 메모리 사용
   - 낮은 CPU 부하

이러한 성능 지표들은 POC를 통해 실제 검증이 필요하며, 특히 한국어 SMS 특성을 고려한 최적화가 추가로 요구됩니다. 