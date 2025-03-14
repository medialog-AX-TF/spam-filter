import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# scikit-learn 패키지 임포트 시도
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn 패키지를 찾을 수 없습니다. 모델 학습 및 예측 기능을 사용할 수 없습니다.")
    SKLEARN_AVAILABLE = False

class SpamClassifier:
    """SMS 스팸 분류 모델 클래스"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        SpamClassifier 초기화
        
        Args:
            model_path: 저장된 모델 경로 (None이면 새 모델 생성)
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 모델 기능을 사용할 수 없습니다.")
            return
            
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_pipeline(self, max_features: int = 5000, 
                       n_estimators: int = 100, 
                       min_samples_split: int = 5) -> None:
        """
        모델 파이프라인 생성
        
        Args:
            max_features: TF-IDF 벡터라이저의 최대 특성 수
            n_estimators: 랜덤 포레스트의 트리 수
            min_samples_split: 랜덤 포레스트의 분할을 위한 최소 샘플 수
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 파이프라인을 생성할 수 없습니다.")
            return
            
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # 유니그램과 바이그램 사용
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words=None,  # 한국어에 맞는 불용어 처리 필요
            min_df=2
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        print("모델 파이프라인이 생성되었습니다.")
    
    def train(self, X_train: List[str], y_train: List[int], 
             X_val: Optional[List[str]] = None, 
             y_val: Optional[List[int]] = None,
             perform_grid_search: bool = False) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            X_train: 학습 텍스트 데이터
            y_train: 학습 레이블 (0: 정상, 1: 스팸)
            X_val: 검증 텍스트 데이터
            y_val: 검증 레이블
            perform_grid_search: 그리드 서치 수행 여부
            
        Returns:
            학습 결과 메트릭
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 모델을 학습할 수 없습니다.")
            return {}
            
        if self.pipeline is None:
            self.create_pipeline()
        
        if perform_grid_search:
            print("그리드 서치를 통한 하이퍼파라미터 최적화를 수행합니다...")
            param_grid = {
                'vectorizer__max_features': [3000, 5000, 10000],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='f1',
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
        else:
            print("모델 학습을 시작합니다...")
            self.pipeline.fit(X_train, y_train)
        
        self.is_trained = True
        print("모델 학습이 완료되었습니다.")
        
        # 학습 결과 평가
        train_predictions = self.pipeline.predict(X_train)
        train_metrics = {
            'accuracy': (train_predictions == y_train).mean(),
            'classification_report': classification_report(y_train, train_predictions, output_dict=True)
        }
        
        print(f"학습 데이터 정확도: {train_metrics['accuracy']:.4f}")
        
        # 검증 데이터가 제공된 경우 검증 결과 평가
        if X_val is not None and y_val is not None:
            val_predictions = self.pipeline.predict(X_val)
            val_metrics = {
                'accuracy': (val_predictions == y_val).mean(),
                'classification_report': classification_report(y_val, val_predictions, output_dict=True)
            }
            
            print(f"검증 데이터 정확도: {val_metrics['accuracy']:.4f}")
            train_metrics['validation'] = val_metrics
        
        return train_metrics
    
    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        텍스트 데이터에 대한 스팸 예측 수행
        
        Args:
            texts: 예측할 텍스트 또는 텍스트 리스트
            
        Returns:
            예측 레이블 (0: 정상, 1: 스팸)
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 예측을 수행할 수 없습니다.")
            return np.array([])
            
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        텍스트 데이터에 대한 스팸 확률 예측 수행
        
        Args:
            texts: 예측할 텍스트 또는 텍스트 리스트
            
        Returns:
            예측 확률 배열 (각 클래스에 대한 확률)
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 예측을 수행할 수 없습니다.")
            return np.array([])
            
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        return self.pipeline.predict_proba(texts)
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                         label_column: Optional[str] = None,
                         prediction_column: str = 'is_spam',
                         probability_column: str = 'spam_probability') -> pd.DataFrame:
        """
        데이터프레임에 대한 스팸 예측 수행
        
        Args:
            df: 처리할 데이터프레임
            text_column: 텍스트가 포함된 컬럼 이름
            label_column: 실제 레이블이 포함된 컬럼 이름 (None이면 평가 생략)
            prediction_column: 예측 레이블을 저장할 컬럼 이름
            probability_column: 예측 확률을 저장할 컬럼 이름
            
        Returns:
            예측 결과가 추가된 데이터프레임
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 예측을 수행할 수 없습니다.")
            return df
            
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        if text_column not in df.columns:
            raise ValueError(f"컬럼 '{text_column}'이 데이터프레임에 존재하지 않습니다.")
        
        # 예측 수행
        texts = df[text_column].fillna("").tolist()
        
        print(f"{len(texts)}개의 메시지에 대한 스팸 예측을 수행합니다...")
        
        # 배치 처리로 메모리 효율성 향상
        batch_size = 1000
        predictions = []
        probabilities = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_preds = self.predict(batch_texts)
            batch_probs = self.predict_proba(batch_texts)[:, 1]  # 스팸 클래스(1)의 확률만 추출
            
            predictions.extend(batch_preds)
            probabilities.extend(batch_probs)
        
        # 결과를 데이터프레임에 추가
        result_df = df.copy()
        result_df[prediction_column] = predictions
        result_df[probability_column] = probabilities
        
        # 실제 레이블이 제공된 경우 평가 수행
        if label_column and label_column in df.columns:
            accuracy = (predictions == df[label_column]).mean()
            print(f"예측 정확도: {accuracy:.4f}")
            
            # 분류 보고서 출력
            print("\n분류 보고서:")
            print(classification_report(df[label_column], predictions))
        
        return result_df
    
    def save_model(self, model_path: str) -> None:
        """
        학습된 모델 저장
        
        Args:
            model_path: 모델을 저장할 경로
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 모델을 저장할 수 없습니다.")
            return
            
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(self.pipeline, model_path)
        print(f"모델이 '{model_path}'에 저장되었습니다.")
    
    def load_model(self, model_path: str) -> None:
        """
        저장된 모델 로드
        
        Args:
            model_path: 로드할 모델 경로
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 모델을 로드할 수 없습니다.")
            return
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일 '{model_path}'을 찾을 수 없습니다.")
        
        self.pipeline = joblib.load(model_path)
        self.vectorizer = self.pipeline.named_steps['vectorizer']
        self.classifier = self.pipeline.named_steps['classifier']
        self.is_trained = True
        
        print(f"모델이 '{model_path}'에서 로드되었습니다.")
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        특성 중요도 시각화
        
        Args:
            top_n: 표시할 상위 특성 수
            figsize: 그림 크기
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 특성 중요도를 시각화할 수 없습니다.")
            return
            
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        # 특성 이름 가져오기
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 특성 중요도 가져오기
        importances = self.classifier.feature_importances_
        
        # 중요도에 따라 정렬
        indices = np.argsort(importances)[::-1][:top_n]
        
        # 시각화
        plt.figure(figsize=figsize)
        plt.title('상위 특성 중요도')
        plt.bar(range(top_n), importances[indices], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                             figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        혼동 행렬 시각화
        
        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            figsize: 그림 크기
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn 패키지가 설치되지 않아 혼동 행렬을 시각화할 수 없습니다.")
            return
            
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['정상', '스팸'], 
                   yticklabels=['정상', '스팸'])
        plt.xlabel('예측')
        plt.ylabel('실제')
        plt.title('혼동 행렬')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 테스트 코드
    if not SKLEARN_AVAILABLE:
        print("scikit-learn 패키지가 설치되지 않아 테스트를 수행할 수 없습니다.")
    else:
        texts = [
            "안녕하세요, 오늘 회의 시간 알려드립니다.",
            "무료 상품권 100만원 당첨! 지금 바로 확인하세요 http://scam.com",
            "내일 점심 같이 먹을래요?",
            "비용 없이 즉시 대출 가능합니다. 지금 전화주세요 010-1234-5678"
        ]
        labels = [0, 1, 0, 1]  # 0: 정상, 1: 스팸
        
        classifier = SpamClassifier()
        classifier.create_pipeline()
        metrics = classifier.train(texts, labels)
        
        print("\n테스트 예측:")
        test_text = "오늘 저녁에 할인 이벤트 진행합니다."
        prediction = classifier.predict([test_text])[0]
        probability = classifier.predict_proba([test_text])[0, 1]
        
        print(f"텍스트: {test_text}")
        print(f"예측: {'스팸' if prediction == 1 else '정상'}")
        print(f"스팸 확률: {probability:.4f}") 