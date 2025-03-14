import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import os
import json
from datetime import datetime

class ModelEvaluator:
    """스팸 분류 모델 평가 클래스"""
    
    def __init__(self, results_dir: str = 'results'):
        """
        ModelEvaluator 초기화
        
        Args:
            results_dir: 평가 결과를 저장할 디렉토리
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def evaluate_model(self, y_true: List[int], y_pred: List[int], 
                      y_prob: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        모델 성능 평가
        
        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            y_prob: 예측 확률 (ROC 곡선 등에 사용)
            
        Returns:
            평가 메트릭 사전
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # 예측 확률이 제공된 경우 ROC 및 PR 곡선 메트릭 추가
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['roc_auc'] = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            
            metrics['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                             figsize: Tuple[int, int] = (8, 6),
                             save_path: Optional[str] = None) -> None:
        """
        혼동 행렬 시각화
        
        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            figsize: 그림 크기
            save_path: 그림 저장 경로 (None이면 저장하지 않음)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['정상', '스팸'], 
                   yticklabels=['정상', '스팸'])
        plt.xlabel('예측')
        plt.ylabel('실제')
        plt.title('혼동 행렬')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"혼동 행렬 그림이 '{save_path}'에 저장되었습니다.")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: List[int], y_prob: List[float], 
                      figsize: Tuple[int, int] = (8, 6),
                      save_path: Optional[str] = None) -> None:
        """
        ROC 곡선 시각화
        
        Args:
            y_true: 실제 레이블
            y_prob: 예측 확률
            figsize: 그림 크기
            save_path: 그림 저장 경로 (None이면 저장하지 않음)
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC 곡선 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('위양성률 (False Positive Rate)')
        plt.ylabel('진양성률 (True Positive Rate)')
        plt.title('ROC 곡선')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"ROC 곡선 그림이 '{save_path}'에 저장되었습니다.")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: List[int], y_prob: List[float], 
                                   figsize: Tuple[int, int] = (8, 6),
                                   save_path: Optional[str] = None) -> None:
        """
        정밀도-재현율 곡선 시각화
        
        Args:
            y_true: 실제 레이블
            y_prob: 예측 확률
            figsize: 그림 크기
            save_path: 그림 저장 경로 (None이면 저장하지 않음)
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR 곡선 (AUC = {pr_auc:.3f})')
        plt.xlabel('재현율 (Recall)')
        plt.ylabel('정밀도 (Precision)')
        plt.title('정밀도-재현율 곡선')
        plt.legend(loc="lower left")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"정밀도-재현율 곡선 그림이 '{save_path}'에 저장되었습니다.")
        
        plt.show()
    
    def plot_threshold_performance(self, y_true: List[int], y_prob: List[float], 
                                  metric: str = 'f1',
                                  thresholds: Optional[List[float]] = None,
                                  figsize: Tuple[int, int] = (10, 6),
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        임계값에 따른 성능 시각화
        
        Args:
            y_true: 실제 레이블
            y_prob: 예측 확률
            metric: 평가 지표 ('accuracy', 'precision', 'recall', 'f1')
            thresholds: 평가할 임계값 목록 (None이면 자동 생성)
            figsize: 그림 크기
            save_path: 그림 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            최적 임계값 및 성능 정보
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        metrics = {
            'threshold': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for threshold in thresholds:
            y_pred = (np.array(y_prob) >= threshold).astype(int)
            
            metrics['threshold'].append(threshold)
            metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        
        # 최적 임계값 찾기
        best_idx = np.argmax(metrics[metric])
        best_threshold = metrics['threshold'][best_idx]
        best_performance = {
            'threshold': best_threshold,
            'accuracy': metrics['accuracy'][best_idx],
            'precision': metrics['precision'][best_idx],
            'recall': metrics['recall'][best_idx],
            'f1': metrics['f1'][best_idx]
        }
        
        # 시각화
        plt.figure(figsize=figsize)
        plt.plot(metrics['threshold'], metrics['accuracy'], 'o-', label='정확도')
        plt.plot(metrics['threshold'], metrics['precision'], 'o-', label='정밀도')
        plt.plot(metrics['threshold'], metrics['recall'], 'o-', label='재현율')
        plt.plot(metrics['threshold'], metrics['f1'], 'o-', label='F1 점수')
        plt.axvline(x=best_threshold, color='r', linestyle='--', 
                   label=f'최적 임계값: {best_threshold:.2f}')
        
        plt.xlabel('임계값')
        plt.ylabel('성능')
        plt.title('임계값에 따른 모델 성능')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"임계값 성능 그림이 '{save_path}'에 저장되었습니다.")
        
        plt.show()
        
        return best_performance
    
    def save_evaluation_results(self, metrics: Dict[str, Any], 
                               model_name: str = 'spam_classifier',
                               include_timestamp: bool = True) -> str:
        """
        평가 결과 저장
        
        Args:
            metrics: 평가 메트릭 사전
            model_name: 모델 이름
            include_timestamp: 파일 이름에 타임스탬프 포함 여부
            
        Returns:
            저장된 파일 경로
        """
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{model_name}_evaluation_{timestamp}.json"
        else:
            file_name = f"{model_name}_evaluation.json"
        
        file_path = os.path.join(self.results_dir, file_name)
        
        # 넘파이 배열을 리스트로 변환 (JSON 직렬화를 위해)
        metrics_json = self._convert_numpy_to_list(metrics)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)
        
        print(f"평가 결과가 '{file_path}'에 저장되었습니다.")
        return file_path
    
    def _convert_numpy_to_list(self, obj):
        """넘파이 배열을 리스트로 변환 (JSON 직렬화를 위해)"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def generate_evaluation_report(self, df: pd.DataFrame, 
                                  text_column: str,
                                  label_column: str,
                                  prediction_column: str,
                                  probability_column: str,
                                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        종합 평가 보고서 생성
        
        Args:
            df: 평가 데이터프레임
            text_column: 텍스트 컬럼 이름
            label_column: 실제 레이블 컬럼 이름
            prediction_column: 예측 레이블 컬럼 이름
            probability_column: 예측 확률 컬럼 이름
            output_dir: 출력 디렉토리 (None이면 self.results_dir 사용)
            
        Returns:
            평가 메트릭 사전
        """
        if output_dir is None:
            output_dir = self.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 필요한 컬럼이 있는지 확인
        required_columns = [text_column, label_column, prediction_column, probability_column]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"컬럼 '{col}'이 데이터프레임에 존재하지 않습니다.")
        
        # 데이터 추출
        y_true = df[label_column].values
        y_pred = df[prediction_column].values
        y_prob = df[probability_column].values
        
        # 기본 평가 메트릭 계산
        metrics = self.evaluate_model(y_true, y_pred, y_prob)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 혼동 행렬 시각화 및 저장
        cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        self.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        
        # ROC 곡선 시각화 및 저장
        roc_path = os.path.join(output_dir, f"roc_curve_{timestamp}.png")
        self.plot_roc_curve(y_true, y_prob, save_path=roc_path)
        
        # 정밀도-재현율 곡선 시각화 및 저장
        pr_path = os.path.join(output_dir, f"pr_curve_{timestamp}.png")
        self.plot_precision_recall_curve(y_true, y_prob, save_path=pr_path)
        
        # 임계값에 따른 성능 시각화 및 저장
        threshold_path = os.path.join(output_dir, f"threshold_performance_{timestamp}.png")
        best_threshold = self.plot_threshold_performance(
            y_true, y_prob, save_path=threshold_path
        )
        
        # 최적 임계값 정보 추가
        metrics['best_threshold'] = best_threshold
        
        # 오분류 샘플 분석
        misclassified = df[y_true != y_pred].copy()
        misclassified['error_type'] = np.where(
            misclassified[label_column] == 1, 
            '위음성 (False Negative)', 
            '위양성 (False Positive)'
        )
        
        # 오분류 샘플 저장
        misclassified_path = os.path.join(output_dir, f"misclassified_samples_{timestamp}.csv")
        misclassified.to_csv(misclassified_path, index=False, encoding='utf-8-sig')
        
        # 오분류 통계 추가
        metrics['misclassified_stats'] = {
            'total_count': len(misclassified),
            'false_positive_count': sum(misclassified['error_type'] == '위양성 (False Positive)'),
            'false_negative_count': sum(misclassified['error_type'] == '위음성 (False Negative)')
        }
        
        # 평가 결과 저장
        results_path = self.save_evaluation_results(metrics, include_timestamp=True)
        
        # 결과 경로 정보 추가
        metrics['result_paths'] = {
            'confusion_matrix': cm_path,
            'roc_curve': roc_path,
            'pr_curve': pr_path,
            'threshold_performance': threshold_path,
            'misclassified_samples': misclassified_path,
            'evaluation_results': results_path
        }
        
        return metrics

if __name__ == "__main__":
    # 테스트 코드
    import numpy as np
    
    # 가상의 테스트 데이터 생성
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.random(100)
    y_pred = (y_prob > 0.5).astype(int)
    
    # 평가 수행
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(y_true, y_pred, y_prob)
    
    print("평가 메트릭:")
    print(f"정확도: {metrics['accuracy']:.4f}")
    print(f"정밀도: {metrics['precision']:.4f}")
    print(f"재현율: {metrics['recall']:.4f}")
    print(f"F1 점수: {metrics['f1']:.4f}")
    
    # 시각화
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.plot_roc_curve(y_true, y_prob)
    evaluator.plot_precision_recall_curve(y_true, y_prob)
    best_threshold = evaluator.plot_threshold_performance(y_true, y_prob)
    
    print(f"최적 임계값: {best_threshold['threshold']:.2f} (F1 점수: {best_threshold['f1']:.4f})") 