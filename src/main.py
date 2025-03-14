import os
import sys
import argparse
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_loader import load_excel_data, load_multiple_excel_files, save_processed_data
from src.preprocessing.text_processor import TextProcessor
from src.models.spam_classifier import SpamClassifier
try:
    from src.evaluation.model_evaluator import ModelEvaluator
    MODEL_EVALUATOR_AVAILABLE = True
except ImportError:
    print("model_evaluator 모듈을 로드할 수 없습니다. 평가 기능을 사용할 수 없습니다.")
    MODEL_EVALUATOR_AVAILABLE = False
from src.utils.utils import setup_logger, analyze_dataset, split_dataset, save_prompt_history, push_to_repository

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='SMS 스팸 필터 시스템')
    
    # 모드 선택
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'evaluate', 'analyze'],
                       help='실행 모드 (train: 모델 학습, predict: 예측 수행, evaluate: 모델 평가, analyze: 데이터 분석)')
    
    # 데이터 관련 인수
    parser.add_argument('--data_path', type=str, help='데이터 파일 또는 디렉토리 경로')
    parser.add_argument('--text_column', type=str, default='message', help='텍스트 컬럼 이름')
    parser.add_argument('--label_column', type=str, default='is_spam', help='레이블 컬럼 이름')
    parser.add_argument('--output_dir', type=str, default='results', help='출력 디렉토리')
    
    # 모델 관련 인수
    parser.add_argument('--model_path', type=str, default='models/spam_classifier.joblib', help='모델 파일 경로')
    parser.add_argument('--test_size', type=float, default=0.2, help='테스트 세트 비율')
    parser.add_argument('--val_size', type=float, default=0.1, help='검증 세트 비율')
    parser.add_argument('--random_state', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--grid_search', action='store_true', help='그리드 서치 수행 여부')
    
    # 전처리 관련 인수
    parser.add_argument('--use_konlpy', action='store_true', help='KoNLPy 사용 여부')
    
    # 깃허브 저장소 관련 인수
    parser.add_argument('--repo_dir', type=str, help='깃허브 저장소 디렉토리 경로')
    parser.add_argument('--commit_message', type=str, default='자동 업데이트', help='커밋 메시지')
    
    # 프롬프트 히스토리 관련 인수
    parser.add_argument('--save_prompt', action='store_true', help='프롬프트 히스토리 저장 여부')
    parser.add_argument('--prompt_file', type=str, default='prompt_history.txt', help='프롬프트 히스토리 파일 경로')
    
    return parser.parse_args()

def train_model(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    """
    모델 학습 수행
    
    Args:
        args: 명령줄 인수
        logger: 로거
        
    Returns:
        학습 결과 메트릭
    """
    logger.info("모델 학습을 시작합니다.")
    
    # 데이터 로드
    if os.path.isdir(args.data_path):
        logger.info(f"디렉토리 '{args.data_path}'에서 데이터를 로드합니다.")
        df = load_multiple_excel_files(args.data_path)
    else:
        logger.info(f"파일 '{args.data_path}'에서 데이터를 로드합니다.")
        df = load_excel_data(args.data_path)
    
    logger.info(f"총 {len(df)} 개의 데이터를 로드했습니다.")
    
    # 텍스트 전처리
    logger.info("텍스트 전처리를 수행합니다.")
    processor = TextProcessor(use_konlpy=args.use_konlpy)
    
    # 전처리된 텍스트 컬럼 추가
    processed_column = f"{args.text_column}_processed"
    df = processor.process_dataframe(df, args.text_column, processed_column)
    
    # 데이터 분할
    logger.info("데이터를 학습/검증/테스트 세트로 분할합니다.")
    train_df, test_df, val_df = split_dataset(
        df, 
        test_size=args.test_size,
        val_size=args.val_size,
        stratify_column=args.label_column,
        random_state=args.random_state
    )
    
    logger.info(f"학습 데이터: {len(train_df)} 개")
    if val_df is not None:
        logger.info(f"검증 데이터: {len(val_df)} 개")
    logger.info(f"테스트 데이터: {len(test_df)} 개")
    
    # 모델 초기화 및 학습
    logger.info("모델을 초기화하고 학습합니다.")
    classifier = SpamClassifier()
    classifier.create_pipeline()
    
    # 학습 데이터 준비
    X_train = train_df[processed_column].tolist()
    y_train = train_df[args.label_column].tolist()
    
    # 검증 데이터 준비 (있는 경우)
    X_val = None
    y_val = None
    if val_df is not None:
        X_val = val_df[processed_column].tolist()
        y_val = val_df[args.label_column].tolist()
    
    # 모델 학습
    metrics = classifier.train(
        X_train, y_train,
        X_val, y_val,
        perform_grid_search=args.grid_search
    )
    
    # 모델 저장
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    classifier.save_model(args.model_path)
    logger.info(f"모델이 '{args.model_path}'에 저장되었습니다.")
    
    # 테스트 데이터로 평가
    logger.info("테스트 데이터로 모델을 평가합니다.")
    X_test = test_df[processed_column].tolist()
    y_test = test_df[args.label_column].tolist()
    
    test_predictions = classifier.predict(X_test)
    test_probabilities = classifier.predict_proba(X_test)[:, 1]
    
    # 평가 수행
    test_metrics = {}
    if MODEL_EVALUATOR_AVAILABLE:
        evaluator = ModelEvaluator(results_dir=args.output_dir)
        test_metrics = evaluator.evaluate_model(y_test, test_predictions, test_probabilities)
        
        logger.info(f"테스트 정확도: {test_metrics['accuracy']:.4f}")
        logger.info(f"테스트 정밀도: {test_metrics['precision']:.4f}")
        logger.info(f"테스트 재현율: {test_metrics['recall']:.4f}")
        logger.info(f"테스트 F1 점수: {test_metrics['f1']:.4f}")
        
        # 결과 시각화 및 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 혼동 행렬 시각화
        cm_path = os.path.join(args.output_dir, f"confusion_matrix_{timestamp}.png")
        evaluator.plot_confusion_matrix(y_test, test_predictions, save_path=cm_path)
        
        # ROC 곡선 시각화
        roc_path = os.path.join(args.output_dir, f"roc_curve_{timestamp}.png")
        evaluator.plot_roc_curve(y_test, test_probabilities, save_path=roc_path)
        
        # 정밀도-재현율 곡선 시각화
        pr_path = os.path.join(args.output_dir, f"pr_curve_{timestamp}.png")
        evaluator.plot_precision_recall_curve(y_test, test_probabilities, save_path=pr_path)
        
        # 임계값에 따른 성능 시각화
        threshold_path = os.path.join(args.output_dir, f"threshold_performance_{timestamp}.png")
        best_threshold = evaluator.plot_threshold_performance(
            y_test, test_probabilities, save_path=threshold_path
        )
        
        logger.info(f"최적 임계값: {best_threshold['threshold']:.2f} (F1 점수: {best_threshold['f1']:.4f})")
    else:
        logger.warning("ModelEvaluator를 사용할 수 없어 상세 평가를 수행할 수 없습니다.")
        # 간단한 정확도 계산
        test_metrics = {'accuracy': (test_predictions == y_test).mean()}
        logger.info(f"테스트 정확도: {test_metrics['accuracy']:.4f}")
    
    # 전체 결과 저장
    results = {
        'train_metrics': metrics,
        'test_metrics': test_metrics
    }
    
    if MODEL_EVALUATOR_AVAILABLE:
        results.update({
            'best_threshold': best_threshold,
            'model_path': args.model_path,
            'result_paths': {
                'confusion_matrix': cm_path,
                'roc_curve': roc_path,
                'pr_curve': pr_path,
                'threshold_performance': threshold_path
            }
        })
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"training_results_{timestamp}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"학습 결과가 '{results_path}'에 저장되었습니다.")
    
    # 처리된 데이터 저장
    processed_data_path = os.path.join(args.output_dir, f"processed_data_{timestamp}.xlsx")
    save_processed_data(df, processed_data_path)
    
    return results

def predict(args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    """
    예측 수행
    
    Args:
        args: 명령줄 인수
        logger: 로거
        
    Returns:
        예측 결과가 추가된 데이터프레임
    """
    logger.info("예측을 시작합니다.")
    
    # 모델 로드
    logger.info(f"모델을 '{args.model_path}'에서 로드합니다.")
    classifier = SpamClassifier(args.model_path)
    
    # 데이터 로드
    if os.path.isdir(args.data_path):
        logger.info(f"디렉토리 '{args.data_path}'에서 데이터를 로드합니다.")
        df = load_multiple_excel_files(args.data_path)
    else:
        logger.info(f"파일 '{args.data_path}'에서 데이터를 로드합니다.")
        df = load_excel_data(args.data_path)
    
    logger.info(f"총 {len(df)} 개의 데이터를 로드했습니다.")
    
    # 텍스트 전처리
    logger.info("텍스트 전처리를 수행합니다.")
    processor = TextProcessor(use_konlpy=args.use_konlpy)
    
    # 전처리된 텍스트 컬럼 추가
    processed_column = f"{args.text_column}_processed"
    df = processor.process_dataframe(df, args.text_column, processed_column)
    
    # 예측 수행
    logger.info("예측을 수행합니다.")
    result_df = classifier.process_dataframe(
        df, 
        processed_column, 
        label_column=args.label_column if args.label_column in df.columns else None
    )
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(args.output_dir, f"prediction_results_{timestamp}.xlsx")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    save_processed_data(result_df, result_path)
    logger.info(f"예측 결과가 '{result_path}'에 저장되었습니다.")
    
    return result_df

def evaluate_model(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    """
    모델 평가 수행
    
    Args:
        args: 명령줄 인수
        logger: 로거
        
    Returns:
        평가 메트릭
    """
    if not MODEL_EVALUATOR_AVAILABLE:
        logger.error("ModelEvaluator를 사용할 수 없어 평가를 수행할 수 없습니다.")
        return {'error': 'ModelEvaluator를 사용할 수 없습니다.'}
    
    logger.info("모델 평가를 시작합니다.")
    
    # 모델 로드
    logger.info(f"모델을 '{args.model_path}'에서 로드합니다.")
    classifier = SpamClassifier(args.model_path)
    
    # 데이터 로드
    if os.path.isdir(args.data_path):
        logger.info(f"디렉토리 '{args.data_path}'에서 데이터를 로드합니다.")
        df = load_multiple_excel_files(args.data_path)
    else:
        logger.info(f"파일 '{args.data_path}'에서 데이터를 로드합니다.")
        df = load_excel_data(args.data_path)
    
    logger.info(f"총 {len(df)} 개의 데이터를 로드했습니다.")
    
    # 필요한 컬럼 확인
    if args.label_column not in df.columns:
        raise ValueError(f"레이블 컬럼 '{args.label_column}'이 데이터프레임에 존재하지 않습니다.")
    
    # 텍스트 전처리
    logger.info("텍스트 전처리를 수행합니다.")
    processor = TextProcessor(use_konlpy=args.use_konlpy)
    
    # 전처리된 텍스트 컬럼 추가
    processed_column = f"{args.text_column}_processed"
    df = processor.process_dataframe(df, args.text_column, processed_column)
    
    # 예측 수행
    logger.info("예측을 수행합니다.")
    result_df = classifier.process_dataframe(
        df, 
        processed_column, 
        label_column=args.label_column
    )
    
    # 평가 수행
    logger.info("평가를 수행합니다.")
    evaluator = ModelEvaluator(results_dir=args.output_dir)
    
    metrics = evaluator.generate_evaluation_report(
        result_df,
        text_column=args.text_column,
        label_column=args.label_column,
        prediction_column='is_spam',
        probability_column='spam_probability'
    )
    
    logger.info(f"정확도: {metrics['accuracy']:.4f}")
    logger.info(f"정밀도: {metrics['precision']:.4f}")
    logger.info(f"재현율: {metrics['recall']:.4f}")
    logger.info(f"F1 점수: {metrics['f1']:.4f}")
    
    return metrics

def analyze_data(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    """
    데이터 분석 수행
    
    Args:
        args: 명령줄 인수
        logger: 로거
        
    Returns:
        분석 결과
    """
    logger.info("데이터 분석을 시작합니다.")
    
    # 데이터 로드
    if os.path.isdir(args.data_path):
        logger.info(f"디렉토리 '{args.data_path}'에서 데이터를 로드합니다.")
        df = load_multiple_excel_files(args.data_path)
    else:
        logger.info(f"파일 '{args.data_path}'에서 데이터를 로드합니다.")
        df = load_excel_data(args.data_path)
    
    logger.info(f"총 {len(df)} 개의 데이터를 로드했습니다.")
    
    # 데이터 분석
    logger.info("데이터 분석을 수행합니다.")
    stats = analyze_dataset(
        df,
        text_column=args.text_column,
        label_column=args.label_column if args.label_column in df.columns else None,
        output_dir=args.output_dir
    )
    
    logger.info("데이터 분석이 완료되었습니다.")
    logger.info(f"총 샘플 수: {stats['total_samples']}")
    logger.info(f"NULL 값 수: {stats['null_count']}")
    logger.info(f"빈 문자열 수: {stats['empty_count']}")
    logger.info(f"평균 텍스트 길이: {stats['text_length']['mean']:.2f}")
    logger.info(f"평균 단어 수: {stats['word_count']['mean']:.2f}")
    
    if 'label_counts' in stats:
        logger.info("레이블 분포:")
        for label, count in stats['label_counts'].items():
            logger.info(f"  - {label}: {count} ({count/stats['total_samples']*100:.2f}%)")
    
    return stats

def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_arguments()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로거 설정
    log_file = os.path.join(args.output_dir, 'spam_filter.log')
    logger = setup_logger(log_file)
    
    logger.info("SMS 스팸 필터 시스템을 시작합니다.")
    logger.info(f"실행 모드: {args.mode}")
    
    # 프롬프트 및 결과 저장을 위한 변수
    prompt = f"SMS 스팸 필터 시스템 실행 (모드: {args.mode})"
    result = ""
    
    try:
        # 모드에 따라 실행
        if args.mode == 'train':
            if not args.data_path:
                raise ValueError("학습 모드에서는 --data_path 인수가 필요합니다.")
            
            results = train_model(args, logger)
            if 'test_metrics' in results and 'accuracy' in results['test_metrics']:
                result = f"모델 학습 완료. 테스트 정확도: {results['test_metrics']['accuracy']:.4f}"
            else:
                result = "모델 학습 완료."
            
        elif args.mode == 'predict':
            if not args.data_path:
                raise ValueError("예측 모드에서는 --data_path 인수가 필요합니다.")
            if not args.model_path or not os.path.exists(args.model_path):
                raise ValueError(f"유효한 모델 파일이 필요합니다. '{args.model_path}'를 찾을 수 없습니다.")
            
            result_df = predict(args, logger)
            result = f"예측 완료. 총 {len(result_df)} 개의 메시지 처리됨."
            
        elif args.mode == 'evaluate':
            if not args.data_path:
                raise ValueError("평가 모드에서는 --data_path 인수가 필요합니다.")
            if not args.model_path or not os.path.exists(args.model_path):
                raise ValueError(f"유효한 모델 파일이 필요합니다. '{args.model_path}'를 찾을 수 없습니다.")
            
            if not MODEL_EVALUATOR_AVAILABLE:
                logger.error("ModelEvaluator를 사용할 수 없어 평가를 수행할 수 없습니다.")
                result = "ModelEvaluator를 사용할 수 없어 평가를 수행할 수 없습니다."
            else:
                metrics = evaluate_model(args, logger)
                result = f"모델 평가 완료. 정확도: {metrics['accuracy']:.4f}, F1 점수: {metrics['f1']:.4f}"
            
        elif args.mode == 'analyze':
            if not args.data_path:
                raise ValueError("분석 모드에서는 --data_path 인수가 필요합니다.")
            
            stats = analyze_data(args, logger)
            result = f"데이터 분석 완료. 총 {stats['total_samples']} 개의 메시지 분석됨."
        
        logger.info("작업이 성공적으로 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        result = f"오류 발생: {str(e)}"
        raise
    
    finally:
        # 프롬프트 히스토리 저장
        if args.save_prompt:
            save_prompt_history(prompt, result, args.prompt_file)
        
        # 깃허브 저장소 푸시
        if args.repo_dir:
            push_to_repository(args.repo_dir, args.commit_message)

if __name__ == "__main__":
    main() 