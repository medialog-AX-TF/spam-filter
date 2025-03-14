import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging
import re

# 로깅 설정
def setup_logger(log_file: str = 'spam_filter.log', level: int = logging.INFO) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        log_file: 로그 파일 경로
        level: 로깅 레벨
        
    Returns:
        설정된 로거
    """
    # 로그 디렉토리 생성
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger('spam_filter')
    logger.setLevel(level)
    
    # 이미 핸들러가 있으면 제거
    if logger.handlers:
        logger.handlers = []
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# 데이터 분석 함수
def analyze_dataset(df: pd.DataFrame, text_column: str, 
                   label_column: Optional[str] = None,
                   output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    데이터셋 분석
    
    Args:
        df: 분석할 데이터프레임
        text_column: 텍스트 컬럼 이름
        label_column: 레이블 컬럼 이름 (None이면 레이블 분석 생략)
        output_dir: 출력 디렉토리 (None이면 저장하지 않음)
        
    Returns:
        분석 결과 사전
    """
    if text_column not in df.columns:
        raise ValueError(f"컬럼 '{text_column}'이 데이터프레임에 존재하지 않습니다.")
    
    if label_column and label_column not in df.columns:
        raise ValueError(f"컬럼 '{label_column}'이 데이터프레임에 존재하지 않습니다.")
    
    # 기본 통계
    stats = {
        'total_samples': len(df),
        'null_count': df[text_column].isnull().sum(),
        'empty_count': (df[text_column] == '').sum(),
        'text_length': {
            'mean': df[text_column].str.len().mean(),
            'std': df[text_column].str.len().std(),
            'min': df[text_column].str.len().min(),
            'max': df[text_column].str.len().max(),
            'median': df[text_column].str.len().median()
        },
        'word_count': {
            'mean': df[text_column].str.split().str.len().mean(),
            'std': df[text_column].str.split().str.len().std(),
            'min': df[text_column].str.split().str.len().min(),
            'max': df[text_column].str.split().str.len().max(),
            'median': df[text_column].str.split().str.len().median()
        }
    }
    
    # URL, 전화번호, 이메일 등 패턴 분석
    df['has_url'] = df[text_column].str.contains(r'https?://\S+|www\.\S+', regex=True).astype(int)
    df['has_phone'] = df[text_column].str.contains(r'\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}', regex=True).astype(int)
    df['has_email'] = df[text_column].str.contains(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', regex=True).astype(int)
    
    stats['pattern_counts'] = {
        'url_count': df['has_url'].sum(),
        'phone_count': df['has_phone'].sum(),
        'email_count': df['has_email'].sum()
    }
    
    # 레이블 분석 (레이블이 제공된 경우)
    if label_column:
        label_counts = df[label_column].value_counts().to_dict()
        stats['label_counts'] = label_counts
        
        # 레이블별 텍스트 길이 분석
        stats['text_length_by_label'] = {}
        for label, group in df.groupby(label_column):
            stats['text_length_by_label'][str(label)] = {
                'mean': group[text_column].str.len().mean(),
                'std': group[text_column].str.len().std(),
                'median': group[text_column].str.len().median()
            }
        
        # 레이블별 패턴 분석
        stats['pattern_by_label'] = {}
        for label, group in df.groupby(label_column):
            stats['pattern_by_label'][str(label)] = {
                'url_ratio': group['has_url'].mean(),
                'phone_ratio': group['has_phone'].mean(),
                'email_ratio': group['has_email'].mean()
            }
    
    # 시각화 및 저장 (출력 디렉토리가 제공된 경우)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 텍스트 길이 분포 시각화
        plt.figure(figsize=(10, 6))
        sns.histplot(df[text_column].str.len(), bins=50)
        plt.title('텍스트 길이 분포')
        plt.xlabel('텍스트 길이')
        plt.ylabel('빈도')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"text_length_dist_{timestamp}.png"))
        plt.close()
        
        # 단어 수 분포 시각화
        plt.figure(figsize=(10, 6))
        sns.histplot(df[text_column].str.split().str.len(), bins=50)
        plt.title('단어 수 분포')
        plt.xlabel('단어 수')
        plt.ylabel('빈도')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"word_count_dist_{timestamp}.png"))
        plt.close()
        
        # 레이블이 제공된 경우 레이블별 시각화
        if label_column:
            # 레이블 분포 시각화
            plt.figure(figsize=(8, 6))
            sns.countplot(x=df[label_column])
            plt.title('레이블 분포')
            plt.xlabel('레이블')
            plt.ylabel('빈도')
            plt.xticks([0, 1], ['정상', '스팸'])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"label_dist_{timestamp}.png"))
            plt.close()
            
            # 레이블별 텍스트 길이 분포 시각화
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[label_column], y=df[text_column].str.len())
            plt.title('레이블별 텍스트 길이 분포')
            plt.xlabel('레이블')
            plt.ylabel('텍스트 길이')
            plt.xticks([0, 1], ['정상', '스팸'])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"text_length_by_label_{timestamp}.png"))
            plt.close()
            
            # 레이블별 패턴 비율 시각화
            pattern_df = pd.DataFrame({
                'URL 포함': df.groupby(label_column)['has_url'].mean(),
                '전화번호 포함': df.groupby(label_column)['has_phone'].mean(),
                '이메일 포함': df.groupby(label_column)['has_email'].mean()
            })
            
            plt.figure(figsize=(10, 6))
            pattern_df.plot(kind='bar')
            plt.title('레이블별 패턴 비율')
            plt.xlabel('레이블')
            plt.ylabel('비율')
            plt.xticks([0, 1], ['정상', '스팸'], rotation=0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pattern_by_label_{timestamp}.png"))
            plt.close()
        
        # 분석 결과 저장
        with open(os.path.join(output_dir, f"dataset_analysis_{timestamp}.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    
    return stats

# 데이터 분할 함수
def split_dataset(df: pd.DataFrame, 
                 test_size: float = 0.2, 
                 val_size: Optional[float] = 0.1,
                 stratify_column: Optional[str] = None,
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    데이터셋 분할
    
    Args:
        df: 분할할 데이터프레임
        test_size: 테스트 세트 비율
        val_size: 검증 세트 비율 (None이면 검증 세트 생성 안 함)
        stratify_column: 계층화 샘플링에 사용할 컬럼 이름
        random_state: 랜덤 시드
        
    Returns:
        (학습 데이터프레임, 테스트 데이터프레임, 검증 데이터프레임)
    """
    from sklearn.model_selection import train_test_split
    
    if stratify_column and stratify_column not in df.columns:
        raise ValueError(f"컬럼 '{stratify_column}'이 데이터프레임에 존재하지 않습니다.")
    
    stratify = df[stratify_column] if stratify_column else None
    
    # 검증 세트가 필요한 경우
    if val_size:
        # 먼저 학습+검증 세트와 테스트 세트로 분할
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, 
            stratify=stratify, 
            random_state=random_state
        )
        
        # 학습+검증 세트에서 계층화 정보 업데이트
        if stratify_column:
            stratify = train_val_df[stratify_column]
        
        # 학습 세트와 검증 세트로 분할
        # val_size를 학습+검증 세트 크기에 대한 비율로 조정
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio,
            stratify=stratify,
            random_state=random_state
        )
        
        return train_df, test_df, val_df
    
    # 검증 세트가 필요 없는 경우
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size,
            stratify=stratify,
            random_state=random_state
        )
        
        return train_df, test_df, None

# 프롬프트 히스토리 저장 함수
def save_prompt_history(prompt: str, result: str, 
                       history_file: str = 'prompt_history.txt') -> None:
    """
    프롬프트 히스토리 저장
    
    Args:
        prompt: 프롬프트 텍스트
        result: 결과 텍스트
        history_file: 히스토리 파일 경로
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(f"===== 프롬프트 ({timestamp}) =====\n")
        f.write(prompt + "\n\n")
        f.write("===== 결과 =====\n")
        f.write(result + "\n\n")
        f.write("=" * 50 + "\n\n")
    
    print(f"프롬프트 히스토리가 '{history_file}'에 저장되었습니다.")

# 깃허브 저장소 푸시 함수
def push_to_repository(repo_dir: str, commit_message: str = "자동 업데이트") -> bool:
    """
    깃허브 저장소에 변경사항 푸시
    
    Args:
        repo_dir: 저장소 디렉토리 경로
        commit_message: 커밋 메시지
        
    Returns:
        성공 여부
    """
    import subprocess
    
    try:
        # 현재 작업 디렉토리 저장
        current_dir = os.getcwd()
        
        # 저장소 디렉토리로 이동
        os.chdir(repo_dir)
        
        # git add
        subprocess.run(['git', 'add', '.'], check=True)
        
        # git commit
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        
        # git push
        subprocess.run(['git', 'push'], check=True)
        
        print(f"저장소 '{repo_dir}'에 변경사항이 푸시되었습니다.")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"저장소 푸시 중 오류 발생: {str(e)}")
        success = False
    except Exception as e:
        print(f"저장소 푸시 중 예외 발생: {str(e)}")
        success = False
    finally:
        # 원래 작업 디렉토리로 복귀
        os.chdir(current_dir)
    
    return success

if __name__ == "__main__":
    # 테스트 코드
    logger = setup_logger()
    logger.info("유틸리티 모듈 테스트")
    
    # 프롬프트 히스토리 저장 테스트
    save_prompt_history(
        "SMS 스팸 필터 시스템을 구축해주세요.",
        "SMS 스팸 필터 시스템을 구축했습니다."
    ) 