import re
import string
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# konlpy 패키지 임포트 시도
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    print("konlpy 패키지를 찾을 수 없습니다. 기본 토큰화 방식을 사용합니다.")
    KONLPY_AVAILABLE = False

class TextProcessor:
    """텍스트 전처리 클래스"""
    
    def __init__(self, use_konlpy: bool = True):
        """
        TextProcessor 초기화
        
        Args:
            use_konlpy: konlpy 사용 여부 (한국어 형태소 분석)
        """
        self.use_konlpy = use_konlpy and KONLPY_AVAILABLE
        
        if self.use_konlpy:
            try:
                self.okt = Okt()
                print("한국어 형태소 분석기(Okt)가 초기화되었습니다.")
            except Exception as e:
                print(f"Okt 초기화 중 오류 발생: {e}")
                self.use_konlpy = False
                print("기본 토큰화 방식을 사용합니다.")
        else:
            print("기본 토큰화 방식을 사용합니다.")
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리 수행
        
        Args:
            text: 전처리할 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if not isinstance(text, str):
            return ""
        
        # 소문자 변환
        text = text.lower()
        
        # URL 제거
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # 이메일 제거
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # 전화번호 패턴 변환
        text = re.sub(r'\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}', '[PHONE]', text)
        
        # 특수문자 제거 (단, 한글, 영문, 숫자, 공백 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 여러 공백을 하나로 변환
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        텍스트 토큰화
        
        Args:
            text: 토큰화할 텍스트
            
        Returns:
            토큰 리스트
        """
        if not isinstance(text, str):
            return []
        
        # 전처리 수행
        text = self.preprocess_text(text)
        
        if self.use_konlpy:
            # konlpy Okt를 사용한 토큰화
            try:
                tokens = self.okt.morphs(text, stem=True)
                return [token for token in tokens if len(token) > 1]  # 1글자 토큰 제거
            except Exception as e:
                print(f"Okt 토큰화 중 오류 발생: {e}")
                print("기본 토큰화 방식으로 대체합니다.")
                # 오류 발생 시 기본 토큰화로 폴백
                return self._basic_tokenize(text)
        else:
            # 기본 토큰화 (공백 기준)
            return self._basic_tokenize(text)
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """
        기본 토큰화 (공백 기준)
        
        Args:
            text: 토큰화할 텍스트
            
        Returns:
            토큰 리스트
        """
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]  # 1글자 토큰 제거
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                         processed_column: str = 'processed_text') -> pd.DataFrame:
        """
        데이터프레임의 텍스트 컬럼 전처리
        
        Args:
            df: 처리할 데이터프레임
            text_column: 텍스트가 포함된 컬럼 이름
            processed_column: 전처리된 텍스트를 저장할 컬럼 이름
            
        Returns:
            전처리된 텍스트 컬럼이 추가된 데이터프레임
        """
        if text_column not in df.columns:
            raise ValueError(f"컬럼 '{text_column}'이 데이터프레임에 존재하지 않습니다.")
        
        # 텍스트 전처리 적용
        df[processed_column] = df[text_column].fillna("").apply(self.preprocess_text)
        
        return df
    
    def get_tokens_from_dataframe(self, df: pd.DataFrame, text_column: str) -> List[List[str]]:
        """
        데이터프레임의 텍스트 컬럼에서 토큰 추출
        
        Args:
            df: 처리할 데이터프레임
            text_column: 텍스트가 포함된 컬럼 이름
            
        Returns:
            토큰화된 텍스트 리스트
        """
        if text_column not in df.columns:
            raise ValueError(f"컬럼 '{text_column}'이 데이터프레임에 존재하지 않습니다.")
        
        # 토큰화 적용
        tokens_list = df[text_column].fillna("").apply(self.tokenize).tolist()
        
        return tokens_list
    
    def analyze_text_length(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """
        텍스트 길이 분석
        
        Args:
            df: 분석할 데이터프레임
            text_column: 텍스트가 포함된 컬럼 이름
            
        Returns:
            텍스트 길이 통계 정보
        """
        if text_column not in df.columns:
            raise ValueError(f"컬럼 '{text_column}'이 데이터프레임에 존재하지 않습니다.")
        
        # 텍스트 길이 계산
        text_lengths = df[text_column].fillna("").apply(len)
        
        # 통계 정보 계산
        stats = {
            'mean': text_lengths.mean(),
            'median': text_lengths.median(),
            'min': text_lengths.min(),
            'max': text_lengths.max(),
            'std': text_lengths.std(),
            'quantiles': {
                '25%': text_lengths.quantile(0.25),
                '50%': text_lengths.quantile(0.5),
                '75%': text_lengths.quantile(0.75),
                '90%': text_lengths.quantile(0.9),
                '95%': text_lengths.quantile(0.95),
                '99%': text_lengths.quantile(0.99)
            }
        }
        
        return stats
    
    def get_common_tokens(self, tokens_list: List[List[str]], top_n: int = 20) -> List[tuple]:
        """
        가장 빈번한 토큰 추출
        
        Args:
            tokens_list: 토큰화된 텍스트 리스트
            top_n: 반환할 상위 토큰 수
            
        Returns:
            (토큰, 빈도수) 튜플 리스트
        """
        # 모든 토큰을 하나의 리스트로 병합
        all_tokens = [token for tokens in tokens_list for token in tokens]
        
        # 토큰 빈도수 계산
        token_counts = {}
        for token in all_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # 빈도수에 따라 정렬하여 상위 N개 반환
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_tokens[:top_n]

if __name__ == "__main__":
    # 테스트 코드
    texts = [
        "안녕하세요, 오늘 회의 시간 알려드립니다.",
        "무료 상품권 100만원 당첨! 지금 바로 확인하세요 http://scam.com",
        "내일 점심 같이 먹을래요?",
        "비용 없이 즉시 대출 가능합니다. 지금 전화주세요 010-1234-5678"
    ]
    
    processor = TextProcessor(use_konlpy=True)
    
    for text in texts:
        processed = processor.preprocess_text(text)
        tokens = processor.tokenize(processed)
        print(f"원본: {text}")
        print(f"전처리: {processed}")
        print(f"토큰: {tokens}")
        print("-" * 50) 