import pandas as pd
import os
from typing import List, Tuple, Optional

def load_excel_data(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    엑셀 파일에서 SMS 데이터를 로드합니다.
    
    Args:
        file_path: 엑셀 파일 경로
        sheet_name: 시트 이름 (기본값: None, 첫 번째 시트 사용)
    
    Returns:
        로드된 데이터프레임
    """
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        print(f"파일 '{file_path}'에서 {len(df)} 개의 SMS 데이터를 로드했습니다.")
        return df
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {str(e)}")
        raise

def load_multiple_excel_files(directory_path: str, file_pattern: str = "*.xlsx") -> pd.DataFrame:
    """
    지정된 디렉토리에서 여러 엑셀 파일을 로드하고 병합합니다.
    
    Args:
        directory_path: 엑셀 파일이 있는 디렉토리 경로
        file_pattern: 파일 패턴 (기본값: "*.xlsx")
    
    Returns:
        병합된 데이터프레임
    """
    import glob
    
    all_files = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not all_files:
        raise ValueError(f"'{directory_path}' 디렉토리에서 '{file_pattern}' 패턴과 일치하는 파일을 찾을 수 없습니다.")
    
    dfs = []
    for file in all_files:
        try:
            df = load_excel_data(file)
            dfs.append(df)
            print(f"파일 '{file}'을 성공적으로 로드했습니다.")
        except Exception as e:
            print(f"파일 '{file}' 로드 중 오류 발생: {str(e)}")
    
    if not dfs:
        raise ValueError("로드할 수 있는 유효한 엑셀 파일이 없습니다.")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"총 {len(combined_df)} 개의 SMS 데이터를 로드했습니다.")
    
    return combined_df

def save_processed_data(df: pd.DataFrame, output_path: str, include_timestamp: bool = True) -> None:
    """
    처리된 데이터를 저장합니다.
    
    Args:
        df: 저장할 데이터프레임
        output_path: 출력 파일 경로
        include_timestamp: 파일 이름에 타임스탬프 포함 여부
    """
    from datetime import datetime
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name, file_ext = os.path.splitext(output_path)
        output_path = f"{file_name}_{timestamp}{file_ext}"
    
    # 파일 확장자에 따라 저장 방식 결정
    if output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    elif output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)  # 기본값으로 CSV 사용
    
    print(f"처리된 데이터를 '{output_path}'에 저장했습니다.")

if __name__ == "__main__":
    # 테스트 코드
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        df = load_excel_data(file_path)
        print(df.head())
    else:
        print("사용법: python data_loader.py <엑셀_파일_경로>") 