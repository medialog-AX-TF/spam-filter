import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import boto3
import json
import datetime
import time
import argparse
import sys

# AWS 설정 파일 가져오기
try:
    import aws_config
    AWS_CREDENTIALS_AVAILABLE = True
except ImportError:
    AWS_CREDENTIALS_AVAILABLE = False
    print("경고: aws_config.py 파일을 찾을 수 없습니다. AWS Bedrock 기능을 사용할 수 없습니다.")
    print("aws_config.py 파일을 생성하고 AWS 자격 증명을 설정하세요.")

# 결과 디렉토리 생성
os.makedirs('results/spam_analysis', exist_ok=True)

# 명령행 인자 파싱
parser = argparse.ArgumentParser(description='스팸 메시지 분석 및 Claude 3.5 Sonnet을 사용한 스팸 판정')
parser.add_argument('--analyze_only', action='store_true', help='데이터 분석만 수행 (Claude 판정 없음)')
parser.add_argument('--claude_sample_size', type=int, default=10, help='Claude로 판정할 샘플 크기')
parser.add_argument('--input_file', type=str, default='data/스팸리스트_20250310_개인정보_삭제.xlsx', help='입력 엑셀 파일 경로')
parser.add_argument('--process_all', action='store_true', help='전체 데이터셋에 Claude 판정 결과 추가')
parser.add_argument('--detailed_reason', action='store_true', help='상세한 판정 근거 요청 (토큰 사용량 증가)')
args = parser.parse_args()

# AWS Bedrock 클라이언트 설정 함수
def get_bedrock_client():
    if not AWS_CREDENTIALS_AVAILABLE:
        return None
    
    try:
        # AWS 설정 파일에서 자격 증명 가져오기
        credentials = aws_config.get_aws_credentials()
        
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=credentials['region_name'],
            aws_access_key_id=credentials['aws_access_key_id'],
            aws_secret_access_key=credentials['aws_secret_access_key']
        )
        return bedrock_runtime
    except Exception as e:
        print(f"AWS Bedrock 클라이언트 생성 오류: {e}")
        return None

# Claude 3.5 Sonnet을 사용하여 스팸 판정 함수
def classify_spam_with_claude(message, client, detailed=False):
    if client is None:
        return "판정 불가", "AWS Bedrock 클라이언트 연결 실패"
    
    try:
        # AWS 설정 파일에서 Claude 설정 가져오기
        claude_settings = aws_config.get_claude_settings()
        
        # 상세 판정 근거 요청 여부에 따라 프롬프트 조정
        if detailed:
            prompt = f"""
            당신은 SMS 메시지 스팸 판정 전문가입니다. 다음 SMS 메시지가 스팸인지 비스팸인지 판단해주세요.
            
            메시지: {message}
            
            다음 기준을 고려하여 판단하세요:
            1. 광고성 메시지인지 여부
            2. URL이나 전화번호 포함 여부
            3. 긴급성이나 과장된 표현 사용 여부
            4. 개인정보 요구 여부
            5. 문법적 오류나 의심스러운 표현 여부
            
            다음 형식으로 응답해주세요:
            판정: [스팸 또는 비스팸]
            판정 근거: [각 기준에 따른 상세한 판정 이유 설명]
            """
        else:
            prompt = f"""
            다음 SMS 메시지가 스팸인지 비스팸인지 판단해주세요.
            메시지: {message}
            
            다음 형식으로 응답해주세요:
            판정: [스팸 또는 비스팸]
            판정 근거: [판정 이유 설명]
            """
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": claude_settings['max_tokens'],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": claude_settings['temperature']
        })
        
        response = client.invoke_model(
            modelId=claude_settings['model_id'],
            body=body
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        result = response_body['content'][0]['text']
        
        # 결과 파싱
        if "판정: 스팸" in result:
            classification = "스팸"
        elif "판정: 비스팸" in result:
            classification = "비스팸"
        else:
            classification = "판정 불확실"
        
        # 판정 근거 추출
        reason_match = re.search(r'판정 근거: (.*?)(?:\n\n|$)', result, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else "판정 근거를 찾을 수 없음"
        
        return classification, reason
    
    except Exception as e:
        print(f"Claude API 호출 오류: {e}")
        return "판정 불가", f"API 오류: {str(e)}"

# 데이터 로드
print(f"데이터 로드 중... ({args.input_file})")
try:
    df = pd.read_excel(args.input_file)
except Exception as e:
    print(f"데이터 로드 오류: {e}")
    print(f"파일이 존재하는지 확인하세요: {args.input_file}")
    sys.exit(1)

# 기본 정보 출력
print(f"데이터셋 크기: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")

# 필수 컬럼 확인
required_columns = ['내용', '휴먼 분류', 'AI 분류']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"경고: 필수 컬럼이 누락되었습니다: {missing_columns}")
    print("분석을 계속하려면 필수 컬럼이 있는 데이터를 사용하세요.")
    sys.exit(1)

# 휴먼 분류 값 분포
print("\n휴먼 분류 값 분포:")
human_label_counts = df['휴먼 분류'].value_counts()
print(human_label_counts)

# 메시지 길이 계산
df['메시지 길이'] = df['내용'].str.len()
df['단어 수'] = df['내용'].apply(lambda x: len(str(x).split()))

# URL 및 전화번호 포함 여부 확인
df['URL 포함'] = df['내용'].str.contains(r'http|www|\.com|\.kr|\.net', case=False, regex=True)
df['전화번호 포함'] = df['내용'].str.contains(r'\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}', regex=True)

# 분류별 특성 분석
print("\n분류별 메시지 길이 통계:")
length_stats = df.groupby('휴먼 분류')['메시지 길이'].describe()
print(length_stats)

print("\n분류별 URL 포함 비율:")
url_ratio = df.groupby('휴먼 분류')['URL 포함'].mean()
print(url_ratio)

print("\n분류별 전화번호 포함 비율:")
phone_ratio = df.groupby('휴먼 분류')['전화번호 포함'].mean()
print(phone_ratio)

# AI 분류와 휴먼 분류 일치율
print("\nAI 분류와 휴먼 분류 일치율:")
ai_human_match_ratio = (df['AI 분류'] == df['휴먼 분류']).mean()
print(f"{ai_human_match_ratio:.4f}")

# 이진 분류를 위한 레이블 생성 (비스팸 vs 스팸)
df['is_spam'] = df['휴먼 분류'].apply(lambda x: 0 if x == '비스팸' else 1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    df['내용'], df['is_spam'], test_size=0.3, random_state=42
)

print(f"\n훈련 데이터 크기: {X_train.shape[0]}")
print(f"테스트 데이터 크기: {X_test.shape[0]}")

# 특성 추출
print("\nTF-IDF 특성 추출 중...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 모델 훈련
print("RandomForest 모델 훈련 중...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# 예측 및 평가
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n정확도: {accuracy:.4f}")

print("\n분류 보고서:")
report = classification_report(y_test, y_pred)
print(report)

# 특성 중요도 확인
feature_names = vectorizer.get_feature_names_out()
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[-10:]

print("\n상위 10개 중요 단어:")
for i in indices:
    print(f"{feature_names[i]}: {feature_importances[i]:.4f}")

# 결과를 파일로 저장
with open('results/spam_analysis/model_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"데이터셋 크기: {df.shape}\n")
    f.write(f"컬럼: {df.columns.tolist()}\n\n")
    
    f.write("휴먼 분류 값 분포:\n")
    f.write(str(human_label_counts) + "\n\n")
    
    f.write("분류별 메시지 길이 통계:\n")
    f.write(str(length_stats) + "\n\n")
    
    f.write("분류별 URL 포함 비율:\n")
    f.write(str(url_ratio) + "\n\n")
    
    f.write("분류별 전화번호 포함 비율:\n")
    f.write(str(phone_ratio) + "\n\n")
    
    f.write(f"AI 분류와 휴먼 분류 일치율: {ai_human_match_ratio:.4f}\n\n")
    
    f.write(f"훈련 데이터 크기: {X_train.shape[0]}\n")
    f.write(f"테스트 데이터 크기: {X_test.shape[0]}\n\n")
    
    f.write(f"정확도: {accuracy:.4f}\n\n")
    
    f.write("분류 보고서:\n")
    f.write(report + "\n\n")
    
    f.write("상위 10개 중요 단어:\n")
    for i in indices:
        f.write(f"{feature_names[i]}: {feature_importances[i]:.4f}\n")

print(f"\n분석 결과가 'results/spam_analysis/model_results.txt'에 저장되었습니다.")

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(len(indices)), feature_importances[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
plt.title('상위 10개 중요 단어')
plt.tight_layout()
plt.savefig('results/spam_analysis/feature_importance.png')

# 분류별 메시지 길이 분포
plt.figure(figsize=(12, 6))
sns.boxplot(x='휴먼 분류', y='메시지 길이', data=df)
plt.title('분류별 메시지 길이 분포')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/spam_analysis/message_length_by_class.png')

# 분류별 URL 및 전화번호 포함 비율
plt.figure(figsize=(12, 6))
df_grouped = df.groupby('휴먼 분류')[['URL 포함', '전화번호 포함']].mean().reset_index()
df_melted = pd.melt(df_grouped, id_vars='휴먼 분류', var_name='특성', value_name='비율')
sns.barplot(x='휴먼 분류', y='비율', hue='특성', data=df_melted)
plt.title('분류별 URL 및 전화번호 포함 비율')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/spam_analysis/url_phone_ratio_by_class.png')

print("시각화 결과가 'results/spam_analysis/' 디렉토리에 저장되었습니다.")

# Claude 3.5 Sonnet을 사용한 스팸 판정 수행 여부 확인
if args.analyze_only:
    print("\n데이터 분석만 수행하도록 설정되어 Claude 판정을 건너뜁니다.")
    print("Claude 판정을 수행하려면 --analyze_only 옵션을 제거하세요.")
    exit(0)

# AWS Bedrock Claude 3.5 Sonnet을 사용한 스팸 판정 추가
print("\nClaude 3.5 Sonnet을 사용한 스팸 판정 시작...")

# AWS Bedrock 클라이언트 초기화
bedrock_client = get_bedrock_client()

if bedrock_client:
    # 현재 시간을 사용하여 결과 파일명 생성
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not args.process_all:
        # 샘플 메시지 선택
        sample_size = min(args.claude_sample_size, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        sample_df = df.iloc[sample_indices].copy()
        
        # Claude 3.5 Sonnet을 사용한 스팸 판정 결과 저장
        sample_df['Claude_판정'] = None
        sample_df['Claude_판정_근거'] = None
        sample_df['판정_시간'] = None
        
        for idx, row in sample_df.iterrows():
            print(f"메시지 {idx+1}/{sample_size} 처리 중...")
            start_time = time.time()
            classification, reason = classify_spam_with_claude(row['내용'], bedrock_client, args.detailed_reason)
            end_time = time.time()
            
            sample_df.at[idx, 'Claude_판정'] = classification
            sample_df.at[idx, 'Claude_판정_근거'] = reason
            sample_df.at[idx, '판정_시간'] = end_time - start_time
        
        # 결과 저장
        result_file = f'results/spam_analysis/claude_spam_classification_{current_time}.xlsx'
        sample_df.to_excel(result_file, index=False)
        print(f"\nClaude 3.5 Sonnet 스팸 판정 결과가 '{result_file}'에 저장되었습니다.")
        
        # 판정 결과 요약
        claude_human_match = (sample_df['Claude_판정'] == sample_df['휴먼 분류']).mean()
        print(f"Claude와 휴먼 분류 일치율: {claude_human_match:.4f}")
        
        claude_ai_match = (sample_df['Claude_판정'] == sample_df['AI 분류']).mean()
        print(f"Claude와 기존 AI 분류 일치율: {claude_ai_match:.4f}")
        
        avg_time = sample_df['판정_시간'].mean()
        print(f"평균 판정 시간: {avg_time:.2f}초")
        
        # 판정 결과 시각화
        plt.figure(figsize=(10, 6))
        comparison_df = pd.DataFrame({
            '휴먼 분류': (sample_df['Claude_판정'] == sample_df['휴먼 분류']).mean(),
            'AI 분류': (sample_df['Claude_판정'] == sample_df['AI 분류']).mean()
        }, index=['일치율'])
        comparison_df.plot(kind='bar')
        plt.title('Claude 3.5 Sonnet 판정 결과 비교')
        plt.ylabel('일치율')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'results/spam_analysis/claude_comparison_{current_time}.png')
        
        print("\n전체 데이터셋에 Claude 판정 결과를 추가하려면 --process_all 옵션을 사용하세요.")
    
    else:
        # 전체 데이터셋에 Claude 판정 결과 추가
        print(f"\n전체 데이터셋({len(df)}개 메시지)에 Claude 판정 결과 추가 중...")
        
        # 새 컬럼 추가
        df['Claude_판정'] = None
        df['Claude_판정_근거'] = None
        df['Claude_판정_시간'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 진행 상황 표시를 위한 변수
        total_count = len(df)
        progress_step = max(1, total_count // 20)  # 5% 단위로 진행 상황 표시
        
        for i in range(total_count):
            if i % progress_step == 0:
                print(f"진행 상황: {i}/{total_count} ({i/total_count*100:.1f}%)")
            
            classification, reason = classify_spam_with_claude(df.iloc[i]['내용'], bedrock_client, args.detailed_reason)
            df.at[i, 'Claude_판정'] = classification
            df.at[i, 'Claude_판정_근거'] = reason
            
            # 엑셀 업로드 일시 컬럼에 Claude 판정 결과 추가
            # 기존 값이 NaN이거나 비어있는 경우에만 업데이트
            if pd.isna(df.at[i, '엑셀 업로드 일시']) or df.at[i, '엑셀 업로드 일시'] == '':
                df.at[i, '엑셀 업로드 일시'] = f"Claude 판정: {classification} - {reason[:50]}..." if len(reason) > 50 else f"Claude 판정: {classification} - {reason}"
        
        # 결과 저장
        output_file = f'data/스팸리스트_20250310_개인정보_삭제_Claude판정_{current_time}.xlsx'
        df.to_excel(output_file, index=False)
        print(f"\nClaude 판정 결과가 '{output_file}'에 저장되었습니다.")
        
        # 판정 결과 요약
        claude_human_match = (df['Claude_판정'] == df['휴먼 분류']).mean()
        print(f"Claude와 휴먼 분류 일치율: {claude_human_match:.4f}")
        
        claude_ai_match = (df['Claude_판정'] == df['AI 분류']).mean()
        print(f"Claude와 기존 AI 분류 일치율: {claude_ai_match:.4f}")
        
        # 판정 결과 시각화
        plt.figure(figsize=(10, 6))
        comparison_df = pd.DataFrame({
            '휴먼 분류': (df['Claude_판정'] == df['휴먼 분류']).mean(),
            'AI 분류': (df['Claude_판정'] == df['AI 분류']).mean()
        }, index=['일치율'])
        comparison_df.plot(kind='bar')
        plt.title('Claude 3.5 Sonnet 판정 결과 비교 (전체 데이터셋)')
        plt.ylabel('일치율')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'results/spam_analysis/claude_full_comparison_{current_time}.png')
        
        # 분류별 Claude 판정 결과 분석
        plt.figure(figsize=(12, 6))
        claude_counts = df['Claude_판정'].value_counts()
        claude_counts.plot(kind='bar')
        plt.title('Claude 3.5 Sonnet 판정 결과 분포')
        plt.ylabel('메시지 수')
        plt.tight_layout()
        plt.savefig(f'results/spam_analysis/claude_classification_counts_{current_time}.png')
        
        # 혼동 행렬 시각화 (휴먼 분류 vs Claude 판정)
        plt.figure(figsize=(10, 8))
        confusion_matrix = pd.crosstab(df['휴먼 분류'], df['Claude_판정'], normalize='index')
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='.2f')
        plt.title('휴먼 분류 vs Claude 판정 혼동 행렬')
        plt.tight_layout()
        plt.savefig(f'results/spam_analysis/claude_confusion_matrix_{current_time}.png')
        
        # 판정 결과 상세 분석 보고서 생성
        with open(f'results/spam_analysis/claude_analysis_report_{current_time}.txt', 'w', encoding='utf-8') as f:
            f.write(f"# Claude 3.5 Sonnet 스팸 판정 분석 보고서\n\n")
            f.write(f"분석 일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"데이터셋: {args.input_file}\n")
            f.write(f"데이터셋 크기: {len(df)}개 메시지\n\n")
            
            f.write(f"## 판정 결과 요약\n\n")
            f.write(f"Claude 판정 분포:\n")
            f.write(str(df['Claude_판정'].value_counts()) + "\n\n")
            
            f.write(f"Claude와 휴먼 분류 일치율: {claude_human_match:.4f}\n")
            f.write(f"Claude와 기존 AI 분류 일치율: {claude_ai_match:.4f}\n\n")
            
            f.write(f"## 분류별 일치율\n\n")
            for category in df['휴먼 분류'].unique():
                category_df = df[df['휴먼 분류'] == category]
                match_rate = (category_df['Claude_판정'] == category_df['휴먼 분류']).mean()
                f.write(f"{category}: {match_rate:.4f} ({len(category_df)}개 메시지)\n")
            
            f.write(f"\n## 불일치 사례 분석\n\n")
            mismatch_df = df[df['Claude_판정'] != df['휴먼 분류']].copy()
            f.write(f"불일치 사례 수: {len(mismatch_df)}개 ({len(mismatch_df)/len(df)*100:.2f}%)\n\n")
            
            # 불일치 사례 중 일부 샘플 출력 (최대 10개)
            sample_count = min(10, len(mismatch_df))
            f.write(f"불일치 사례 샘플 ({sample_count}개):\n\n")
            
            for i in range(sample_count):
                row = mismatch_df.iloc[i]
                f.write(f"사례 {i+1}:\n")
                f.write(f"메시지: {row['내용'][:100]}...\n")
                f.write(f"휴먼 분류: {row['휴먼 분류']}\n")
                f.write(f"Claude 판정: {row['Claude_판정']}\n")
                f.write(f"Claude 판정 근거: {row['Claude_판정_근거'][:200]}...\n\n")
        
        print(f"\n상세 분석 보고서가 'results/spam_analysis/claude_analysis_report_{current_time}.txt'에 저장되었습니다.")
        
else:
    print("\nAWS Bedrock 클라이언트 연결에 실패하여 Claude 스팸 판정을 수행할 수 없습니다.")
    print("AWS 자격 증명과 리전 설정을 확인하세요.")
    print("aws_config.py 파일에 올바른 AWS 자격 증명을 설정했는지 확인하세요.")

print("\n분석 완료!")

# 사용 방법 출력
print("\n사용 방법:")
print("1. 데이터 분석만 수행: python analyze_spam_data.py --analyze_only")
print("2. 샘플 데이터 Claude 판정: python analyze_spam_data.py --claude_sample_size 20")
print("3. 전체 데이터셋 Claude 판정: python analyze_spam_data.py --process_all")
print("4. 다른 입력 파일 사용: python analyze_spam_data.py --input_file 파일경로.xlsx")
print("5. 상세한 판정 근거 요청: python analyze_spam_data.py --detailed_reason") 