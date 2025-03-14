"""
테스트용 샘플 데이터 생성 스크립트

이 스크립트는 SMS 스팸 필터 시스템 테스트를 위한 샘플 데이터를 생성합니다.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# 정상 SMS 메시지 샘플
normal_messages = [
    "안녕하세요, 오늘 회의 시간 알려드립니다. 오후 2시에 회의실에서 뵙겠습니다.",
    "내일 점심 같이 먹을래요? 12시에 회사 앞 식당에서 만나요.",
    "주문하신 상품이 배송 완료되었습니다. 감사합니다.",
    "오늘 저녁에 약속 있으신가요? 시간 되시면 연락주세요.",
    "어제 보내주신 자료 잘 받았습니다. 검토 후 연락드리겠습니다.",
    "다음 주 화요일에 휴가를 쓰려고 합니다. 일정 조정 부탁드립니다.",
    "프로젝트 마감일이 다음 주 금요일로 연장되었습니다.",
    "오늘 날씨가 좋네요. 퇴근 후 산책 어떠세요?",
    "어제 말씀하신 문서 첨부해서 보내드립니다.",
    "내일 회의 자료 준비 부탁드립니다.",
    "주말에 가족 모임이 있어서 참석이 어렵습니다. 죄송합니다.",
    "지난번에 문의하신 건에 대해 답변드립니다. 자세한 내용은 이메일로 보내드렸습니다.",
    "오늘 저녁 식사 약속 시간에 늦을 것 같습니다. 30분 정도 늦을 예정입니다.",
    "내일 오전에 시간 되시면 커피 한잔 하실래요?",
    "주문하신 상품의 재고가 부족하여 다음 주에 발송될 예정입니다. 양해 부탁드립니다.",
    "회의 장소가 변경되었습니다. 3층 대회의실에서 진행됩니다.",
    "어제 보내주신 파일이 열리지 않습니다. 다시 보내주실 수 있을까요?",
    "다음 달 일정 조율을 위해 가능한 날짜 알려주세요.",
    "오늘 점심 식사 맛있게 하셨나요? 오후에 뵙겠습니다.",
    "내일 출장 가시는 거 맞죠? 필요한 서류 준비해 드리겠습니다."
]

# 스팸 SMS 메시지 샘플
spam_messages = [
    "무료 상품권 100만원 당첨! 지금 바로 확인하세요 http://scam.com",
    "비용 없이 즉시 대출 가능합니다. 지금 전화주세요 010-1234-5678",
    "축하합니다! 귀하는 아이폰15 당첨자로 선정되었습니다. 수령하기: http://fakeprize.kr",
    "적금 대비 10배 수익 보장! 지금 투자하세요. 문의: 02-123-4567",
    "마지막 찬스! 오늘까지만 90% 할인 이벤트 중입니다. 지금 클릭: www.fakesale.com",
    "귀하의 계좌가 해킹 위험에 노출되었습니다. 즉시 확인: http://bankscam.kr",
    "건강보험공단 환급금 5만원 발생. 아래 링크에서 확인하세요 http://fakehealthcare.kr",
    "럭셔리 명품 시계 초특가 세일! 정품 보장! 지금 주문: 010-9876-5432",
    "세금 환급금이 발생했습니다. 아래 링크를 통해 확인하세요 http://taxrefundscam.com",
    "당신의 소포가 배송 중 분실되었습니다. 확인: http://deliveryscam.kr",
    "긴급! 귀하의 신용카드가 해외에서 사용되었습니다. 확인: http://cardscam.com",
    "무료 주식 정보! 다음 주 급등주 공개합니다. 지금 등록: 010-1111-2222",
    "마지막 기회! 코인 대박 정보! 지금 바로 투자하세요. 문의: 010-3333-4444",
    "비아그라 정품 50% 할인! 비밀 배송! 주문: 010-5555-6666",
    "당신만을 위한 특별 대출! 신용 무관! 최대 5천만원! 지금 전화: 02-333-4444",
    "국세청 세금 체납 안내. 24시간 내 납부하지 않으면 법적 조치 진행됩니다. 확인: http://taxscam.kr",
    "귀하의 통신비 과다 청구 확인되었습니다. 환급 신청: http://telecomscam.com",
    "코로나 재난지원금 신청하세요. 마감 임박! 신청: http://covidscam.kr",
    "당신의 개인정보 유출 확인됨! 지금 바로 확인하세요: http://privacyscam.com",
    "적금보다 높은 수익! 원금 보장! 지금 상담: 010-7777-8888"
]

def create_sample_data(output_file='sample_sms_data.xlsx', num_samples=100, spam_ratio=0.3):
    """
    샘플 SMS 데이터 생성
    
    Args:
        output_file: 출력 파일 경로
        num_samples: 생성할 샘플 수
        spam_ratio: 스팸 메시지 비율
    """
    # 스팸 및 정상 메시지 수 계산
    num_spam = int(num_samples * spam_ratio)
    num_normal = num_samples - num_spam
    
    # 메시지 샘플링 (중복 허용)
    sampled_normal = np.random.choice(normal_messages, num_normal, replace=True)
    sampled_spam = np.random.choice(spam_messages, num_spam, replace=True)
    
    # 데이터프레임 생성
    messages = np.concatenate([sampled_normal, sampled_spam])
    labels = np.concatenate([np.zeros(num_normal), np.ones(num_spam)])
    
    # 인덱스 섞기
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    messages = messages[indices]
    labels = labels[indices]
    
    # 타임스탬프 생성
    timestamps = [datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in range(num_samples)]
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'timestamp': timestamps,
        'message': messages,
        'is_spam': labels.astype(int)
    })
    
    # 파일 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_excel(output_file, index=False)
    
    print(f"샘플 데이터가 '{output_file}'에 저장되었습니다.")
    print(f"총 {num_samples}개의 메시지 (정상: {num_normal}, 스팸: {num_spam})")
    
    return df

def create_unlabeled_data(output_file='unlabeled_sms_data.xlsx', num_samples=50, spam_ratio=0.3):
    """
    레이블이 없는 샘플 SMS 데이터 생성 (예측용)
    
    Args:
        output_file: 출력 파일 경로
        num_samples: 생성할 샘플 수
        spam_ratio: 스팸 메시지 비율 (내부적으로만 사용)
    """
    # 스팸 및 정상 메시지 수 계산
    num_spam = int(num_samples * spam_ratio)
    num_normal = num_samples - num_spam
    
    # 메시지 샘플링 (중복 허용)
    sampled_normal = np.random.choice(normal_messages, num_normal, replace=True)
    sampled_spam = np.random.choice(spam_messages, num_spam, replace=True)
    
    # 데이터프레임 생성
    messages = np.concatenate([sampled_normal, sampled_spam])
    
    # 인덱스 섞기
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    messages = messages[indices]
    
    # 타임스탬프 생성
    timestamps = [datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in range(num_samples)]
    
    # 데이터프레임 생성 (레이블 없음)
    df = pd.DataFrame({
        'timestamp': timestamps,
        'message': messages
    })
    
    # 파일 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_excel(output_file, index=False)
    
    print(f"레이블이 없는 샘플 데이터가 '{output_file}'에 저장되었습니다.")
    print(f"총 {num_samples}개의 메시지")
    
    return df

if __name__ == "__main__":
    # 랜덤 시드 설정
    np.random.seed(42)
    
    # 학습용 데이터 생성 (레이블 포함)
    train_df = create_sample_data('data/train_sms_data.xlsx', num_samples=100, spam_ratio=0.3)
    
    # 테스트용 데이터 생성 (레이블 포함)
    test_df = create_sample_data('data/test_sms_data.xlsx', num_samples=50, spam_ratio=0.3)
    
    # 예측용 데이터 생성 (레이블 없음)
    unlabeled_df = create_unlabeled_data('data/unlabeled_sms_data.xlsx', num_samples=30, spam_ratio=0.3) 