# 기업 메시징 중계 서버 스팸 필터 시스템 설치 가이드

## 목차
1. [시스템 요구사항](#1-시스템-요구사항)
2. [사전 준비사항](#2-사전-준비사항)
3. [설치 과정](#3-설치-과정)
4. [설정 방법](#4-설정-방법)
5. [시스템 시작 및 중지](#5-시스템-시작-및-중지)
6. [문제 해결](#6-문제-해결)
7. [업그레이드 방법](#7-업그레이드-방법)

## 1. 시스템 요구사항

### 1.1 하드웨어 요구사항
- **CPU**: 최소 8코어 이상 (권장: 16코어 이상)
- **메모리**: 최소 16GB 이상 (권장: 32GB 이상)
- **디스크**: 최소 500GB SSD (권장: 1TB SSD 이상)
- **네트워크**: 1Gbps 이상의 네트워크 대역폭

### 1.2 소프트웨어 요구사항
- **운영체제**: Ubuntu 20.04 LTS 이상 또는 CentOS 8 이상
- **컨테이너 플랫폼**: Docker 20.10 이상
- **데이터베이스**: PostgreSQL 13 이상
- **웹 서버**: Nginx 1.18 이상
- **메시지 큐**: RabbitMQ 3.8 이상 또는 Kafka 2.8 이상

## 2. 사전 준비사항

### 2.1 운영체제 설치 및 업데이트
```bash
# Ubuntu 시스템 업데이트
sudo apt update
sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y curl wget git vim
```

### 2.2 Docker 설치
```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/download/v2.10.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Docker 서비스 시작 및 자동 시작 설정
sudo systemctl start docker
sudo systemctl enable docker

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER
```

### 2.3 데이터베이스 설치 (독립 실행 시)
```bash
# PostgreSQL 설치
sudo apt install -y postgresql postgresql-contrib

# PostgreSQL 서비스 시작 및 자동 시작 설정
sudo systemctl start postgresql
sudo systemctl enable postgresql

# 데이터베이스 및 사용자 생성
sudo -u postgres psql -c "CREATE DATABASE spamfilter;"
sudo -u postgres psql -c "CREATE USER spamfilter WITH ENCRYPTED PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE spamfilter TO spamfilter;"
```

## 3. 설치 과정

### 3.1 소스코드 다운로드
```bash
# 소스코드 클론
git clone https://github.com/medialog-AX-TF/spam-filter.git
cd spam-filter
```

### 3.2 환경 설정
```bash
# 환경 설정 파일 복사
cp .env.example .env

# 환경 설정 파일 편집
nano .env
```

`.env` 파일에서 다음 항목을 설정합니다:
```
# 데이터베이스 설정
DB_HOST=localhost
DB_PORT=5432
DB_NAME=spamfilter
DB_USER=spamfilter
DB_PASSWORD=your_password

# API 설정
API_PORT=8000
API_SECRET_KEY=your_secret_key

# 로깅 설정
LOG_LEVEL=info
LOG_PATH=/var/log/spamfilter

# 스팸 필터 설정
SPAM_THRESHOLD=0.7
MAX_MESSAGES_PER_SECOND=1000
```

### 3.3 Docker Compose를 이용한 설치
```bash
# Docker Compose로 시스템 빌드 및 실행
docker-compose build
docker-compose up -d
```

## 4. 설정 방법

### 4.1 스팸 필터 엔진 설정
스팸 필터 엔진의 설정은 관리자 웹 인터페이스를 통해 수행할 수 있습니다. 기본 설정은 다음과 같습니다:

- **스팸 판단 임계값**: 0.7 (0.0 ~ 1.0 사이의 값, 높을수록 엄격함)
- **키워드 필터링**: 기본 키워드 목록 제공
- **패턴 필터링**: 기본 패턴 목록 제공
- **블랙리스트**: 기본 블랙리스트 제공

### 4.2 API 서버 설정
API 서버의 설정은 `config/api.yaml` 파일에서 수정할 수 있습니다:

```yaml
server:
  port: 8000
  host: 0.0.0.0
  workers: 4
  timeout: 30

security:
  secret_key: your_secret_key
  token_expiration: 86400  # 24시간
  cors_origins:
    - http://localhost:3000
    - https://admin.example.com

rate_limiting:
  enabled: true
  max_requests: 1000
  time_window: 60  # 초
```

### 4.3 로깅 및 모니터링 설정
로깅 및 모니터링 설정은 `config/logging.yaml` 파일에서 수정할 수 있습니다:

```yaml
logging:
  level: info
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: /var/log/spamfilter/app.log
  max_size: 100MB
  backup_count: 10

monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    port: 3000
```

## 5. 시스템 시작 및 중지

### 5.1 Docker Compose를 이용한 시작 및 중지
```bash
# 시스템 시작
docker-compose up -d

# 시스템 상태 확인
docker-compose ps

# 시스템 로그 확인
docker-compose logs -f

# 시스템 중지
docker-compose down

# 시스템 재시작
docker-compose restart
```

## 6. 문제 해결

### 6.1 일반적인 문제 해결
- **데이터베이스 연결 오류**: 데이터베이스 호스트, 포트, 사용자 이름, 비밀번호가 올바른지 확인합니다.
- **API 서버 시작 실패**: 포트 충돌이 없는지 확인하고, 로그를 확인하여 오류 메시지를 확인합니다.
- **스팸 필터 엔진 오류**: 로그를 확인하여 오류 메시지를 확인하고, 필요한 경우 모델 파일이 올바르게 로드되었는지 확인합니다.

### 6.2 로그 확인
```bash
# Docker Compose 로그 확인
docker-compose logs -f

# 특정 서비스 로그 확인
docker-compose logs -f api
```

### 6.3 시스템 상태 확인
```bash
# Docker 컨테이너 상태 확인
docker ps

# Docker Compose 서비스 상태 확인
docker-compose ps
```

## 7. 업그레이드 방법

### 7.1 Docker Compose를 이용한 업그레이드
```bash
# 최신 소스코드 다운로드
git pull

# 이미지 재빌드
docker-compose build

# 서비스 재시작
docker-compose down
docker-compose up -d
```

### 7.2 데이터베이스 마이그레이션
```bash
# Docker Compose를 이용한 마이그레이션
docker-compose run --rm api python manage.py migrate
```

## 8. 추가 정보

### 8.1 보안 권장사항
- 모든 비밀번호와 API 키는 강력하고 고유한 값으로 설정하세요.
- 프로덕션 환경에서는 HTTPS를 사용하세요.
- 방화벽을 설정하여 필요한 포트만 외부에 노출하세요.
- 정기적으로 시스템 및 라이브러리를 업데이트하세요.

### 8.2 백업 및 복구
```bash
# 데이터베이스 백업
docker-compose exec db pg_dump -U spamfilter spamfilter > backup.sql

# 데이터베이스 복구
cat backup.sql | docker-compose exec -T db psql -U spamfilter spamfilter
```

### 8.3 모니터링 접근
- **Prometheus**: http://your-server:9090
- **Grafana**: http://your-server:3000
- **관리자 웹 인터페이스**: http://your-server:3000/admin 