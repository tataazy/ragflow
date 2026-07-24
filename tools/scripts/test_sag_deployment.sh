#!/bin/bash
#
# SAG 功能测试和部署脚本
# 使用方法: ./test_sag_deployment.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# 1. 后端 Python 语法检查
# =============================================================================
check_python_syntax() {
    log_info "检查 Python 语法..."
    
    cd "$PROJECT_ROOT"
    
    # 检查 SAG 模块语法
    python3 -m py_compile rag/sag/__init__.py
    python3 -m py_compile rag/sag/config.py
    python3 -m py_compile rag/sag/models.py
    python3 -m py_compile rag/sag/cleanup.py
    python3 -m py_compile rag/sag/task_queue.py
    python3 -m py_compile rag/sag/retriever.py
    python3 -m py_compile rag/sag/extractor.py
    python3 -m py_compile api/apps/restful_apis/sag_api.py
    
    log_info "Python 语法检查通过 ✓"
}

# =============================================================================
# 2. 前端 TypeScript 编译检查
# =============================================================================
check_frontend_build() {
    log_info "检查前端构建..."
    
    cd "$PROJECT_ROOT/web"
    
    # 安装依赖（使用淘宝镜像）
    if [ ! -d "node_modules" ]; then
        log_info "安装前端依赖..."
        npm config set registry https://registry.npmmirror.com
        npm install
    fi
    
    # TypeScript 类型检查
    log_info "运行 TypeScript 类型检查..."
    npm run type-check
    
    # 构建前端
    log_info "构建前端..."
    npm run build
    
    log_info "前端构建通过 ✓"
}

# =============================================================================
# 3. Docker 镜像构建（使用国内镜像源）
# =============================================================================
build_docker_image() {
    log_info "构建 Docker 镜像（使用国内镜像源）..."
    
    cd "$PROJECT_ROOT"
    
    # 使用 NEED_MIRROR=1 启用国内镜像
    docker build \
        --build-arg NEED_MIRROR=1 \
        -t ragflow-sag:latest \
        -f Dockerfile \
        .
    
    log_info "Docker 镜像构建完成 ✓"
}

# =============================================================================
# 4. Docker Compose 部署
# =============================================================================
deploy_with_docker_compose() {
    log_info "使用 Docker Compose 部署..."
    
    cd "$PROJECT_ROOT/docker"
    
    # 修改 .env 使用本地构建的镜像
    export RAGFLOW_IMAGE=ragflow-sag:latest
    
    # 启动服务
    docker compose --profile cpu up -d
    
    log_info "等待服务启动..."
    sleep 30
    
    # 检查服务状态
    docker compose ps
    
    log_info "Docker Compose 部署完成 ✓"
}

# =============================================================================
# 5. API 功能验证
# =============================================================================
verify_sag_apis() {
    log_info "验证 SAG API..."
    
    BASE_URL="${RAGFLOW_URL:-http://localhost}"
    API_KEY="${RAGFLOW_API_KEY:-}"
    
    if [ -z "$API_KEY" ]; then
        log_warn "未设置 RAGFLOW_API_KEY，跳过 API 验证"
        log_warn "请设置: export RAGFLOW_API_KEY=your_api_key"
        return 0
    fi
    
    # 需要先创建一个知识库
    log_info "创建测试知识库..."
    KB_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/datasets" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"name": "SAG Test KB", "chunk_method": "naive"}')
    
    KB_ID=$(echo "$KB_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['id'])" 2>/dev/null || echo "")
    
    if [ -z "$KB_ID" ]; then
        log_error "创建知识库失败"
        echo "$KB_RESPONSE"
        return 1
    fi
    
    log_info "知识库 ID: $KB_ID"
    
    # 测试 SAG 配置 API
    log_info "测试 GET /api/v1/sag/kb/{kb_id}/config ..."
    curl -s "$BASE_URL/api/v1/sag/kb/$KB_ID/config" \
        -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    
    # 测试更新 SAG 配置
    log_info "测试 PUT /api/v1/sag/kb/{kb_id}/config ..."
    curl -s -X PUT "$BASE_URL/api/v1/sag/kb/$KB_ID/config" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"enabled": true, "search_strategy": "multi", "search_top_k": 10}' | python3 -m json.tool
    
    # 测试 SAG 状态 API
    log_info "测试 GET /api/v1/sag/kb/{kb_id}/status ..."
    curl -s "$BASE_URL/api/v1/sag/kb/$KB_ID/status" \
        -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    
    # 测试图谱 API
    log_info "测试 GET /api/v1/sag/kb/{kb_id}/graph ..."
    curl -s "$BASE_URL/api/v1/sag/kb/$KB_ID/graph?event_limit=10&entity_limit=10" \
        -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    
    # 测试实体列表 API
    log_info "测试 GET /api/v1/sag/kb/{kb_id}/entities ..."
    curl -s "$BASE_URL/api/v1/sag/kb/$KB_ID/entities?page=1&page_size=10" \
        -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    
    # 测试事件列表 API
    log_info "测试 GET /api/v1/sag/kb/{kb_id}/events ..."
    curl -s "$BASE_URL/api/v1/sag/kb/$KB_ID/events?page=1&page_size=10" \
        -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
    
    # 清理：删除测试知识库
    log_info "清理测试知识库..."
    curl -s -X DELETE "$BASE_URL/api/v1/datasets" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"ids\": [\"$KB_ID\"]}"
    
    log_info "SAG API 验证完成 ✓"
}

# =============================================================================
# 6. 运行单元测试
# =============================================================================
run_unit_tests() {
    log_info "运行 SAG 单元测试..."
    
    cd "$PROJECT_ROOT"
    
    # 使用 pytest 运行 SAG 测试
    python3 -m pytest test/unit_test/rag/sag/ -v --tb=short
    
    log_info "单元测试完成 ✓"
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    echo "========================================"
    echo "SAG 功能测试和部署脚本"
    echo "========================================"
    echo ""
    
    case "${1:-all}" in
        syntax)
            check_python_syntax
            ;;
        frontend)
            check_frontend_build
            ;;
        docker-build)
            build_docker_image
            ;;
        deploy)
            deploy_with_docker_compose
            ;;
        api)
            verify_sag_apis
            ;;
        test)
            run_unit_tests
            ;;
        all)
            check_python_syntax
            # check_frontend_build  # 需要 node_modules
            # build_docker_image    # 需要 Docker
            # deploy_with_docker_compose
            # verify_sag_apis
            log_info "基础检查完成。Docker 构建和部署需要手动执行。"
            ;;
        *)
            echo "用法: $0 {syntax|frontend|docker-build|deploy|api|test|all}"
            exit 1
            ;;
    esac
}

main "$@"
