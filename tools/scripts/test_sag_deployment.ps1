#
# SAG 功能测试和部署脚本 (Windows PowerShell)
# 使用方法: .\test_sag_deployment.ps1 [-Step <step>]
#

param(
    [ValidateSet("syntax", "frontend", "docker-build", "deploy", "api", "test", "all")]
    [string]$Step = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# =============================================================================
# 1. 后端 Python 语法检查
# =============================================================================
function Check-PythonSyntax {
    Write-Info "检查 Python 语法..."
    
    Set-Location $ProjectRoot
    
    $files = @(
        "rag/sag/__init__.py",
        "rag/sag/config.py",
        "rag/sag/models.py",
        "rag/sag/cleanup.py",
        "rag/sag/task_queue.py",
        "rag/sag/retriever.py",
        "rag/sag/extractor.py",
        "api/apps/restful_apis/sag_api.py"
    )
    
    foreach ($file in $files) {
        python -m py_compile $file
        if ($LASTEXITCODE -ne 0) {
            Write-Err "语法检查失败: $file"
            exit 1
        }
    }
    
    Write-Info "Python 语法检查通过 ✓"
}

# =============================================================================
# 2. 前端 TypeScript 编译检查
# =============================================================================
function Check-FrontendBuild {
    Write-Info "检查前端构建..."
    
    Set-Location "$ProjectRoot/web"
    
    # 安装依赖（使用淘宝镜像）
    if (-not (Test-Path "node_modules")) {
        Write-Info "安装前端依赖..."
        npm config set registry https://registry.npmmirror.com
        npm install
    }
    
    # TypeScript 类型检查
    Write-Info "运行 TypeScript 类型检查..."
    npm run type-check
    
    # 构建前端
    Write-Info "构建前端..."
    npm run build
    
    Write-Info "前端构建通过 ✓"
}

# =============================================================================
# 3. Docker 镜像构建（使用国内镜像源）
# =============================================================================
function Build-DockerImage {
    Write-Info "构建 Docker 镜像（使用国内镜像源）..."
    
    Set-Location $ProjectRoot
    
    # 使用 NEED_MIRROR=1 启用国内镜像
    docker build `
        --build-arg NEED_MIRROR=1 `
        -t ragflow-sag:latest `
        -f Dockerfile `
        .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Docker 构建失败"
        exit 1
    }
    
    Write-Info "Docker 镜像构建完成 ✓"
}

# =============================================================================
# 4. Docker Compose 部署
# =============================================================================
function Deploy-WithDockerCompose {
    Write-Info "使用 Docker Compose 部署..."
    
    Set-Location "$ProjectRoot/docker"
    
    # 设置环境变量使用本地构建的镜像
    $env:RAGFLOW_IMAGE = "ragflow-sag:latest"
    
    # 启动服务
    docker compose --profile cpu up -d
    
    Write-Info "等待服务启动..."
    Start-Sleep -Seconds 30
    
    # 检查服务状态
    docker compose ps
    
    Write-Info "Docker Compose 部署完成 ✓"
}

# =============================================================================
# 5. API 功能验证
# =============================================================================
function Verify-SagApis {
    Write-Info "验证 SAG API..."
    
    $BaseUrl = $env:RAGFLOW_URL ?? "http://localhost"
    $ApiKey = $env:RAGFLOW_API_KEY
    
    if (-not $ApiKey) {
        Write-Warn "未设置 RAGFLOW_API_KEY，跳过 API 验证"
        Write-Warn "请设置: `$env:RAGFLOW_API_KEY = 'your_api_key'"
        return
    }
    
    $headers = @{
        "Authorization" = "Bearer $ApiKey"
        "Content-Type" = "application/json"
    }
    
    # 创建测试知识库
    Write-Info "创建测试知识库..."
    $kbBody = @{ name = "SAG Test KB"; chunk_method = "naive" } | ConvertTo-Json
    $kbResponse = Invoke-RestMethod -Uri "$BaseUrl/api/v1/datasets" -Method POST -Headers $headers -Body $kbBody
    $kbId = $kbResponse.data.id
    
    if (-not $kbId) {
        Write-Err "创建知识库失败"
        return
    }
    
    Write-Info "知识库 ID: $kbId"
    
    # 测试 SAG 配置 API
    Write-Info "测试 GET /api/v1/sag/kb/{kb_id}/config ..."
    Invoke-RestMethod -Uri "$BaseUrl/api/v1/sag/kb/$kbId/config" -Headers $headers | ConvertTo-Json
    
    # 测试更新 SAG 配置
    Write-Info "测试 PUT /api/v1/sag/kb/{kb_id}/config ..."
    $configBody = @{ enabled = $true; search_strategy = "multi"; search_top_k = 10 } | ConvertTo-Json
    Invoke-RestMethod -Uri "$BaseUrl/api/v1/sag/kb/$kbId/config" -Method PUT -Headers $headers -Body $configBody | ConvertTo-Json
    
    # 测试 SAG 状态 API
    Write-Info "测试 GET /api/v1/sag/kb/{kb_id}/status ..."
    Invoke-RestMethod -Uri "$BaseUrl/api/v1/sag/kb/$kbId/status" -Headers $headers | ConvertTo-Json
    
    # 测试图谱 API
    Write-Info "测试 GET /api/v1/sag/kb/{kb_id}/graph ..."
    Invoke-RestMethod -Uri "$BaseUrl/api/v1/sag/kb/$kbId/graph?event_limit=10&entity_limit=10" -Headers $headers | ConvertTo-Json
    
    # 测试实体列表 API
    Write-Info "测试 GET /api/v1/sag/kb/{kb_id}/entities ..."
    Invoke-RestMethod -Uri "$BaseUrl/api/v1/sag/kb/$kbId/entities?page=1&page_size=10" -Headers $headers | ConvertTo-Json
    
    # 测试事件列表 API
    Write-Info "测试 GET /api/v1/sag/kb/{kb_id}/events ..."
    Invoke-RestMethod -Uri "$BaseUrl/api/v1/sag/kb/$kbId/events?page=1&page_size=10" -Headers $headers | ConvertTo-Json
    
    # 清理：删除测试知识库
    Write-Info "清理测试知识库..."
    $deleteBody = @{ ids = @($kbId) } | ConvertTo-Json
    Invoke-RestMethod -Uri "$BaseUrl/api/v1/datasets" -Method DELETE -Headers $headers -Body $deleteBody
    
    Write-Info "SAG API 验证完成 ✓"
}

# =============================================================================
# 主函数
# =============================================================================
Write-Host "========================================"
Write-Host "SAG 功能测试和部署脚本"
Write-Host "========================================"
Write-Host ""

switch ($Step) {
    "syntax" {
        Check-PythonSyntax
    }
    "frontend" {
        Check-FrontendBuild
    }
    "docker-build" {
        Build-DockerImage
    }
    "deploy" {
        Deploy-WithDockerCompose
    }
    "api" {
        Verify-SagApis
    }
    "all" {
        Check-PythonSyntax
        Write-Info "基础检查完成。"
        Write-Info ""
        Write-Info "后续步骤（需要相应环境）："
        Write-Info "  1. 前端构建: .\test_sag_deployment.ps1 -Step frontend"
        Write-Info "  2. Docker 构建: .\test_sag_deployment.ps1 -Step docker-build"
        Write-Info "  3. 部署: .\test_sag_deployment.ps1 -Step deploy"
        Write-Info "  4. API 验证: .\test_sag_deployment.ps1 -Step api"
    }
    default {
        Write-Host "用法: .\test_sag_deployment.ps1 [-Step <syntax|frontend|docker-build|deploy|api|test|all>]"
        exit 1
    }
}
