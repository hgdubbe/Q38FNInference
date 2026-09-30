# Local equivalent of .github/workflows/build-windows-cuda.yml: builds
# llama.cpp (llama-server + CPU backends with clang, ggml-cuda.dll with
# MSVC/nvcc) at the pinned commit, applies this repo's patches, and puts it
# all next to the launcher in .\dist\Q38FNInference.
#
# Requirements: Visual Studio 2022 or newer with the C++ workload and the
# "C++ Clang tools" component, CMake 3.21+, Ninja, CUDA Toolkit 12.4+, Go
# 1.24+, git.
#
# Usage: .\scripts\build-windows-cuda.ps1 [-LlamaCppRef <commit>]

param(
    # patches/*.patch are verified against this commit
    [string]$LlamaCppRef = "6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17",
    [string]$WorkDir = "$PSScriptRoot\..\.build"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$outDir = Join-Path $repoRoot "dist\Q38FNInference"

function Invoke-Checked([string]$cmd) {
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $cmd" }
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "Visual Studio with the C++ workload not found" }
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
if (-not $env:CUDA_PATH) { throw "CUDA_PATH is not set; install the CUDA Toolkit first" }

New-Item -ItemType Directory -Force -Path $WorkDir, $outDir | Out-Null
$llama = Join-Path $WorkDir "llama.cpp"
if (-not (Test-Path "$llama\.git")) {
    git clone https://github.com/ggml-org/llama.cpp $llama
}
Push-Location $llama
try {
    git fetch origin $LlamaCppRef
    git checkout --force FETCH_HEAD
    Get-ChildItem "$repoRoot\patches\*.patch" | ForEach-Object {
        Write-Host "==> applying $($_.Name)"
        git apply --verbose $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "patch $($_.Name) does not apply to $LlamaCppRef" }
    }

    Write-Host "==> llama-server + CPU backends"
    Invoke-Checked "`"$vcvars`" x64 && cmake -S . -B build-cpu -G `"Ninja Multi-Config`" -D CMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake -DLLAMA_BUILD_BORINGSSL=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=ON -DGGML_OPENMP_FETCH=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON && cmake --build build-cpu --config Release"

    Write-Host "==> ggml-cuda.dll"
    Invoke-Checked "`"$vcvars`" x64 && cmake -S . -B build-cuda -G `"Ninja Multi-Config`" -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_CUDA=ON -DLLAMA_BUILD_BORINGSSL=ON -DGGML_CUDA_CUB_3DOT2=ON && cmake --build build-cuda --config Release --target ggml-cuda"

    Copy-Item build-cpu\bin\Release\* $outDir -Recurse -Force
    Copy-Item build-cuda\bin\Release\ggml-cuda.dll $outDir -Force
    Copy-Item LICENSE (Join-Path $outDir "LICENSE-llama.cpp.txt") -Force
} finally {
    Pop-Location
}

foreach ($d in "$env:CUDA_PATH\bin", "$env:CUDA_PATH\bin\x64") {
    if (Test-Path $d) {
        Copy-Item "$d\cudart64_*.dll", "$d\cublas64_*.dll", "$d\cublasLt64_*.dll" $outDir -ErrorAction SilentlyContinue
    }
}

Write-Host "==> launcher"
Push-Location $repoRoot
try {
    $env:GOOS = "windows"; $env:GOARCH = "amd64"; $env:CGO_ENABLED = "0"
    go build -trimpath -ldflags "-H windowsgui -s -w" -o (Join-Path $outDir "q38fninference.exe") ./cmd/q38fninference
    if ($LASTEXITCODE -ne 0) { throw "go build failed" }
} finally {
    Remove-Item Env:\GOOS, Env:\GOARCH, Env:\CGO_ENABLED -ErrorAction SilentlyContinue
    Pop-Location
}

Write-Host "`nDone: run $outDir\q38fninference.exe"
