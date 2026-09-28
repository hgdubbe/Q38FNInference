# Builds llama.cpp with CUDA support for Windows, applies this project's
# patches, and packages it alongside the Go launcher into ./dist.
#
# Mirrors .github/workflows/build-windows-cuda.yml so a contributor without
# GitHub Actions access can reproduce the same release build locally.
#
# Requirements: Visual Studio 2022 (Desktop C++ workload), CMake, Ninja,
# CUDA Toolkit 12.4+, Go 1.24+, and git. Run from a "Developer PowerShell
# for VS 2022" prompt (or this script will try to locate vcvarsall.bat).
#
# Usage:
#   .\scripts\build-windows-cuda.ps1 [-LlamaCppRef master] [-OutDir .\dist]

param(
    [string]$LlamaCppRef = "master",
    [string]$OutDir = "$PSScriptRoot\..\dist",
    [string]$WorkDir = "$PSScriptRoot\..\.build"
)

$ErrorActionPreference = "Stop"

function Find-VcVarsAll {
    $candidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    throw "Could not find vcvarsall.bat; run this from a Developer PowerShell for VS 2022 instead."
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$repoRoot = Resolve-Path "$PSScriptRoot\.."
$llamaDir = Join-Path $WorkDir "llama.cpp"

if (-not (Test-Path $llamaDir)) {
    Write-Host "==> Cloning ggml-org/llama.cpp ($LlamaCppRef)"
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp $llamaDir
}
Push-Location $llamaDir
git fetch origin $LlamaCppRef
git checkout $LlamaCppRef
git reset --hard "origin/$LlamaCppRef" 2>$null

Write-Host "==> Applying Q38FNInference patches"
Get-ChildItem "$repoRoot\patches\*.patch" | ForEach-Object {
    Write-Host "    $($_.Name)"
    git apply --verbose $_.FullName
}

$vcvars = Find-VcVarsAll
Write-Host "==> Using $vcvars"

Write-Host "==> Configuring + building CPU/DL base (llama-server.exe)"
cmd /c "`"$vcvars`" x64 && cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=ON -DLLAMA_CURL=OFF && cmake --build build-cpu --config Release"
if ($LASTEXITCODE -ne 0) { throw "CPU/base build failed" }

Write-Host "==> Configuring + building ggml-cuda backend"
cmd /c "`"$vcvars`" x64 && cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_BACKEND_DL=ON -DGGML_CUDA=ON -DGGML_CUDA_F16=ON && cmake --build build-cuda --config Release --target ggml-cuda"
if ($LASTEXITCODE -ne 0) { throw "CUDA backend build failed" }

Pop-Location

Write-Host "==> Collecting binaries into $OutDir"
Copy-Item "$llamaDir\build-cpu\bin\*" $OutDir -Recurse -Force
Copy-Item "$llamaDir\build-cuda\bin\ggml-cuda.dll" $OutDir -Force

$cudaPath = $env:CUDA_PATH
if ($cudaPath) {
    Write-Host "==> Copying CUDA runtime DLLs from $cudaPath"
    Copy-Item "$cudaPath\bin\cudart64_*.dll" $OutDir -Force -ErrorAction SilentlyContinue
    Copy-Item "$cudaPath\bin\cublas64_*.dll" $OutDir -Force -ErrorAction SilentlyContinue
    Copy-Item "$cudaPath\bin\cublasLt64_*.dll" $OutDir -Force -ErrorAction SilentlyContinue
} else {
    Write-Warning "CUDA_PATH not set; copy cudart64_*.dll / cublas64_*.dll / cublasLt64_*.dll from your CUDA install into $OutDir yourself."
}

Write-Host "==> Building the launcher"
Push-Location $repoRoot
$env:GOOS = "windows"
$env:GOARCH = "amd64"
go build -o "$OutDir\q38fninference.exe" .\cmd\q38fninference
Remove-Item Env:\GOOS, Env:\GOARCH
Pop-Location

Write-Host ""
Write-Host "Done. Binaries are in $OutDir"
Write-Host "Run $OutDir\q38fninference.exe to start the launcher."
