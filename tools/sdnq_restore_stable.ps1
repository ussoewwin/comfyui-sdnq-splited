param(
  [Parameter(Mandatory=$true)]
  [string]$SamplerStable,
  [Parameter(Mandatory=$true)]
  [string]$VaeEncodeStable
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$samplerTarget = Join-Path $root "nodes\\samplerv2.py"
$vaeTarget = Join-Path $root "nodes\\vae_encode.py"

Write-Host "Restoring..." -ForegroundColor Cyan
Write-Host "  Sampler: $SamplerStable -> $samplerTarget"
Write-Host "  VAEEnc : $VaeEncodeStable -> $vaeTarget"

Copy-Item -Force $SamplerStable $samplerTarget
Copy-Item -Force $VaeEncodeStable $vaeTarget

Write-Host "Done. Restart ComfyUI." -ForegroundColor Green

