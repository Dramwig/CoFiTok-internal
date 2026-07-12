param(
  [Parameter(Mandatory=$true)][string]$ManifestJson,
  [Parameter(Mandatory=$true)][string]$OutputRawDir,
  [int]$Retry = 20
)

$ErrorActionPreference = 'Stop'
$manifest = Get-Content -Raw -Encoding UTF8 $ManifestJson | ConvertFrom-Json
New-Item -ItemType Directory -Force $OutputRawDir | Out-Null

$total = $manifest.files.Count
$index = 0
foreach ($item in $manifest.files) {
  $index += 1
  $relative = [string]$item.path
  $url = [string]$item.url
  $out = Join-Path $OutputRawDir ($relative -replace '/', '\')
  $parent = Split-Path -Parent $out
  New-Item -ItemType Directory -Force $parent | Out-Null
  Write-Host "[$index/$total] $relative"
  & curl.exe -L -C - --retry $Retry --retry-delay 5 --retry-all-errors --connect-timeout 20 -o $out $url
  if ($LASTEXITCODE -ne 0) {
    throw "curl failed for $relative with exit code $LASTEXITCODE"
  }
  if ($null -ne $item.size) {
    $size = (Get-Item -LiteralPath $out).Length
    if ($size -ne [int64]$item.size) {
      throw "size mismatch for $relative expected=$($item.size) actual=$size"
    }
  }
}

Write-Host "DOWNLOAD_OK"

