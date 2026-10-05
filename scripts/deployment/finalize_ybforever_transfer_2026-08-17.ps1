$ErrorActionPreference = 'Stop'

$markerDir = '/tmp/cofitok_generation_archive_parallel_20260817'
$sourceKey = '/root/.ssh/cofitok_ybforever_push_20260817'
$sourceScripts = @(
    '/tmp/cofitok_generation_archive_rsync_20260817.sh',
    '/tmp/cofitok_generation_archive_parallel_20260817.sh',
    '/tmp/cofitok_generation_archive_recovery_supervisor_20260817.sh',
    '/tmp/cofitok_legacy_dataset_fallback_after_generation_20260817.sh',
    '/tmp/cofitok_legacy_dataset_fallback_independent_20260817.sh',
    '/tmp/cofitok_legacy_checkpoint_runs_after_generation_20260817.sh',
    '/tmp/cofitok_rsync_optional_dataset_metadata_20260817.sh'
)
$localTunnelPattern = '-R 127\.0\.0\.1:22179:100\.88\.240\.40:2222 pro6000'
$localTemporaryFiles = @(
    (Join-Path $env:TEMP 'cofitok_generation_archive_rsync_20260817.sh'),
    (Join-Path $env:TEMP 'cofitok_legacy_checkpoint_runs_after_generation_20260817.sh'),
    (Join-Path $env:TEMP 'cofitok_fix_authorized_keys_20260817.py')
)
$logPath = Join-Path $env:TEMP 'cofitok_ybforever_transfer_finalizer_20260817.log'

function Write-FinalizerLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append
}

Write-FinalizerLog 'waiting for generation archive, legacy datasets, and historical run verification markers'
while ($true) {
    & ssh pro6000 "test -f $markerDir/completed -a -f $markerDir/legacy_dataset_fallback.completed -a -f $markerDir/legacy_checkpoint_runs.completed"
    if ($LASTEXITCODE -eq 0) {
        break
    }
    Start-Sleep -Seconds 300
}

$targetCleanup = @'
set -euo pipefail
key_file=/home/yubohuang/.ssh/authorized_keys
marker=cofitok-pro6000-push-ybforever-20260817
matches=$(grep -F -c "$marker" "$key_file" || true)
test "$matches" -eq 1
temporary_file=$(mktemp "${key_file}.cofitok.XXXXXX")
awk -v marker="$marker" 'index($0, marker) == 0 { print }' "$key_file" > "$temporary_file"
chmod 600 "$temporary_file"
mv "$temporary_file" "$key_file"
test "$(grep -F -c "$marker" "$key_file" || true)" -eq 0
'@

while ($true) {
    $targetCleanup | & ssh ybforever 'bash -s'
    if ($LASTEXITCODE -eq 0) {
        break
    }
    Write-FinalizerLog 'target temporary authorization cleanup failed; retrying in 300 seconds'
    Start-Sleep -Seconds 300
}

$sourceTargets = @($sourceKey, "$sourceKey.pub") + $sourceScripts
$sourceDeleteCommand = 'rm -f ' + (($sourceTargets | ForEach-Object { "'$_'" }) -join ' ')
& ssh pro6000 $sourceDeleteCommand
if ($LASTEXITCODE -ne 0) {
    throw 'Source temporary key cleanup failed.'
}

$tunnelProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'ssh.exe' -and $_.CommandLine -match $localTunnelPattern
})
if ($tunnelProcesses.Count -ne 1) {
    throw "Expected exactly one temporary reverse tunnel, found $($tunnelProcesses.Count)."
}
Stop-Process -Id $tunnelProcesses[0].ProcessId -ErrorAction Stop

foreach ($temporaryFile in $localTemporaryFiles) {
    if (Test-Path -LiteralPath $temporaryFile) {
        Remove-Item -LiteralPath $temporaryFile -Force
    }
}

Write-FinalizerLog 'completed temporary credential and tunnel cleanup'
