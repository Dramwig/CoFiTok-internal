[CmdletBinding()]
param(
    [string]$HostAlias = "pro6000",
    [string]$ExpectedRemoteCommit = "781a01444fddbf0d48a427ba58bdeed50167b5be"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Program,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Program $($Arguments -join ' ')"
    }
}

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepositoryRoot
try {
    $Branch = (& git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $Branch -ne "scale/generative-system") {
        throw "Deployment requires the scale/generative-system branch; found '$Branch'."
    }

    $TrackedChanges = (& git status --porcelain --untracked-files=no) -join "`n"
    if ($LASTEXITCODE -ne 0 -or $TrackedChanges) {
        throw "Deployment requires a clean tracked local worktree."
    }

    & git merge-base --is-ancestor $ExpectedRemoteCommit HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Expected remote commit is not an ancestor of local HEAD."
    }
    $TargetCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to resolve local HEAD."
    }

    $Bundle = Join-Path $env:TEMP "cofitok-generation-upgrade-$($TargetCommit.Substring(0, 12)).bundle"
    $RemoteBundle = "/tmp/cofitok-generation-upgrade.bundle"
    $RemoteHelper = "/tmp/deploy_generation_posttraining_pipeline_remote.sh"
    $Helper = Join-Path $RepositoryRoot "artifacts/runbooks/deploy_generation_posttraining_pipeline_remote.sh"

    try {
        if (Test-Path -LiteralPath $Bundle) {
            Remove-Item -LiteralPath $Bundle -Force
        }
        # The remote already owns the pinned training commit, so transfer only
        # the verified fast-forward range needed for the generation upgrade.
        Invoke-CheckedCommand git @(
            "bundle",
            "create",
            $Bundle,
            "HEAD",
            "^$ExpectedRemoteCommit"
        )
        Invoke-CheckedCommand git @("bundle", "verify", $Bundle)
        $BundleHeads = @(& git bundle list-heads $Bundle)
        if ($LASTEXITCODE -ne 0 -or $BundleHeads.Count -ne 1) {
            throw "Upgrade bundle must advertise exactly one head."
        }
        $BundleHead = ($BundleHeads[0] -split "\s+", 2)[0]
        if ($BundleHead -ne $TargetCommit) {
            throw "Upgrade bundle advertises '$BundleHead' instead of '$TargetCommit'."
        }
        Invoke-CheckedCommand scp @("-O", $Bundle, "${HostAlias}:${RemoteBundle}")
        Invoke-CheckedCommand scp @("-O", $Helper, "${HostAlias}:${RemoteHelper}")
        Invoke-CheckedCommand ssh @(
            $HostAlias,
            "bash",
            $RemoteHelper,
            $RemoteBundle,
            $ExpectedRemoteCommit,
            $TargetCommit
        )
    }
    finally {
        if (Test-Path -LiteralPath $Bundle) {
            Remove-Item -LiteralPath $Bundle -Force
        }
    }
}
finally {
    Pop-Location
}
