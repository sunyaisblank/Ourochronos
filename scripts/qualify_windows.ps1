#Requires -Version 5.1
<#
.SYNOPSIS
Qualify an exact, already built native Windows x86-64 Ourochronos runtime.
.DESCRIPTION
This finite synthetic campaign builds no compiler or SDK and installs nothing.
It uses bounded asynchronous byte-stream capture, a 10-second process limit and
a 60-second campaign limit. Sources never request native host operations except
the explicit zero-duration SLEEP case that must be denied on Windows.

Default fixtures are deleted. -Artifacts must name a new absolute directory;
explicit retained fixtures belong to the caller, including on failure. Optional
-PeerArtifacts names a directory containing Linux-produced flip.ourobc,
count.ourobc, affine.ourobc and linux.ourofp. Peer inputs are read, never changed.
.EXAMPLE
powershell.exe -NoProfile -NonInteractive -File scripts\qualify_windows.ps1 `
  -Runtime C:\candidate\runtime\ourochronos.exe -Report C:\scratch\windows.json
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Runtime,
    [string]$PeerArtifacts,
    [string]$Report,
    [string]$Artifacts
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$script:CampaignClock = [System.Diagnostics.Stopwatch]::StartNew()
$script:Cases = New-Object 'System.Collections.Generic.List[object]'
$script:CurrentCase = 'input/runtime identity'
$script:LastProcess = $null
$script:Work = $null
$script:RuntimePath = $null
$script:Identity = $null
$script:NativeIdentity = $null
$script:PeerIdentities = @{}
$script:Version = $null
$script:VersionOutput = $null
$script:Discovery = $null
$script:ReportPath = $null
$script:RetainArtifacts = -not [string]::IsNullOrEmpty($Artifacts)
$script:Utf8 = New-Object System.Text.UTF8Encoding($false, $true)
$script:MaxOutput = 1048576
$script:MaxRuntime = 134217728
$script:MaxNative = 67108864
$script:MaxSmallArtifact = 1048576
$script:MaxArtifactTotal = 268435456

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Absolute-Path([string]$Path) {
    Assert-True ($Path -match '^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\[^\\]+[\\/])') 'An absolute Windows drive or UNC path is required.'
    return [System.IO.Path]::GetFullPath($Path)
}

function File-Identity([string]$Path, [long]$Limit) {
    $item = Get-Item -LiteralPath $Path -Force
    Assert-True (-not $item.PSIsContainer) "Expected a regular file: $Path"
    Assert-True (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0) "Reparse-point input is outside this gate: $Path"
    Assert-True ($item.Length -gt 0 -and $item.Length -le $Limit) "File outside byte bound: $Path"
    $digest = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-True ((Get-Item -LiteralPath $Path).Length -eq $item.Length) "File changed while hashing: $Path"
    return [pscustomobject]@{ path = $item.FullName; bytes = [long]$item.Length; sha256 = $digest }
}

function Same-Identity($Before, $After) {
    return $Before.path -eq $After.path -and $Before.bytes -eq $After.bytes -and $Before.sha256 -eq $After.sha256
}

function Parse-Version([string]$Text) {
    $lines = $Text.Replace("`r`n", "`n").TrimEnd([char[]]@([char]10, [char]13)).Split([char]10)
    $first = [regex]::Match($lines[0], '^ourochronos ([0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.+-]+)?)$')
    Assert-True ($first.Success -and $lines[0].Length -lt 128) 'Version first line exceeded expected semver shape.'
    if ($lines.Count -eq 1) {
        Assert-True ($first.Groups[1].Value -eq '0.2.0') 'Current runtime omits platform/features discovery.'
        return [pscustomobject]@{ version = $lines[0]; semver = $first.Groups[1].Value; discovery = $null }
    }
    Assert-True ($lines.Count -eq 3) 'Unexpected version discovery line count.'
    Assert-True ($lines[1] -eq 'platform: x86_64-windows; runtime ABI 1; native effect commits: Linux only') 'Runtime platform/ABI disagrees with native x86-64 Windows gate.'
    $features = [regex]::Match($lines[2], '^compiled features: lsp=(true|false) dynamic-ffi=(true|false); Z3 native library required$')
    Assert-True $features.Success 'Missing or malformed compiled feature discovery.'
    return [pscustomobject]@{
        version = $lines[0]; semver = $first.Groups[1].Value
        discovery = @{ platform = 'x86_64-windows'; runtime_abi = 1; compiled_features = @{ lsp = $features.Groups[1].Value -eq 'true'; 'dynamic-ffi' = $features.Groups[2].Value -eq 'true' } }
    }
}

function Assert-X64PE([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $reader = New-Object System.IO.BinaryReader($stream)
    try {
        Assert-True ($stream.Length -ge 90) 'Truncated PE input.'
        Assert-True ($reader.ReadUInt16() -eq 0x5A4D) 'Runtime must have an MZ header.'
        $stream.Position = 0x3C
        $offset = $reader.ReadUInt32()
        Assert-True ($offset -le $stream.Length - 26) 'PE header offset is outside input.'
        $stream.Position = $offset
        Assert-True ($reader.ReadUInt32() -eq 0x00004550) 'Runtime must have a PE signature.'
        Assert-True ($reader.ReadUInt16() -eq 0x8664) 'Gate supports x86-64 PE only.'
        $stream.Position = $offset + 24
        Assert-True ($reader.ReadUInt16() -eq 0x20B) 'Gate supports PE32+ only.'
    }
    finally { $reader.Dispose() }
}

function Quote-Argument([string]$Value) {
    Assert-True ($Value.Length -le 2048 -and $Value.IndexOf([char]0) -lt 0) 'Argument outside fixed input bound.'
    # Windows CRT quoting: double backslashes before a quote and at the end.
    $builder = New-Object System.Text.StringBuilder
    $null = $builder.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq [char]92) { $slashes++; continue }
        if ($character -eq [char]34) {
            $null = $builder.Append(('\' * (2 * $slashes + 1)))
        }
        elseif ($slashes -gt 0) { $null = $builder.Append(('\' * $slashes)) }
        $null = $builder.Append($character)
        $slashes = 0
    }
    $null = $builder.Append(('\' * (2 * $slashes)))
    $null = $builder.Append('"')
    return $builder.ToString()
}

function Run-Bounded([string]$Executable, [string[]]$Arguments) {
    Assert-True ($Arguments.Count -le 32) 'Process argument count exceeded.'
    $quoted = @($Arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    Assert-True ($quoted.Length -le 8192) 'Process input exceeded fixed argument bound.'
    $remaining = 60.0 - $script:CampaignClock.Elapsed.TotalSeconds
    Assert-True ($remaining -gt 0) 'Campaign wall time cap exceeded.'
    $limit = [Math]::Min(10.0, $remaining)
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $info.Arguments = $quoted
    $info.WorkingDirectory = $script:Work
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    $output = New-Object System.IO.MemoryStream
    $errors = New-Object System.IO.MemoryStream
    $buffers = @((New-Object byte[] 8192), (New-Object byte[] 8192))
    $streams = @($output, $errors)
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    $started = $false
    $script:LastProcess = [pscustomobject]@{
        executable = $Executable; arguments = $Arguments; pid = $null; exit = $null
        stdout_bytes = 0; stderr_bytes = 0; seconds = 0; stdout = ''; stderr = ''
    }
    try {
        $started = $process.Start()
        Assert-True $started 'Native process failed to start.'
        $script:LastProcess.pid = $process.Id
        $process.StandardInput.Close()
        # ReadAsync on raw byte streams avoids ReadToEnd deadlocks, unbounded
        # line buffering and PowerShell callback/runspace dependencies.
        $pipes = @($process.StandardOutput.BaseStream, $process.StandardError.BaseStream)
        $tasks = @($pipes[0].ReadAsync($buffers[0], 0, 8192), $pipes[1].ReadAsync($buffers[1], 0, 8192))
        $ended = @($false, $false)
        while ($true) {
            Assert-True ($clock.Elapsed.TotalSeconds -lt $limit) 'Native process wall time cap exceeded.'
            for ($index = 0; $index -lt 2; $index++) {
                if (-not $ended[$index] -and $tasks[$index].IsCompleted) {
                    $count = $tasks[$index].GetAwaiter().GetResult()
                    if ($count -eq 0) { $ended[$index] = $true }
                    else {
                        Assert-True (($output.Length + $errors.Length + $count) -le $script:MaxOutput) 'Combined stdout/stderr cap exceeded.'
                        $streams[$index].Write($buffers[$index], 0, $count)
                        $tasks[$index] = $pipes[$index].ReadAsync($buffers[$index], 0, 8192)
                    }
                }
            }
            if ($ended[0] -and $ended[1] -and $process.HasExited) { break }
            [System.Threading.Thread]::Sleep(5)
        }
        $script:LastProcess.exit = $process.ExitCode
        $script:LastProcess.stdout = $script:Utf8.GetString($output.ToArray())
        $script:LastProcess.stderr = $script:Utf8.GetString($errors.ToArray())
        return $script:LastProcess
    }
    finally {
        if ($started) {
            if (-not $process.HasExited) { $process.Kill() }
            Assert-True ($process.WaitForExit(1000)) 'Native child did not stop during bounded cleanup.'
            $script:LastProcess.exit = $process.ExitCode
        }
        $script:LastProcess.seconds = [Math]::Round($clock.Elapsed.TotalSeconds, 3)
        $script:LastProcess.stdout_bytes = $output.Length
        $script:LastProcess.stderr_bytes = $errors.Length
        # Failure reports retain at most 1024 decoded characters per stream.
        if (-not $script:LastProcess.stdout) {
            $script:LastProcess.stdout = [System.Text.Encoding]::UTF8.GetString($output.ToArray())
        }
        if (-not $script:LastProcess.stderr) {
            $script:LastProcess.stderr = [System.Text.Encoding]::UTF8.GetString($errors.ToArray())
        }
        if ($started) {
            $process.StandardInput.Close()
            $process.StandardOutput.Close()
            $process.StandardError.Close()
        }
        $process.Dispose()
        $output.Dispose()
        $errors.Dispose()
    }
}

function Check-Run([string]$Name, [string[]]$Arguments, [int]$ExpectedExit, [string[]]$ExpectedText,
                   [string]$Executable = $script:RuntimePath, [string]$ExactStdout = '') {
    $script:CurrentCase = $Name
    $result = Run-Bounded $Executable $Arguments
    Assert-True ($result.exit -eq $ExpectedExit) "$Name returned exit $($result.exit), expected $ExpectedExit."
    $combined = $result.stdout + $result.stderr
    foreach ($text in $ExpectedText) {
        Assert-True ($combined.Contains($text)) "$Name omitted expected text: $text"
    }
    if (-not [string]::IsNullOrEmpty($ExactStdout)) {
        Assert-True ($result.stdout.Replace("`r`n", "`n") -eq $ExactStdout) "$Name produced unexpected or additional stdout."
    }
    $script:Cases.Add([pscustomobject]@{
        name = $Name; executable = $Executable; arguments = $Arguments; pid = $result.pid
        exit = $result.exit; seconds = $result.seconds
        stdout_bytes = $result.stdout_bytes; stderr_bytes = $result.stderr_bytes
        output_excerpt = $combined.Substring(0, [Math]::Min(1024, $combined.Length)).Trim()
    })
    return $result
}

function Fixture([string]$Name, [string]$Text) {
    Assert-True ($script:Utf8.GetByteCount($Text) -le 4096) 'Fixture source exceeded fixed input bound.'
    $path = Join-Path $script:Work $Name
    [System.IO.File]::WriteAllText($path, $Text, $script:Utf8)
    return $path
}

function Artifact-Identity([string]$Name, [long]$Limit = $script:MaxSmallArtifact) {
    return File-Identity (Join-Path $script:Work $Name) $Limit
}

function Peer-Path([string]$Name) { return Join-Path $script:PeerPath $Name }

try {
    $script:RuntimePath = Absolute-Path $Runtime
    $script:Identity = File-Identity $script:RuntimePath $script:MaxRuntime
    Assert-X64PE $script:RuntimePath
    $native = Join-Path ([System.IO.Path]::GetDirectoryName($script:RuntimePath)) 'libz3.dll'
    $script:NativeIdentity = File-Identity $native $script:MaxNative
    Assert-X64PE $native
    if ($script:RetainArtifacts) {
        $script:Work = Absolute-Path $Artifacts
        Assert-True (-not (Test-Path -LiteralPath $script:Work)) '-Artifacts must be a new synthetic directory.'
    }
    else {
        $script:Work = Join-Path ([System.IO.Path]::GetTempPath()) ('ouro windows qualification ' + [guid]::NewGuid().ToString('N'))
    }
    $null = [System.IO.Directory]::CreateDirectory($script:Work)
    # The standalone launcher must use the exact solver adjacent to the selected
    # runtime. The DLL copy is a synthetic fixture, not a system installation.
    Copy-Item -LiteralPath $native -Destination (Join-Path $script:Work 'libz3.dll')
    Assert-True ((Artifact-Identity 'libz3.dll' $script:MaxNative).sha256 -eq $script:NativeIdentity.sha256) 'Solver fixture copy changed bytes.'
    if (-not [string]::IsNullOrEmpty($PeerArtifacts)) {
        $script:PeerPath = Absolute-Path $PeerArtifacts
        foreach ($name in @('flip.ourobc', 'count.ourobc', 'affine.ourobc', 'linux.ourofp')) {
            $script:PeerIdentities[$name] = File-Identity (Peer-Path $name) $script:MaxSmallArtifact
        }
    }
    if (-not [string]::IsNullOrEmpty($Report)) {
        $proposedReport = [System.IO.Path]::GetFullPath($Report)
        $protectedPaths = @($script:RuntimePath, $native, $PSCommandPath)
        foreach ($peerIdentity in $script:PeerIdentities.Values) { $protectedPaths += $peerIdentity.path }
        foreach ($name in @('libz3.dll', 'wrap.ouro', 'procedure.ouro', 'quote.ouro', 'dependency.ouro',
                            'module.ouro', 'flip.ouro', 'count.ouro', 'affine.ouro', 'flip.ourobc',
                            'count.ourobc', 'affine.ourobc', 'windows.ourofp', 'windows.ourocp',
                            'peer.ourocp', 'failure-library.ouro', 'failure.ouro', 'failure.ouropkg',
                            'sleep.ouro', 'windows.ouropkg', 'launcher.exe')) {
            $protectedPaths += Join-Path $script:Work $name
        }
        Assert-True ($protectedPaths -notcontains $proposedReport) '-Report collides with a runtime, peer input, gate or synthetic artifact.'
        # Assign only after admission so even a failure report cannot overwrite
        # an executable or a retained certificate after its final hash check.
        $script:ReportPath = $proposedReport
    }

    $result = Check-Run 'version' @('--version') 0 @('ourochronos ')
    $versionIdentity = Parse-Version $result.stdout
    $script:Version = $versionIdentity.version
    $script:Discovery = $versionIdentity.discovery
    $script:VersionOutput = $result.stdout.Replace("`r`n", "`n")
    $null = Check-Run 'help' @('--help') 0 @('Usage:', 'run-package', '--build-executable')
    $wrap = Fixture 'wrap.ouro' "18446744073709551615 1 ADD OUTPUT`n"
    $procedure = Fixture 'procedure.ouro' "PROCEDURE answer { 41 1 ADD }`nanswer OUTPUT`n"
    $quote = Fixture 'quote.ouro' "40 [ 2 ADD ] EXEC OUTPUT`n"
    $library = Fixture 'dependency.ouro' "PROCEDURE imported { 6 7 MUL }`n"
    $module = Fixture 'module.ouro' "IMPORT `"dependency.ouro`"`nimported OUTPUT`n"
    $null = Check-Run 'wrapping words' @($wrap, '--memory-cells', '4', '--max-inst', '1000') 0 @('[0]') -ExactStdout "[0]`n"
    $null = Check-Run 'procedure calls' @($procedure, '--memory-cells', '4', '--max-inst', '1000') 0 @('[42]') -ExactStdout "[42]`n"
    $null = Check-Run 'quotation EXEC' @($quote, '--memory-cells', '4', '--max-inst', '1000') 0 @('[42]') -ExactStdout "[42]`n"
    $null = Check-Run 'importer-relative module' @($module, '--memory-cells', '4', '--max-inst', '1000') 0 @('[42]') -ExactStdout "[42]`n"

    $flip = Fixture 'flip.ouro' "TEMPORAL 0 1 BITS 1 { 0 ORACLE 1 XOR 0 PROPHECY }`n"
    $count = Fixture 'count.ouro' "5 WHILE { DUP } { 1 SUB } POP 42 OUTPUT`n"
    $affine = Fixture 'affine.ouro' "TEMPORAL 0 3 BITS 1 { 1 ORACLE 2 ORACLE XOR 0 PROPHECY 2 ORACLE 1 PROPHECY 0 2 PROPHECY }`n"
    foreach ($row in @(@('flip', $flip, '1'), @('count', $count, '1'), @('affine', $affine, '4'))) {
        $path = Join-Path $script:Work ($row[0] + '.ourobc')
        $null = Check-Run ($row[0] + ' bytecode build') @($row[1], '--emit-bytecode', $path, '--memory-cells', $row[2], '--max-inst', '1000') 0 @('Wrote linked bytecode')
        $null = Artifact-Identity ($row[0] + '.ourobc')
    }
    $flipBC = Join-Path $script:Work 'flip.ourobc'
    $countBC = Join-Path $script:Work 'count.ourobc'
    $affineBC = Join-Path $script:Work 'affine.ourobc'
    $proof = Join-Path $script:Work 'windows.ourofp'
    $null = Check-Run 'own finite proof' @('prove-finite', $flipBC, $proof, '1', '64') 0 @('CHECKED NO POINT FIXED STATE: 2 states')
    $proofIdentity = Artifact-Identity 'windows.ourofp'
    $null = Check-Run 'own finite proof reload' @('check-finite', $flipBC, $proof, '1', '64') 0 @('CHECKED NO POINT FIXED STATE: 2 states')
    $null = Check-Run 'finite stale gas rejected' @('check-finite', $flipBC, $proof, '1', '63') 1 @('resource configuration')
    Assert-True ((Artifact-Identity 'windows.ourofp').sha256 -eq $proofIdentity.sha256) 'Rejected proof check altered the certificate.'

    $checkpoint = Join-Path $script:Work 'windows.ourocp'
    $null = Check-Run 'checkpoint initial slice' @('halt-slice', $countBC, $checkpoint, '4', '1', '64') 3 @('UNKNOWN after 4 fetched instructions')
    $checkpointIdentity = Artifact-Identity 'windows.ourocp'
    $null = Check-Run 'checkpoint ceiling reset rejected' @('halt-resume', $countBC, $checkpoint, '64', '1', '65') 1 @('cumulative resource policy changed')
    Assert-True ((Artifact-Identity 'windows.ourocp').sha256 -eq $checkpointIdentity.sha256) 'Rejected ceiling reset altered the checkpoint.'
    $null = Check-Run 'checkpoint real continuation' @('halt-resume', $countBC, $checkpoint, '64', '1', '64') 0 @('HALTED after 32 fetched instructions', '[42]') -ExactStdout "HALTED after 32 fetched instructions.`n[42]`n"
    $null = Check-Run 'affine all-class recurrence' @('analyse-affine', $affineBC, '1', '4', '64') 0 @('UNIFORM PARITY READOUT: 0', 'all 1 recurrent classes')

    $failureLibrary = Fixture 'failure-library.ouro' "PROCEDURE fail { 0 99 INDEX POP }`n"
    $failure = Fixture 'failure.ouro' "IMPORT `"failure-library.ouro`"`nfail`n"
    $failurePackage = Join-Path $script:Work 'failure.ouropkg'
    $null = Check-Run 'source-aware failing package build' @($failure, '--build', $failurePackage, '--memory-cells', '4', '--max-inst', '1000', '--strict') 0 @('Wrote portable package')
    $null = Artifact-Identity 'failure.ouropkg'
    Remove-Item -LiteralPath $failureLibrary, $failure
    $null = Check-Run 'deleted-source packaged diagnostic' @('run-package', $failurePackage) 1 @('failure-library.ouro', 'bytes 22..27', 'outside 4 cells')
    $sleep = Fixture 'sleep.ouro' "0 SLEEP 42 OUTPUT`n"
    $null = Check-Run 'Linux-only native effect denied' @($sleep, '--allow-sleep-ms', '0', '--memory-cells', '1', '--max-inst', '1000') 1 @('native effect commits are supported on Linux only')
    $null = Check-Run 'Linux-only isolation denied' @('isolate', '2000', '256', $wrap) 1 @('process resource supervision requires Linux')

    $package = Join-Path $script:Work 'windows.ouropkg'
    $launcher = Join-Path $script:Work 'launcher.exe'
    $null = Check-Run 'Windows package build' @($wrap, '--build', $package, '--memory-cells', '1', '--max-inst', '1000') 0 @('Wrote portable package')
    $null = Artifact-Identity 'windows.ouropkg'
    $null = Check-Run 'Windows package reload' @('run-package', $package) 0 @('[0]') -ExactStdout "[0]`n"
    $null = Check-Run 'Windows standalone launcher build' @($wrap, '--build-executable', $launcher, '--memory-cells', '1', '--max-inst', '1000') 0 @('Wrote native executable')
    $null = Artifact-Identity 'launcher.exe' $script:MaxRuntime
    Assert-X64PE $launcher
    $null = Check-Run 'Windows standalone launcher execution' @() 0 @('[0]') $launcher -ExactStdout "[0]`n"

    if ($script:PeerIdentities.Count -gt 0) {
        $null = Check-Run 'Linux exact-byte finite proof checked on Windows' @('check-finite', (Peer-Path 'flip.ourobc'), (Peer-Path 'linux.ourofp'), '1', '64') 0 @('CHECKED NO POINT FIXED STATE: 2 states')
        $null = Check-Run 'Linux proof stale gas rejected on Windows' @('check-finite', (Peer-Path 'flip.ourobc'), (Peer-Path 'linux.ourofp'), '1', '63') 1 @('resource configuration')
        $peerCheckpoint = Join-Path $script:Work 'peer.ourocp'
        $null = Check-Run 'Linux bytecode checkpoint slice on Windows' @('halt-slice', (Peer-Path 'count.ourobc'), $peerCheckpoint, '4', '1', '64') 3 @('UNKNOWN after 4 fetched instructions')
        $null = Artifact-Identity 'peer.ourocp'
        $null = Check-Run 'Linux bytecode checkpoint continuation on Windows' @('halt-resume', (Peer-Path 'count.ourobc'), $peerCheckpoint, '64', '1', '64') 0 @('HALTED after 32 fetched instructions', '[42]') -ExactStdout "HALTED after 32 fetched instructions.`n[42]`n"
        $null = Check-Run 'Linux affine bytecode checked on Windows' @('analyse-affine', (Peer-Path 'affine.ourobc'), '1', '4', '64') 0 @('UNIFORM PARITY READOUT: 0')
        foreach ($name in $script:PeerIdentities.Keys) {
            Assert-True (Same-Identity $script:PeerIdentities[$name] (File-Identity (Peer-Path $name) $script:MaxSmallArtifact)) "Peer bytes changed during qualification: $name"
        }
    }

    $script:CurrentCase = 'final runtime, solver and fixture identity'
    $null = Check-Run 'final version, platform and feature identity' @('--version') 0 @($script:Version) -ExactStdout $script:VersionOutput
    $script:CurrentCase = 'final runtime, solver and fixture identity'
    Assert-True (Same-Identity $script:Identity (File-Identity $script:RuntimePath $script:MaxRuntime)) 'Runtime bytes changed during qualification.'
    Assert-True (Same-Identity $script:NativeIdentity (File-Identity $native $script:MaxNative)) 'Native solver bytes changed during qualification.'
    $fixtureFiles = @(Get-ChildItem -LiteralPath $script:Work -File)
    Assert-True ($fixtureFiles.Count -le 64) 'Synthetic fixture count exceeded.'
    $total = [long]0
    $fixtureIdentities = @()
    foreach ($file in $fixtureFiles) {
        $total += $file.Length
        Assert-True ($total -le $script:MaxArtifactTotal) 'Synthetic artifact bytes exceeded total cap.'
        $fixtureIdentities += File-Identity $file.FullName $script:MaxRuntime
    }
    Assert-True ($script:CampaignClock.Elapsed.TotalSeconds -lt 60) 'Campaign wall time cap exceeded.'
    $resultReport = [ordered]@{
        schema = 'ourochronos.windows-qualification/1'; status = 'passed'
        runtime = $script:Identity; version = $script:Version; discovery = $script:Discovery; native_solver = $script:NativeIdentity
        environment = @{ powershell = $PSVersionTable.PSVersion.ToString(); clr = [Environment]::Version.ToString(); os = [Environment]::OSVersion.VersionString }
        cases = @($script:Cases.ToArray()); case_count = $script:Cases.Count
        seconds = [Math]::Round($script:CampaignClock.Elapsed.TotalSeconds, 3)
        peer_inputs = $script:PeerIdentities; artifacts = $fixtureIdentities
        artifacts_directory = $script:Work; artifacts_retained = $script:RetainArtifacts
        cleanup_owner = $(if ($script:RetainArtifacts) { 'caller' } else { 'gate' })
        bounds = @{ process_seconds = 10; campaign_seconds = 60; cleanup_grace_seconds = 1; process_arguments = 32; argument_characters = 8192; combined_process_output_bytes = $script:MaxOutput; small_artifact_bytes = $script:MaxSmallArtifact; total_fixture_bytes = $script:MaxArtifactTotal }
        scope = 'finite native Windows x86-64 portable campaign; explicit unsupported Linux-only profiles; no native effects, install or publication'
    }
    $exitCode = 0
}
catch {
    $last = $null
    if ($null -ne $script:LastProcess) {
        $last = [ordered]@{
            executable = $script:LastProcess.executable; arguments = $script:LastProcess.arguments; pid = $script:LastProcess.pid
            exit = $script:LastProcess.exit; seconds = $script:LastProcess.seconds
            stdout_bytes = $script:LastProcess.stdout_bytes; stderr_bytes = $script:LastProcess.stderr_bytes
            stdout_excerpt = $script:LastProcess.stdout.Substring(0, [Math]::Min(1024, $script:LastProcess.stdout.Length))
            stderr_excerpt = $script:LastProcess.stderr.Substring(0, [Math]::Min(1024, $script:LastProcess.stderr.Length))
        }
    }
    $message = $_.Exception.Message
    $resultReport = [ordered]@{
        schema = 'ourochronos.windows-qualification/1'; status = 'failed'; runtime = $script:Identity
        case = $script:CurrentCase; error = $message.Substring(0, [Math]::Min(2048, $message.Length))
        passed_cases = @($script:Cases.ToArray()); last_process = $last
        seconds = [Math]::Round($script:CampaignClock.Elapsed.TotalSeconds, 3)
        artifacts_directory = $script:Work; artifacts_retained = $script:RetainArtifacts
        cleanup_owner = $(if ($script:RetainArtifacts) { 'caller' } else { 'gate' })
    }
    $exitCode = 1
}
finally {
    if ($null -ne $script:Work -and -not $script:RetainArtifacts -and (Test-Path -LiteralPath $script:Work)) {
        Remove-Item -LiteralPath $script:Work -Recurse -Force
    }
}

$json = $resultReport | ConvertTo-Json -Depth 10
if ($null -ne $script:ReportPath) {
    [System.IO.File]::WriteAllText($script:ReportPath, $json + [Environment]::NewLine, $script:Utf8)
}
[Console]::WriteLine($json)
exit $exitCode
