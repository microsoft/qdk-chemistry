<#
.SYNOPSIS
    Report Windows memory limits and enlarge the page file if it is too small.

.DESCRIPTION
    Compiling our libint2 translation unit instantiates enough templates that
    cl.exe runs out of heap on small agents:

        libint2\./engine.impl.h(643): fatal error C1060: compiler is out of heap space

    The Windows ARM64 CI agent has only 7 GB of RAM, and reducing the build to a
    single compiler process was not enough on its own -- what actually bounds a
    single cl.exe is the system commit limit, which is physical memory plus the
    page file, shared with the agent, Defender and the CodeQL tooling.

    Raising the page file raises the commit limit, which is what gives one
    compiler process room to finish. This matters most where physical memory is
    smallest, so the script is a no-op when the commit limit is already
    comfortable.

    Everything here is best-effort: memory limits are an environment detail, not
    a build input, so a failure to adjust them warns rather than fails the build.
    The reported numbers are the useful part either way -- they turn "out of heap
    space" into an observation about how much commit was actually available.

.PARAMETER TargetCommitGB
    Commit limit to aim for. The default leaves a single cl.exe room for the
    heaviest translation unit alongside everything else on the agent.
#>
[CmdletBinding()]
param(
    [int]$TargetCommitGB = 24
)

$ErrorActionPreference = 'Stop'

function Get-CommitGB {
    $os = Get-CimInstance Win32_OperatingSystem
    [pscustomobject]@{
        # TotalVirtualMemorySize is the commit limit: RAM plus page file.
        LimitGB = [math]::Round($os.TotalVirtualMemorySize / 1MB, 1)
        FreeGB  = [math]::Round($os.FreeVirtualMemory     / 1MB, 1)
        RamGB   = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    }
}

$before = Get-CommitGB
Write-Host "Memory before:"
Write-Host "  physical RAM  : $($before.RamGB) GB"
Write-Host "  commit limit  : $($before.LimitGB) GB"
Write-Host "  commit free   : $($before.FreeGB) GB"

Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  page file     : $($_.Name) initial=$($_.InitialSize)MB maximum=$($_.MaximumSize)MB"
}
Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | ForEach-Object {
    Write-Host ("  drive {0} free : {1} GB" -f $_.DeviceID, [math]::Round($_.FreeSpace / 1GB, 1))
}

if ($before.LimitGB -ge $TargetCommitGB) {
    Write-Host "Commit limit is already at least $TargetCommitGB GB; leaving it alone."
    return
}

try {
    # Pick the fixed drive with the most free space. On Azure agents that is
    # usually a large scratch volume rather than the smaller OS disk.
    $drive = Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |
             Sort-Object FreeSpace -Descending | Select-Object -First 1
    $freeGB = [math]::Round($drive.FreeSpace / 1GB, 1)
    # Leave the volume room for the build itself as well as the page file.
    $maxMB  = [math]::Min($TargetCommitGB * 1024, [int](($freeGB - 20) * 1024))
    if ($maxMB -lt 4096) {
        Write-Warning "Only $freeGB GB free on $($drive.DeviceID); not enlarging the page file."
        return
    }

    $cs = Get-CimInstance Win32_ComputerSystem
    if ($cs.AutomaticManagedPagefile) {
        Write-Host "Turning off automatic page file management"
        Set-CimInstance -InputObject $cs -Property @{ AutomaticManagedPagefile = $false }
    }

    $path     = "$($drive.DeviceID)\pagefile.sys"
    $existing = Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -ieq $path }
    if ($existing) {
        Write-Host "Resizing page file $path to $maxMB MB"
        Set-CimInstance -InputObject $existing -Property @{ InitialSize = $maxMB; MaximumSize = $maxMB }
    } else {
        Write-Host "Creating page file $path at $maxMB MB"
        New-CimInstance -ClassName Win32_PageFileSetting `
                        -Property @{ Name = $path; InitialSize = $maxMB; MaximumSize = $maxMB } | Out-Null
    }
} catch {
    Write-Warning "Could not adjust the page file: $($_.Exception.Message)"
    return
}

$after = Get-CommitGB
Write-Host "Memory after:"
Write-Host "  commit limit  : $($after.LimitGB) GB"
Write-Host "  commit free   : $($after.FreeGB) GB"

if ($after.LimitGB -le $before.LimitGB) {
    # Windows often defers page file growth to the next boot, which we cannot do
    # on a hosted agent. Say so plainly: the build may still run out of heap, and
    # this line is what explains why.
    Write-Warning ("Commit limit did not grow (still $($after.LimitGB) GB). " +
                   "Windows may be deferring the change to the next reboot.")
}
