# Hashes all files that influence how the C++ dependencies are built and
# exposes the result as the pipeline variable DEPS_FILES_HASH via ##vso logging.
param(
    [Parameter(Mandatory)][string]$SrcDir
)
$ErrorActionPreference = 'Stop'

$files = @(
    "$SrcDir\vcpkg.json",
    "$SrcDir\vcpkg-configuration.json",
    "$SrcDir\cpp\manifest\qdk-chemistry\cgmanifest.json",
    "$SrcDir\external\macis\manifest\cgmanifest.json",
    "$SrcDir\cpp\cmake\third_party.cmake",
    "$SrcDir\cpp\cmake\modules\DependencyManager.cmake",
    "$SrcDir\external\macis\src\lobpcgxx\CMakeLists.txt",
    "$SrcDir\.pipelines\toolchains\windows.cmake"
)
if (Test-Path "$SrcDir\vcpkg-overlay") {
    $files += Get-ChildItem "$SrcDir\vcpkg-overlay" -Recurse -File |
              Select-Object -ExpandProperty FullName
}
if (Test-Path "$SrcDir\cpp\cmake\patches") {
    $files += Get-ChildItem "$SrcDir\cpp\cmake\patches" -Recurse -File |
              Select-Object -ExpandProperty FullName
}
$hashes = $files | Where-Object { Test-Path $_ } | Sort-Object |
          ForEach-Object { (Get-FileHash $_ -Algorithm SHA256).Hash }
$combined  = $hashes -join ","
$hashBytes = [System.Security.Cryptography.SHA256]::Create().ComputeHash(
                 [System.Text.Encoding]::UTF8.GetBytes($combined))
$hashStr   = ($hashBytes | ForEach-Object { $_.ToString("x2") }) -join ""
Write-Host "Dependency files hash: $hashStr"
Write-Host "##vso[task.setvariable variable=DEPS_FILES_HASH]$hashStr"
