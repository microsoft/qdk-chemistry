<#
.SYNOPSIS
    Rewrite relative README links to absolute GitHub URLs.

.DESCRIPTION
    Equivalent to prepare-readme.sh — converts relative Markdown links in
    README.md to absolute github.com URLs so they render correctly on PyPI,
    then writes the result to python/README.md.

    Transformations applied (mirrors the sed -E expressions in prepare-readme.sh):
      - ](./path)         -> ](https://.../blob/main/path)
      - ](UPPER_FILE.md)  -> ](https://.../blob/main/UPPER_FILE.md)
      - `examples/`       -> [`examples/`](https://.../tree/main/examples)
      - `cpp/include/`    -> [`cpp/include/`](https://.../tree/main/cpp/include)
      - ## Project Structure ... closing ``` (range) deleted

.PARAMETER SrcDir
    Root of the qdk-chemistry repository checkout.
    Defaults to the directory two levels above this script
    (i.e. the repo root when the script lives in .pipelines/pip-scripts/).
#>
param(
    [string]$SrcDir = (Resolve-Path "$PSScriptRoot\..\..")
)

$ErrorActionPreference = 'Stop'

$ghBlob = 'https://github.com/microsoft/qdk-chemistry/blob/main'
$ghTree = 'https://github.com/microsoft/qdk-chemistry/tree/main'

$lines = Get-Content -Encoding utf8 "$SrcDir\README.md"

$out      = [System.Collections.Generic.List[string]]::new()
$deleting = $false
foreach ($line in $lines) {
    # Range delete: ## Project Structure ... closing ```
    if ($line -match '^## Project Structure$') { $deleting = $true;  continue }
    if ($deleting) {
        if ($line -match '^```$') { $deleting = $false }
        continue
    }
    $line = $line -replace '\]\(\./([^)]+)\)',               "]($ghBlob/`$1)"
    $line = $line -replace '\]\(([A-Z][A-Z_]*\.(md|txt))\)', "]($ghBlob/`$1)"
    $line = $line -replace '`examples/`',                    "[``examples/``]($ghTree/examples)"
    $line = $line -replace '`cpp/include/`',                 "[``cpp/include/``]($ghTree/cpp/include)"
    $out.Add($line)
}

$out | Set-Content -Encoding utf8 "$SrcDir\python\README.md"
Write-Host "README prepared."
