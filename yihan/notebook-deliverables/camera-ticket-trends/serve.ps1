param([int]$Port = 4187)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw 'Install uv and run uv sync --locked from the repository root first.'
}
$pagePath = Join-Path $PSScriptRoot 'Camera_Ticket_Trends.html'
if (-not (Test-Path -LiteralPath $pagePath)) {
    throw 'Download Camera_Ticket_Trends.html into this folder first.'
}
Write-Host "Open http://127.0.0.1:$Port/Camera_Ticket_Trends.html . Press Ctrl+C to stop."
& $uvCommand.Source run --no-sync --project $projectRoot python -m http.server $Port --bind 127.0.0.1 --directory $PSScriptRoot
