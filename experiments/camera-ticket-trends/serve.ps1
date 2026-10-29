param([int]$Port = 4186, [string]$Uv = 'uv')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$browserDist = Join-Path $PSScriptRoot 'browser/dist'
if (Test-Path -LiteralPath (Join-Path $browserDist 'index.html')) {
    $entryPath = ''
} else {
    $browserDist = $PSScriptRoot
    $entryPath = 'Camera_Ticket_Trends.html'
    if (-not (Test-Path -LiteralPath (Join-Path $browserDist $entryPath))) {
        throw 'Download Camera_Ticket_Trends.html or build the dashboard before starting the server.'
    }
}
Write-Host "Open http://127.0.0.1:$Port/$entryPath . Press Ctrl+C to stop."
& $Uv run --project $projectRoot --locked python -m http.server $Port --bind 127.0.0.1 --directory $browserDist
