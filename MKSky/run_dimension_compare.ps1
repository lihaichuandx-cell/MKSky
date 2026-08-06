param(
    [int]$PointCount = 5000000,
    [int]$WarmupRuns = 1,
    [int]$MeasuredRuns = 3,
    [int]$Seed = 12345
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $projectRoot 'bin\Release\MKSkyBenchmark.exe'
$resultRoot = Join-Path $projectRoot 'results\dimension_6_16'
$log = Join-Path $resultRoot 'dimension_compare.log'
$dimensions = @(6, 7, 8, 10, 12, 16)
$distributions = @('ind', 'corr', 'anti')

if (-not (Test-Path -LiteralPath $exe)) {
    throw "Release executable not found: $exe"
}
New-Item -ItemType Directory -Path $resultRoot -Force | Out-Null

foreach ($distribution in $distributions) {
    foreach ($dim in $dimensions) {
        $csv = Join-Path $resultRoot ("raw_{0}_d{1}.csv" -f $distribution, $dim)
        $complete = $false
        if (Test-Path -LiteralPath $csv) {
            $rows = @(Import-Csv -LiteralPath $csv)
            $complete = $rows.Count -eq 2 -and
                @($rows | Where-Object { $_.verification -notlike 'PASS:*' }).Count -eq 0
        }
        if ($complete) {
            "SKIP completed: distribution=$distribution dim=$dim" | Tee-Object -FilePath $log -Append
            continue
        }

        "START distribution=$distribution dim=$dim n=$PointCount" | Tee-Object -FilePath $log -Append
        & $exe --n $PointCount --dim $dim --distribution $distribution --seed $Seed `
            --warmup $WarmupRuns --repeat $MeasuredRuns `
            --algorithms optimized,current --csv $csv 2>&1 |
            Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Benchmark failed: distribution=$distribution dim=$dim exit=$LASTEXITCODE"
        }
        "DONE distribution=$distribution dim=$dim" | Tee-Object -FilePath $log -Append
    }
}

$allRows = foreach ($distribution in $distributions) {
    foreach ($dim in $dimensions) {
        $csv = Join-Path $resultRoot ("raw_{0}_d{1}.csv" -f $distribution, $dim)
        Import-Csv -LiteralPath $csv
    }
}
$allRows | Export-Csv -LiteralPath (Join-Path $resultRoot 'dimension_compare_raw.csv') `
    -NoTypeInformation -Encoding UTF8
"ALL MEASUREMENTS COMPLETE" | Tee-Object -FilePath $log -Append
