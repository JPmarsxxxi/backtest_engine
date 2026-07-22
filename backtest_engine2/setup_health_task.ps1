# Registers the MONTHLY #016 maintenance task (separate file from setup_live_tasks.ps1 so re-running
# this never touches the live trading ticks). Runs NON-ELEVATED, re-runnable (/f recreates).
#
#   powershell -ExecutionPolicy Bypass -File setup_health_task.ps1
#
# What the task does, 1st of every month at 12:00 (catches up if the box was off/asleep):
#   run_monthly_health_016.py, which runs sequentially:
#   1. edge_health_016.py  — extend hourly history to today, recompute the 21:00/23:00 hour averages
#                            full-window + trailing, write dated report + history csv (HUMAN reads it;
#                            it changes nothing live)
#   2. recal_caps_016.py   — re-measure the broker's entry-hour spreads from MT5 ticks and refresh
#                            data\fx_entry_caps_016.json (which fx_seasonal_live_v3.py loads, sanity-gated)
# Both scripts self-redirect stdout when run under pythonw (no console window).
# schtasks is used because New-ScheduledTaskTrigger has no -Monthly (and the CIM monthly trigger class
# rejects Register-ScheduledTask); the friendlier settings are patched on afterwards via Set-ScheduledTask.

$ErrorActionPreference = "Stop"
$dir  = "C:\Users\User\backtest_engine\backtest_engine2"
$pyw  = "C:\ProgramData\anaconda3\pythonw.exe"
$name = "FX_016_monthly_health"

schtasks /create /f /tn $name /sc monthly /d 1 /st 12:00 `
  /tr "\`"$pyw\`" \`"$dir\run_monthly_health_016.py\`"" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /create failed ($LASTEXITCODE)" }

$settings = New-ScheduledTaskSettingsSet `
              -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -StartWhenAvailable -WakeToRun `
              -MultipleInstances IgnoreNew `
              -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
              -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
Set-ScheduledTask -TaskName $name -Settings $settings | Out-Null

Write-Host "registered $name (1st of month 12:00, catch-up + wake enabled)"
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State | Format-Table -AutoSize
