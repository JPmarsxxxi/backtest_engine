# Registers the 3 forward-test strategies as every-minute one-shot "tick" tasks — the crash-proof
# replacement for the old logon-only daemons that kept dying on sleep/reboot. Runs NON-ELEVATED.
# Re-runnable: it unregisters the new-style tasks first, then recreates them. Idempotent.
#
#   powershell -ExecutionPolicy Bypass -File setup_live_tasks.ps1
#
# NOTE: the OLD tasks (FX_Seasonal_016, FX_Seasonal_016_v2, US500_Overnight_023) were registered
# elevated and CANNOT be removed from here (Access denied). Delete them once from an ADMIN prompt:
#   schtasks /delete /tn FX_Seasonal_016 /f
#   schtasks /delete /tn FX_Seasonal_016_v2 /f
#   schtasks /delete /tn US500_Overnight_023 /f
# Until then they are harmless: they only fire on a NEW logon, and the tick refreshes the same lock
# file that makes an old daemon self-exit. But deleting them removes any doubt.

$ErrorActionPreference = "Stop"
$dir  = "C:\Users\User\backtest_engine\backtest_engine2"
$user = "$env:USERDOMAIN\$env:USERNAME"
# pythonw.exe = the WINDOWLESS Python interpreter -> no console window pops up each minute. The script
# redirects its own output to data\<module>_tick.out, so nothing is lost by dropping the .bat wrapper.
$pyw  = "C:\ProgramData\anaconda3\pythonw.exe"

$tasks = @(
  # FX_016_v1_tick REMOVED 2026-07-15 (killed on its pre-committed criterion — do not re-add)
  @{ Name = "FX_016_v2_tick";     Args = "strat_tick.py --kind fx --module fx_seasonal_live_v2" },
  @{ Name = "FX_016_v3_tick";     Args = "strat_tick.py --kind fx --module fx_seasonal_live_v3" },
  @{ Name = "US500_023_tick";     Args = "strat_tick.py --kind overnight --module overnight_live" },
  @{ Name = "US500_023_v2_tick";  Args = "strat_tick.py --kind overnight --module overnight_live_v2" }
)

foreach ($t in $tasks) {
  $name = $t.Name
  try { Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop; Write-Host "removed old $name" } catch {}

  $action = New-ScheduledTaskAction -Execute $pyw -Argument $t.Args -WorkingDirectory $dir

  # Every-minute repetition, indefinitely (10-yr duration = effectively forever), anchored to today.
  $repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
              -RepetitionInterval (New-TimeSpan -Minutes 1) `
              -RepetitionDuration  (New-TimeSpan -Days 3650)
  # Re-arm on logon so it resumes immediately after a reboot / re-login (MT5 needs the interactive session).
  $logon  = New-ScheduledTaskTrigger -AtLogOn -User $user

  $settings = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -StartWhenAvailable -WakeToRun `
                -MultipleInstances IgnoreNew `
                -ExecutionTimeLimit (New-TimeSpan -Minutes 4) `
                -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

  $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

  Register-ScheduledTask -TaskName $name -Action $action -Trigger @($repeat, $logon) `
      -Settings $settings -Principal $principal `
      -Description "Every-minute one-shot tick for the live FTMO forward test ($($t.Args)). Windowless via pythonw. Replaces the fragile logon daemon." | Out-Null
  Write-Host "registered $name -> pythonw $($t.Args)"
}

# Keep the trading box awake on AC so overnight actions actually fire (the daemons' #1 killer was sleep).
try { powercfg /change standby-timeout-ac 0; powercfg /change hibernate-timeout-ac 0; Write-Host "power: sleep/hibernate on AC disabled" }
catch { Write-Host "power plan change skipped (set 'Sleep = Never' on AC manually)" }

Write-Host "`n=== new tasks ==="
Get-ScheduledTask | Where-Object { $_.TaskName -match "_tick$" } | Select-Object TaskName, State | Format-Table -AutoSize
