# relocate_mbrola.ps1 - Modular Relocation Script for MBROLA Voice Data (e.g., hb1 to mbrola dir)
param (
    [string]$Voice = "hb1"  # Scalable param for hb1/hb2/etc
)

# Modular Paths (secure, absolute—avoid relative errors)
$VoiceDataCurrent = "C:\Program Files\eSpeak NG\espeak-ng-data\voices\mb\$Voice\$Voice"
$TargetDir = "C:\Program Files\eSpeak NG\espeak-ng-data\mbrola\$Voice"
$BackupPath = "D:\AI\Gits\hebrew_tutor_ai_poc\data\mbrola_voices\$Voice_backup"

# Validation Before Action (best-practice error handling)
if (Test-Path $VoiceDataCurrent) {
    Write-Output "Validated: hb1 data at current path $VoiceDataCurrent—proceeding to relocate."
} else {
    Write-Error "Source $Voice data missing—re-download/copy from project data\mbrola_voices\$Voice. Aborting."
    exit 1  # Secure exit on error
}

# Create Target Dir (idempotent, silent)
mkdir $TargetDir -Force | Out-Null
Write-Output "Target dir ready: $TargetDir"

# Backup Original (security: prevent loss, scalable copy)
Copy-Item $VoiceDataCurrent -Destination $BackupPath -Force
Write-Output "Backup created at $BackupPath (for recovery if needed)."

# Relocate (move for clean-up, force for overwrite safety)
Move-Item $VoiceDataCurrent -Destination "$TargetDir\$Voice" -Force
if (Test-Path "$TargetDir\$Voice") {
    Write-Output "Relocation successful: $Voice now at $TargetDir\$Voice."
} else {
    Write-Error "Relocation failed—check permissions/path (run as admin?). Restore from backup $BackupPath."
    exit 1
}

# Post-Relocation Test (functional check, modular)
$env:PATH += ";C:\Program Files\eSpeak NG"
refreshenv  # Reload env (ignore wmic warning)
espeak-ng -v "mb/mb-$Voice" "שלום עולם"  # Basic test