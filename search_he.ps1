$SourceDir = "D:\AI\Gits\hebrew_tutor_ai_poc\data\espeak-ng-source"
$BuildDir = "$SourceDir\build"
$InstallDir = "C:\Program Files\eSpeak NG"
$HePath = Get-ChildItem -Path $SourceDir, $BuildDir, $InstallDir -Recurse -Filter "he_dict" -File -ErrorAction SilentlyContinue | Select-Object FullName
if ($HePath) {
    $HePath | ForEach-Object { Write-Output "Found 'he_dict' at: $($_.FullName)" }
} else {
    Write-Error "'he_dict' file not found—check install logs for Hebrew compilation."
}