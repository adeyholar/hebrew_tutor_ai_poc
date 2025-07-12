# test_wav.ps1 - Modular Voice Test Script with WAV Output (English/Hebrew)
param (
    [string]$Language = "en",  # Scalable param: 'en' or 'he'
    [string]$Text = "Hello world"  # Default English; override for Hebrew
)

# Modular Paths (secure absolute, scalable for other binaries)
$EspeakPath = "C:\Program Files\eSpeak NG\bin\espeak-ng.exe"
$WavPath = "D:\AI\Gits\hebrew_tutor_ai_poc\data\test_$Language.wav"

# Validation (best-practice error handling, secure check)
if (Test-Path $EspeakPath) {
    Write-Output "Validated: eSpeak-NG binary at $EspeakPath."
} else {
    Write-Error "eSpeak binary missing—check path or re-install."
    exit 1  # Secure exit on error
}

# Override Text for Hebrew (modular Tanach sample for tutor relevance)
if ($Language -eq "he") {
    $Text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ"
}

# Generate WAV (scalable -w option, UTF8 for Hebrew)
& $EspeakPath -v $Language -w $WavPath $Text
if (Test-Path $WavPath) {
    Write-Output "WAV created at $WavPath—play with Media Player (volume up, no mute)."
    Start-Process $WavPath  # Auto-open in default player
} else {
    Write-Error "WAV creation failed—check eSpeak logs or voice deps."
}

# Optional Clean (secure removal after test)
# Remove-Item $WavPath -Force; Write-Output "Cleaned WAV."