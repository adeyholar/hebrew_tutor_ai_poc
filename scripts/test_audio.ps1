$EspeakPath = "C:\Program Files\eSpeak NG\bin\espeak-ng.exe"
$WavPath = "D:\AI\Gits\hebrew_tutor_ai_poc\data\test.wav"
if (Test-Path $EspeakPath) {
    & $EspeakPath -v en -w $WavPath "Hello world"  # Output WAV
    if (Test-Path $WavPath) {
        Write-Output "WAV created at $WavPath—play with Media Player."
        Start-Process $WavPath  # Auto-open in default player
    } else {
        Write-Error "WAV creation failed—check eSpeak logs."
    }
} else {
    Write-Error "eSpeak binary missing."
}