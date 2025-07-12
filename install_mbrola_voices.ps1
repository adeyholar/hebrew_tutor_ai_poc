# install_mbrola_voices.ps1
param (
    [string]$Voice = "hb1"
)
# Modular dir creation
mkdir data\mbrola_voices -Force | Out-Null

# Download & Extract ZIP (archive.org fallback)
Invoke-WebRequest -Uri "https://github.com/thiekus/MBROLA/releases/download/3.3/mbrola_build_3.3_rev2.zip" -OutFile "data\mbrola_voices\mbrola_build_3.3_rev2.zip" -UseBasicParsing  # BasicParsing for network issues
7z x "data\mbrola_voices\mbrola_build_3.3_rev2.zip" -odata\mbrola_voices\extracted -y

# Copy DLL (Win64)
$Arch = "Win64"
$DllSource = "data\mbrola_voices\extracted\$Arch\Release\mbrola.dll"
Copy-Item $DllSource -Destination "C:\Program Files\eSpeak NG\mbrola.dll" -Force

# Relocation & Config with absolute path
$TargetDir = "C:\Program Files\eSpeak NG\espeak-ng-data\mbrola\$Voice"
mkdir $TargetDir -Force | Out-Null
Copy-Item "data\mbrola_voices\$Voice" -Destination "$TargetDir\$Voice" -Force  # From project data
$ConfigPath = "C:\Program Files\eSpeak NG\espeak-ng-data\voices\mb\mb-$Voice"
@"
name mb-$Voice
language he
gender male
mbrola $Voice C:\Program Files\eSpeak NG\espeak-ng-data\mbrola\$Voice\$Voice `$voice
"@.Trim() | Set-Content $ConfigPath -Encoding UTF8 -Force -NoNewline

# Test
$env:PATH += ";C:\Program Files\eSpeak NG"
espeak-ng -v "mb/mb-$Voice" "שלום עולם"
espeak-ng -v "mb/mb-$Voice" --phonemes "בְּרֵאשִׁית בָּרָא אֱלֹהִים"