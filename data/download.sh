#!/usr/bin/env bash
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$1" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -f /tmp/cookies.txt
}
mkdir -p converted
mkdir -p converted/librosa
mkdir -p converted/mauch
mkdir -p audio
#download raw audio
gdrive_download 1WzdcHeLeFHrYu_2_NDTEEglfrKLLKc1c ./audio/TheBeatles.zip
gdrive_download 161eEk-o1ulujRh_n-hYmQxlwdmbhWWja ./audio/billboard.zip
gdrive_download 1GVBNRwZ_YFHD9aroqP_NaI57H07_R3GR ./audio/Queen_CaroleKing.zip
gdrive_download 1s55LgFKyybeSueruV8Xvtwh6TE4yJnAb ./audio/Jay_Chou.zip
#download converted by librosa audio
gdrive_download 19aXFrhoZdg3Jvf6tDpkPwPFMznuLUW1U ./converted/librosa/TheBeatles.zip
gdrive_download 1w8Mo2r6ml1v76SiU3MTjgoWiNjFLOH9a ./converted/librosa/USPop179.zip
gdrive_download 1WbyWA4UcYuMHw7QvXMpb-i1PJrpKDKiZ ./converted/librosa/CaroleKing_Queen.zip
gdrive_download 19kqa5sZ7YwWd4KHZ8DdyUSP1eDkKnhiL ./converted/librosa/fullmod-1-3.zip
gdrive_download 1tcB1rcxebdrv88Sv2oUDI3YYL_zlDqoO ./converted/librosa/fullmod24.zip
#download converted list
gdrive_download 1wagyEwmqS6rX1CqIptFtzHVNkVHQOoJD ../TheBeatles180List_converted_librosa.txt
gdrive_download 1V5zdvcB50YLfnTlsKyZspvTuVbWghrLE ../USPop179List_converted_librosa.txt
gdrive_download 1vMFUvgJrAzCsO4PjZ98pYkirGRGNt3Sk ../CaroleKingQueen26List_converted_librosa.txt
gdrive_download 1m8wC0vAc4p-HbNx68PH1gOfSKv2FE_EU ../FullList386_converted_librosa.txt
gdrive_download 1HSWo6Wv1fWmWjViN13TERxIgUpCMxoHe ../FullList386_converted_mod_librosa.txt
#unzip
unzip -q audio/TheBeatles.zip -d audio
unzip -q audio/billboard.zip -d audio
unzip -q audio/Queen_CaroleKing.zip -d audio
unzip -q audio/Jay_Chou.zip -d audio


unzip -q converted/librosa/TheBeatles.zip -d converted/librosa
unzip -q converted/librosa/USPop179.zip -d converted/librosa
unzip -q converted/librosa/CaroleKing_Queen.zip -d converted/librosa
unzip -q converted/librosa/fullmod-1-3.zip -d converted/librosa
unzip -q converted/librosa/fullmod24.zip -d converted/librosa

rm -f audio/TheBeatles.zip
rm -f audio/billboard.zip
rm -f audio/Queen_CaroleKing.zip
rm -f audio/Jay_Chou.zip

rm -f converted/librosa/TheBeatles.zip
rm -f converted/librosa/USPop179.zip
rm -f converted/librosa/CaroleKing_Queen.zip
rm -f converted/librosa/fullmod24.zip
rm -f converted/librosa/fullmod-1-3.zip

echo OK