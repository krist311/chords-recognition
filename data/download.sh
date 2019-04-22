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
#download converted by librosa audio
gdrive_download 1hiTi_CPKxu9Qpli-zch1vINa4iY5iE9s ./converted/librosa/TheBeatles.zip
#download converted by using Mauch's algorithm
gdrive_download 1WzdcHeLeFHrYu_2_NDTEEglfrKLLKc1c ./converted/mauch/TheBeatles.zip
#download converted list
gdrive_download 1E-TVqZvlFIJ2KzxmkkdhPxlKXxzQAcZJ TheBeatles180List_converted.txt
#unzip
unzip -q audio/TheBeatles.zip -d audio
unzip -q converted/librosa/TheBeatles.zip -d converted/librosa
unzip -q converted/mauch/TheBeatles.zip -d converted/mauch

rm -f audio/TheBeatles.zip
rm -f converted/librosa/TheBeatles.zip
rm -f converted/mauch/TheBeatles.zip

echo OK