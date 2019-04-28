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
#download converted by librosa audio
gdrive_download 1hiTi_CPKxu9Qpli-zch1vINa4iY5iE9s ./converted/librosa/TheBeatles.zip
#download converted by Mauch's algorithm
gdrive_download 1WzdcHeLeFHrYu_2_NDTEEglfrKLLKc1c ./converted/mauch/TheBeatles.zip
#download converted list
gdrive_download 11fJyHqZGN0baKEjYlaFdVqsnUTIkCJb3 ../TheBeatles180List_converted_librosa.txt
gdrive_download 1ny7Vyir9sEVIMqnFa-3gkPVd7f46wUY1 ../TheBeatles180List_converted_mauch.txt
#unzip
unzip -q audio/TheBeatles.zip -d audio
unzip -q audio/billboard.zip -d audio
unzip -q converted/librosa/TheBeatles.zip -d converted/librosa
unzip -q converted/mauch/TheBeatles.zip -d converted/mauch

rm -f audio/TheBeatles.zip
rm -f audio/billboard.zip
rm -f audio/Queen_CaroleKing.zip
rm -f converted/librosa/TheBeatles.zip
rm -f converted/mauch/TheBeatles.zip

echo OK