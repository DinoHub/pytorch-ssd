wget -P models https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
wget "https://docs.google.com/uc?export=download&id=18cR_cQaLCAOb63xZYrEWSuEIFlqPMvJj" -O models/mb2-ssd-lite-seaships.zip
unzip models/mb2-ssd-lite-seaships.zip -d models/mb2-ssd-lite-seaships
mv models/mb2-ssd-lite-seaships/mb2-ssd-lite/* models/mb2-ssd-lite-seaships
rm -r models/mb2-ssd-lite-seaships/mb2-ssd-lite