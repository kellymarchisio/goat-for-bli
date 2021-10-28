HOST=https://dl.fbaipublicfiles.com/arrival/dictionaries

echo Collecting bilingual dictionaries... 
for pair in en-de ru-en 
do
	mkdir -p $pair/dev $pair/train $pair/test
	wget $HOST/$pair.txt -P $pair
	wget $HOST/$pair.0-5000.txt -P $pair/train
	wget $HOST/$pair.5000-6500.txt -P $pair/test
done

echo Making dev sets...
python make_devsets.py

echo Making data one-to-one...
python one-to-one.py en-de/train/en-de.0-5000.txt 2
python one-to-one.py ru-en/train/ru-en.0-5000.txt 2

echo Dictionary creation done.

