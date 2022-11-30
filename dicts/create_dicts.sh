HOST=https://dl.fbaipublicfiles.com/arrival/dictionaries

echo Collecting bilingual dictionaries...
for lang in bn bs et fa id mk ms ta vi zh en de fr es it ja
do
	mkdir -p en-$lang/dev en-$lang/train en-$lang/test
	wget $HOST/en-$lang.txt -P en-$lang
	wget $HOST/en-$lang.0-5000.txt -P en-$lang/train
	wget $HOST/en-$lang.5000-6500.txt -P en-$lang/test

	mkdir -p $lang-en/dev $lang-en/train $lang-en/test
	wget $HOST/$lang-en.txt -P $lang-en
	wget $HOST/$lang-en.0-5000.txt -P $lang-en/train
	wget $HOST/$lang-en.5000-6500.txt -P $lang-en/test
done

for pair in de-es es-de it-fr fr-it es-pt pt-es pt-de de-pt
do
	mkdir -p $pair/dev $pair/train $pair/test
	wget $HOST/$pair.txt -P $pair
	wget $HOST/$pair.0-5000.txt -P $pair/train
	wget $HOST/$pair.5000-6500.txt -P $pair/test
done

# Fix tokenization issue with some words in some to-En dictionaris.
for lang in ro ja de fi it ru ta zh fr
do
	sed -i 's/,//g' $lang-en/$lang-en.txt
	sed -i 's/,//g' $lang-en/train/$lang-en.0-5000.txt
done


echo Making dev sets...
python make_devsets.py

echo Making data one-to-one...
for lang in bn bs et fa id mk ms ta vi zh en de fr es it ja
do
	python one-to-one.py en-$lang/train/en-$lang.0-5000.txt 2
	python one-to-one.py $lang-en/train/$lang-en.0-5000.txt 2
done
for pair in de-es es-de it-fr fr-it es-pt pt-es pt-de de-pt
do
	python one-to-one.py $pair/train/$pair.0-5000.txt 2
done

echo Dictionary creation done.

