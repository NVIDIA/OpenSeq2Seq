#!/usr/bin/env bash
zip_name=LibriSpeech-dev-clean.tar.gz
mkdir calibration/sound_files
output_dir=calibration/sound_files/
target_dir=calibration/sound_files_wav/
curl 'http://www.openslr.org/resources/12/dev-clean.tar.gz' -H 'Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1'\
-H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36' \
-H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8' \
-H 'Referer: http://www.openslr.org/12/' \
-H 'Accept-Encoding: gzip, deflate' \
-H 'Accept-Language: en-US,en;q=0.9' \
--compressed -o $zip_name
echo Download complete, Extracting now...
tar -xzf $zip_name -C $output_dir
echo Extraction Complete, Deleting tar...
rm $zip_name
echo Deletion Complete, converting now...
python scripts/change_sample_rate.py --source_dir=$output_dir --target_dir=$target_dir
rm -rf $output_dir
echo Conversion complete, preparing calibration file...
sed -i "s|#FILE_PATH_HERE#|$target_dir|g" calibration/sample.csv
sed -i "s|#FILE_PATH_HERE#|$target_dir|g" calibration/target.json
echo Calibration files prepared.