#!/usr/bin/env bash
echo -e
zip_name=dev-clean.tar.gz
mkdir calibration/sound_files
output_dir=calibration/sound_files/
target_dir=calibration/sound_files_wav/
wget http://www.openslr.org/resources/12/${zip_name} -O $zip_name
echo Download complete, extracting now...
tar -xzf $zip_name -C $output_dir
echo Extraction Complete, deleting tar...
rm $zip_name
echo Deletion Complete, converting now...
python scripts/change_sample_rate.py --source_dir=$output_dir --target_dir=$target_dir
rm -rf $output_dir
echo Conversion complete, preparing calibration file...
sed -i "s|#FILE_PATH_HERE#|$target_dir|g" calibration/sample.csv
sed -i "s|#FILE_PATH_HERE#|$target_dir|g" calibration/target.json
echo Calibration files prepared.

