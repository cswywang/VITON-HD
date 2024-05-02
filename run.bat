@echo off
f:
cd F:\VITON-HD\preprocess
call activate VITON
python .\cloth-seg\infer.py  --cloth=%2
python color.py --type 1 --cloth=%2
cd cloth-seg
python clean_mask.py --cloth=%2
cd ..
cd ./person-seg/image_segmentation/human_part_segmentation
python human_part_segmentation_atr.py  --person=%1
python human_part_segmentation_lip.py  --person=%1
python palette.py --person=%1
python clean_mask.py --person=%1
move input\%1.jpg  ..\..\..\openpose\examples\media
cd ..
cd ..
cd ..
cd openpose
bin\OpenPoseDemo.exe --image_dir examples\media --hand --write_images output\ --write_json output/ --disable_blending
move examples\media\%1.jpg ..\..\datasets\test\image
move output\%1_keypoints.json   ..\..\datasets\test\openpose-json
move output\%1_rendered.png  ..\..\datasets\test\openpose-img
cd ..
move cloth-seg\input_images\%2  ..\datasets\test\cloth
move cloth-seg\output_images\%2 ..\datasets\test\cloth-mask
move person-seg\image_segmentation\human_part_segmentation\output\%1.png  ..\datasets\test\image-parse
echo %1.jpg %2  > ..\datasets\test_pairs.txt
cd ..
python test.py --name test