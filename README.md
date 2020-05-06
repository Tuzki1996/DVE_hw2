# DVE_hw2

## Requirement
```
numpy
scipy
opencv-python
```

## Usage
Example:
```
python projection.py --img_dir dir/to/images --f 459.581 --ext JPG

python image_stitching.py --img_dir dir/to/images --ext JPG --output_dir dir/to/output
```
Required arguments:
```
projection.py

--img_dir     存放圖片的資料夾
--f           焦距
--ext         圖片附檔名
```
投影完的圖片放在`img_dir_projected`
例如說`img_dir`是`parrington`，結果就會在`parrington_projected`

```
image_stitching.py

--img_dir     存放圖片（投影過）的資料夾
--ext         圖片附檔名
--output_dir  輸出資料夾
```

輸出的圖片`panaroma.jpg`放在`output_dir`
