# Video Action Recognition Models

PyTorch implementation of the video action recognition models from recent popular research works.

## Code

The code framework of this repository is based on [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) ([Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf)). For implementation of recent popular models, we have done following new works:

- Update code to PyTorch 1.0
- Add new models: R(2+1)D, NL-I3D(baseline)
- Add new datasets: Something-to-Something V1
- Adjust datasets processing
- Add sample rate option for temporal sampling
- Add MultiStepLR policy
- Add uniformly temporal and spatial cropping for multiple crop testing (refer to [NL-I3D](https://github.com/facebookresearch/video-nonlocal-net))
- Integrate test and evaluation processing
- Add top-5 accuracy calculation

## Models

We have currently implemented following two models. More newly models will be implemented and added continuously.

- r2plus1d.py (R(2+1)D, original [Code/Pretrained-model](https://github.com/facebookresearch/VMZ) and [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf))
- i3d_nl.py (NL-I3D baseline, original [Code/Pretrained-model](https://github.com/facebookresearch/video-nonlocal-net) and [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf))

## Datasets

We have currently tested following four datasets. More newly datasets will be tested and added continuously.

- [Kinetics-400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
- [UCF101](http://crcv.ucf.edu/data/UCF101.php)
- [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [Something-Something V1](https://20bn.com/datasets/something-something/v1/)

## Requirements

- Python3
- PyTorch 1.0 and Dependencies
- FFMPEG

## Preprocessing

The content of this part (Preprocessing)  uses the template of README.md from [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).

#### Kinetics-400

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
* Convert from video files to jpg files using ```preprocess/video_jpg_kinetics.py```.

```bash
python preprocess/video_jpg_kinetics.py video_directory jpg_video_directory
```

* Generate n_frames files using ```preprocess/n_frames_kinetics.py```.

```bash
python preprocess/n_frames_kinetics.py jpg_video_directory
```

* Generate annotation file in json format using ```preprocess/kinetics_json.py```.
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python preprocess/kinetics_json.py train_csv_path val_csv_path test_csv_path dst_json_path
```

#### UCF101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```preprocess/video_jpg_ucf101_hmdb51.py```.

```bash
python preprocess/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```preprocess/n_frames_ucf101_hmdb51.py```.

```bash
python preprocess/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format using ```preprocess/ucf101_json.py```.
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt.

```bash
python preprocess/ucf101_json.py annotation_dir_path
```

#### HMDB51

* Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Convert from avi to jpg files using ```preprocess/video_jpg_ucf101_hmdb51.py```.

```bash
python preprocess/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```preprocess/n_frames_ucf101_hmdb51.py```.

```bash
python preprocess/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format using ```preprocess/hmdb51_json.py```.
  * ```annotation_dir_path``` includes brush_hair_test_split1.txt, ...

```bash
python preprocess/hmdb51_json.py annotation_dir_path
```

#### Something-Something V1

* Download videos and labels [here](https://20bn.com/datasets/something-something/v1/).
* Convert from origin image files to new jpg files using ```preprocess/video_jpg_something.py```.

```bash
python preprocess/video_jpg_something.py origin_image_directory jpg_video_directory
```

* Generate n_frames files using ```preprocess/n_frames_something.py```.

```bash
python preprocess/n_frames_something.py jpg_video_directory
```

* Generate annotation file in json format using ```preprocess/something_json.py```.
  * ```annotation_dir_path``` includes something-something-v1-labels.csv, ...

```bash
python preprocess/something_json.py annotation_dir_path
```

## Running

The content of this part (Running) uses the template of README.md from [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).

Assume the structure of data directories is the following:

```misc

root_path/
  annotations/
    kinetics/
      kinetics_01.json
  video/
    kinetics/
      training/
        .../ (directories of class names)
          .../ (directories of videos names)
            .../ (jpg files)
      validation/
        .../ (directories of class names)
          .../ (directories of videos names)
            .../ (jpg files)
  results/
    kinetics/
      save_100.pth
      log.txt

```

Train R(2+1)D-34 on the Kinetics dataset (400 classes).  
```bash
bash scripts/train_kinetics_r2plus1d.sh
```

Batch size is 32.  
Use 16 threads for data loading.  
Temporal sample duration is 16, and sample rate is 4.  
Spatial sample size is 112.  
Save models at every 5 epochs, totally train 100 epochs. 

Details of script:    

```bash
python tools/main.py \
--root_path ${ROOT_PATH} \
--video_path videos/kinetics \
--annotation_path annotations/kinetics_01.json \
--result_path results/kinetics \
--dataset kinetics --n_classes 400 \
--sample_size 112 --sample_duration 16 --sample_rate 4 \
--learning_rate 0.01 --n_epochs 100 \
--batch_size 32 --n_threads 16 --checkpoint 5  \
--model r2plus1d --model_depth 34 \
2>&1 | tee ${ROOT_PATH}/results/kinetics/log.txt
```

Continue training option:  
```--resume_path results/save_100.pth```

## Reference

- Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? ([Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf), [Code](https://github.com/kenshohara/3D-ResNets-PyTorch))
- A Closer Look at Spatiotemporal Convolutions for Action Recognition. ([Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf), [Code](https://github.com/facebookresearch/VMZ))
- Non-local Neural Networks. ([Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf), [Code](https://github.com/facebookresearch/video-nonlocal-net))

