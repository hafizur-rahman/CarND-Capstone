
# Prepare Env
* Clone capstone project at `$HOME/CarND-Capstone`
* Install conda
* Create `capstone` env
    ```
    cd $HOME/CarND-Capstone

    conda create -n capstone python=2.7
    conda install --yes --file requirements.txt
    ```
* Prepare Tensorflow models lib
    ```
    conda activate capstone

    mkdir -p $HOME/tensorflow/models

    git clone https://github.com/tensorflow/models.git $HOME/tensorflow/models

    cd $HOME/tensorflow/models

    git checkout d1173bc9714b

    protoc object_detection/protos/*.proto --python_out=.
    ```
* 

# Download Data
* Download from https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view
* Download from https://1drv.ms/u/s!AtMG4jW974a6m8B-Q0A3tc2JbCigPw
* Unzip to `$HOME/data`

# Preprocess
* Add tensorflow models lib in python path
    ```
    export PYTHONPATH=$HOME/tensorflow/models:$HOME/tensorflow/models/slim
    ```
* Preprocess annotations data
    ```
    wget https://raw.githubusercontent.com/DavidObando/carnd/master/Term3/Project3/tlc/training/anno_capture_2_train.txt
    
    python preprocess.py -i anno_capture_2_train.txt -o $HOME/data/
    ```
* Generate TF Record
    ```
    python generate_tfrecord.py --csv_input=$HOME/data/train_labels.csv --output_path=annotations/sim_train.record --img_path=$HOME/data/capture-2
    ```

# Download model from model zoo
* Visit https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models and download the preferred model

* Unzip to your preferred dir
* Copy pipeline config file to `$HOME/CarND-Capstone/detection/training`, find PATH_TO_BE_CONFIGURED inside the file and modify accordingly

# Train the model (SSD Inception Model example)
    
* Sample command for SSD Inception Model Training
    ```
    cd $HOME/CarND-Capstone/detection

    python ~/tensorflow/models/object_detection/train.py --logtostderr --train_dir=training --pipeline_config_path=config/ssd_inception_v2_coco.config
    ```

# Prepare Inference Graph (SSD Inception Model example)
* Create inference graph
    ```
    cd $HOME/CarND-Capstone/detection

    python ~/tensorflow/models/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path config/ssd_inception_v2_coco.config --trained_checkpoint_prefix model.ckpt-2000000 --output_directory inference
    ```
* Copy frozen graph
    ```
    cd $HOME/CarND-Capstone/detection

    cp  ./inference/frozen_inference_graph.pb $HOME/CarND-Capstone/ros/src/tl_detector/light_classification/models/ssd_sim.pb
    ```

# Run in Simulator
Now you can try running ROS nodes and simulator to see whether traffic lights are detected properly.

# Reference:
* https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#