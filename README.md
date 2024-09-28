# face_recognition_using_raspberrypi
This repo contains a face recognition model based on Mobilenetv2 for feature extraction and opencv Haarcascade classifier for face detection.
You can deploy this on your raspberrypi with a camera module attached.

## Prerequisites
Firstly clone the repo using the followind command:

    git clone https://github.com/Amd36/face_recognition.git

Make sure you have python installed (preferred version python=3.10) in your system. cd into the face_recognition folder you just created and install the prerequisites from the requirements.txt folder using:

    cd face_recognition
    pip install -r requirements.txt

**Note:** If you are using another device, it is recommended to use a virtual environment using venv or anaconda.

## Creating the custom dataset
If you already have a dataset prepared then you can copy it here. The final dataset directory should look like this:

    dataset/
    ├── person1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── image3.jpg
    ├── person2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── image3.jpg
    ├── person3/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── image3.jpg
    └── ...

The person names will be considered as class names so use short names, preferably first names only. 

Now update the labels.txt file with your class names (persons) manually or you can choose to run the following command that runs the **update_labels.py** script to update the labels according to your privided dataset directory (defaults to **"datasets/"**):

    python update_labels.py --dataset_dir <base directory of your dataset>

Then run the extract_faces_from_dataset specifyting the base directory of your custom dataset (deafults to **"datasets/"**). Example usage:

    python extract_faces_from_dataset.py --dataset_dir <base directory of your dataset>

This script will extract faces from your dataset and prepare the dataset for training. If you have already done this then you can skip this step.

If you do not have your dataset prepared then don't worry! I got you!
Make sure you have the camera module attached to the raspberrypi device you are using. 
Run the **create_samples.py** script specifying the name of the person you intend to recognise. The script will automatically open the camera module and create the dataset with 100 images under the specified name within **"datasets/"** directory. Example usage:

    python create_samples.py --name <name of the person>

## Training your model
Now comes the time to train your model. Run the following command specifying the number of epochs you want (defaults to **35**). This depends on number of classes and number of samples per class of your dataset. Try to look for a balance between accuracy and loss by observing the output graphs after each training session.

    python train.py --epochs <number of epochs>

You can run the training as many times as you want to get your desired accuracy and loss. This script will show you a graph of your accuracy and loss function after each training session for better judgement.
This script also saves the model in **.keras** and **.tflite** format with a **training_histroy.json** file with accuracy scores.

## Running inference
Finally we can try to run inference of our trained model. It is recommended to use the **.tflite** format for raspberrypi. 
If you want to use it in realtime on raspberrypi, run the following command:

    python predict.py --source webcam

Otherwise, you can validate the model using a test dataset that you have already prepared. Copy your test_dataset into the **"face_recogniton/"** directory and run the following command:

    python predict.py --source directory --test_dir <directory containing test images>

This command will show you each image located in the provided **test_dir** with labels and confidence score.

## Conluding Remarks
If you run into any problems, please refer to the scipts provided in the repo. I have created elaborate descriptions of each of the scripts for better debugging.