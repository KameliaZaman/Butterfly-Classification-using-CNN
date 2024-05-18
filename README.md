<a name="readme-top"></a>

<div align="center">
  <img src="https://huggingface.co/KameliaZaman/Butterfly-Classification-Using-CNN/resolve/main/assets/logo.png" alt="Logo" width="400" height="200">

  <h3 align="center">Butterfly Classification using CNN</h3>

  <p align="center">
    Butterfly image classification using ResNet50V2
    <br />
    <a href="https://huggingface.co/spaces/KameliaZaman/Butterfly-Classification-using-CNN">View Demo</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<img src="https://huggingface.co/KameliaZaman/Butterfly-Classification-Using-CNN/resolve/main/assets/About.png" alt="Logo" width=400" height="200">

The project aims to develop a butterfly image classification system utilizing the ResNet50V2 architecture. The goal is to accurately identify different species of butterflies from images, leveraging the deep learning capabilities of ResNet50V2. This involves training the model on a large dataset of butterfly images, fine-tuning its parameters, and optimizing its performance to achieve high accuracy in classifying various butterfly species. Ultimately, the project seeks to provide a reliable tool for researchers, conservationists, and enthusiasts to easily identify and catalog different butterfly species, aiding in biodiversity studies and conservation efforts.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python]][Python-url]
* [![TensorFlow][TensorFlow]][TensorFlow-url]
* [![OpenCV][OpenCV]][OpenCV-url]
* [![NumPy][NumPy]][NumPy-url]
* [![Pandas][Pandas]][Pandas-url]
* [![Matplotlib][Matplotlib]][Matplotlib-url]
* [![Plotly][Plotly]][Plotly-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Please follow these simple steps to setup this project locally.

### Dependencies

Here are the list all libraries, packages and other dependencies that need to be installed to run this project.

For example, this is how you would list them:
* TensorFlow 2.16.1
  ```sh
  conda install -c conda-forge tensorflow
  ```
* OpenCV 4.9.0
  ```sh
  conda install -c conda-forge opencv
  ```
* Gradio 4.24.0
  ```sh
  conda install -c conda-forge gradio
  ```
* NumPy 1.26.4
  ```sh
  conda install -c conda-forge numpy
  ```

### Alternative: Export Environment

Alternatively, clone the project repository, install it and have all dependencies needed.

  ```sh
  conda env export > requirements.txt
  ```

Recreate it using:

  ```sh
  conda env create -f requirements.txt
  ```

### Installation

```sh
# clone project   
git clone https://huggingface.co/spaces/KameliaZaman/Butterfly-Classification-using-CNN/tree/main

# go inside the project directory 
cd Butterfly-Classification-using-CNN

# install the required packages
pip install -r requirements.txt

# run the gradio app
python app.py 
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

#### Dataset

Dataset is from "https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species" which contains train, test and validation sets for 100 butterfly or moth species.

#### Model Architecture

ResNet50V2 was used to to train the model. Adam optimizer was applied with a learning rate of 0.0001. 

<img src="https://huggingface.co/KameliaZaman/Butterfly-Classification-Using-CNN/resolve/main/assets/arch.png" alt="Logo" width="400" height="200">

#### Data Preparation
- The dataset is loaded from a CSV file containing information about the butterflies and moths.
- Image paths are constructed based on the dataset information.
- The dataset is split into training, validation, and test sets.

#### Exploratory Data Analysis (EDA)
- Visualizations are created to explore the distribution of labels in the dataset.

  ```sh
  label_counts = df['labels'].value_counts()[:10]

  fig = px.bar(x=label_counts.index, 
               y=label_counts.values,
               color=label_counts.values,
               text=label_counts.values,
               color_continuous_scale='Blues')
  
  fig.update_layout(
      title_text='Top 10 Labels Distribution',
      template='plotly_white',
      xaxis=dict(
          title='Label',
      ),
      yaxis=dict(
          title='Count',
      )
  )
  
  fig.update_traces(marker_line_color='black', 
                    marker_line_width=1.5, 
                    opacity=0.8)
   
  fig.show()
  ```
  
  <img src="https://huggingface.co/KameliaZaman/Butterfly-Classification-Using-CNN/resolve/main/assets/eda.png" alt="Logo" width="400" height="200">

#### Image Data Generation
- Image data generators are used to augment the training data.
- Training and validation data generators are created.

  ```sh
  train_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=1/255.)
  val_gen = ImageDataGenerator(rescale=1/255.)
  
  BATCH_SIZE = 64
  SEED = 56
  IMAGE_SIZE = (244, 244)
  
  train_flow_gen = train_gen.flow_from_directory(directory=train_dir,
                                                class_mode='sparse',
                                                batch_size=BATCH_SIZE,
                                                target_size=IMAGE_SIZE,
                                                seed=SEED)
  
  val_flow_gen = val_gen.flow_from_directory(directory=val_dir,
                                              class_mode='sparse',
                                              batch_size=BATCH_SIZE,
                                              target_size=IMAGE_SIZE,
                                              seed=SEED)
  ```

#### Model Training
- The ResNet50V2-based model is constructed and compiled.
- The model is trained on the augmented training data, and its performance is monitored using validation data.
- Callbacks for reducing learning rate and early stopping are employed during training.

  ```sh
  resnet_model.fit(train_flow_gen, epochs=15,
         steps_per_epoch=int(np.ceil(train_df.shape[0]/BATCH_SIZE)),
         validation_data=val_flow_gen,
         validation_steps=int(np.ceil(val_df.shape[0]/BATCH_SIZE)),
         callbacks=[rlr_cb, early_cb])
  ```

  <img src="https://huggingface.co/KameliaZaman/Butterfly-Classification-Using-CNN/resolve/main/assets/train_acc.png" alt="Logo" width="400" height="200">

#### Model Evaluation
- The trained model is evaluated on the test set to measure its accuracy.

  <img src="https://huggingface.co/KameliaZaman/Butterfly-Classification-Using-CNN/resolve/main/assets/test_acc.png" alt="Logo" width="400" height="50">

#### Deployment
- Gradio is utilized for deploying the trained model.
- Users can input an image, and the model will predict the butterfly species.

  ```sh
  import gradio as gr
  import tensorflow as tf
  from tensorflow.keras.models import load_model
  import numpy as np
  import cv2
  
  model_path = './model_checkpoint_manual_resnet.h5'
  model = load_model(model_path)
  
  class_names = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ARCIGERA FLOWER MOTH', 'ATALA', 'ATLAS MOTH', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK']
  
  def preprocess_image(img):
      if isinstance(img, str):
          # Load and preprocess the image
          img = cv2.imread(img)
          img = cv2.resize(img, (224, 224))
          img = img / 255.0  # Normalize pixel values
          img = np.expand_dims(img, axis=0)  # Add batch dimension
      elif isinstance(img, np.ndarray):
          img = cv2.resize(img, (224, 224))
          img = img / 255.0  # Normalize pixel values
          img = np.expand_dims(img, axis=0)  # Add batch dimension
      else:
          raise ValueError("Unsupported input type. Please provide a file path or a NumPy array.")  
      return img
  
  def classify_image(img):
      img = preprocess_image(img)
      predictions = model.predict(img)
      predicted_class = np.argmax(predictions)
      predicted_class_name = class_names[predicted_class]
      
      return f"Predicted Class: {predicted_class_name}"
  
  iface = gr.Interface(fn=classify_image, 
                       inputs="image",
                       outputs="text",
                       live=True)
  
  iface.launch()
  ```

  <img src="https://huggingface.co/KameliaZaman/Butterfly-Classification-Using-CNN/resolve/main/assets/About.png" alt="Logo" width="400" height="200">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See [MIT License](LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Kamelia Zaman Moon - kamelia.stu2017@juniv.edu

Project Link: [https://huggingface.co/spaces/KameliaZaman/Butterfly-Classification-using-CNN](https://huggingface.co/spaces/KameliaZaman/Butterfly-Classification-using-CNN/tree/main)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[TensorFlow]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[TensorFlow-url]: https://tensorflow.org/
[OpenCV]: https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/
[Plotly]: https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white
[Plotly-url]: https://plotly.com/
