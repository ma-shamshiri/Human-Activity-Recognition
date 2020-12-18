<h1 align="center"> Human Activity Recognition </h1>
<h3 align="center"> A Comparative Study between Different Pre-processing Approaches and Classifiers </h3>

</br>

<p align="center"> 
  <img src="images/Signal22.gif" alt="Animated gif">
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 
  This project aims to classify human activities using data obtained from accelerometer and gyroscope sensors from phone and watch. The raw data will be preprocessed using two
  different approaches such as topological data analysis and statistical features extraction from segmented time series. The goal is to compare and evaluate the performance of
  different classifiers (Decision Tree, k Nearest Neighbors, Random Forest, SVM and CNN) which are trained on the two sets of preprocessed data.
</p>

<p align="center">
  <img src="images/WISDM Activities.png" alt="Table1: 18 Activities" width="70%" height="70%">        
  <figcaption>Caption goes here</figcaption>
</p>

<!-- PREREQUISITES -->
<h2 id="prerequisites"> :pencil: Prerequisites</h2>

This project is written in Python programming language.
The following python packages were used in this project.
* Numpy
* Pandas
* Matplotlib
* Scikit-Learn
* Scikit-tda
* Giotto-tda
* TensorFlow
* Keras

<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
  The WISDM (Wireless Sensor Data Mining) dataset includes raw time-series data collected from accelerometer and gyroscope sensors of a smartphone and smartwatch with their corresponding labels for each activity. The sensor data was collected at a rate of 20 Hz (i.e., every 50ms). Weiss et.al., collected this dataset from 51 subjects who performed 18 different activities listed in Table 2, each for 3 minutes, while having the smartphone in their right pant pocket and wearing the smartwatch in their dominant hand. Each line of the time-series sensor file is considered as input.
 
 _The WISDM dataset is publicly available. Please refer to the [Link](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)_  
 
  The following table shows the 18 activities represented in data set.
</p>

<p align="center">
  <img src="images/activity_table.png" alt="Table1: 18 Activities" width="45%" height="45%">
</p>

<!-- ROADMAP -->
<h2 id="roadmap"> :dart: Roadmap</h2>

<p align="justify"> 
  Weiss et. al. [1], has trained three models namely Decision Tree, k-Nearest Neighbors, and Random Forest for human activity classification by preprocessing the raw time series data using statistical feature extraction from segmented time series. The goals of this project include the following:
<ol>
  <li><a href="#about-the-project">About The Project</a></li>
  <li><a href="#prerequisites">Prerequisites</a></li>
</ol>
</p>

<!-- REFERENCES -->
<h2 id="references"> :books: Refrences</h2>
<ul>
  <li>
    <p>Matthew B. Kennel, Reggie Brown, and Henry D. I. Abarbanel. Determining embedding dimension for phase-space reconstruction using a geometrical construction. Phys. Rev. A, 45:3403–3411, Mar 1992.
    </p>
  </li>
  <li>
    <p>
      L. M. Seversky, S. Davis, and M. Berger. On time-series topological data analysis: New data and opportunities. In 2016 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1014–1022, 2016.
    </p>
  </li>
  <li>
    <p>
      Floris Takens. Detecting strange attractors in turbulence. In David Rand and Lai-Sang Young, editors, Dynamical Systems and Turbulence, Warwick 1980, pages 366–381, Berlin, Heidelberg, 1981. Springer Berlin Heidelberg.
    </p>
  </li>
  <li>
    <p>
      Guillaume Tauzin, Umberto Lupo, Lewis Tunstall, Julian Burella P´erez, Matteo Caorsi, Anibal Medina-Mardones, Alberto Dassatti, and Kathryn Hess. giotto-tda: A topological data analysis toolkit for machine learning and data exploration, 2020.
    </p>
  </li>
  <li>
    <p>
      G. M. Weiss and A. E. O’Neill. Smartphone and smartwatchbased activity recognition. Jul 2019.
    </p>
  </li>
  <li>
    <p>
      G. M. Weiss, K. Yoneda, and T. Hayajneh. Smartphone and smartwatch-based biometrics using activities of daily living. IEEE Access, 7:133190–133202, 2019.
    </p>
  </li>
  <li>
    <p>
      Jian-Bo Yang, Nguyen Nhut, Phyo San, Xiaoli li, and Priyadarsini Shonali. Deep convolutional neural networks on multichannel time series for human activity recognition. IJCAI, 07 2015.
    </p>
  </li>
</ul>
