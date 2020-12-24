<h1 align="center"> Human Activity Recognition </h1>
<h3 align="center"> A Comparative Study between Different Pre-processing Approaches and Classifiers </h3>

</br>

<p align="center"> 
  <img src="images/Signal.gif" alt="Sample signal" width="70%" height="70%">
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    ➤ <li><a href="#about-the-project"> ➤ About The Project</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#folder-structure">Folder Structure</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li>
      <a href="#preprocessing">Preprocessing</a>
      <ul>
        <li><a href="#preprocessed-data">Pre-processed data</a></li>
        <li><a href="#statistical-feature">Statistical feature</a></li>
        <li><a href="#topological-feature">Topological feature</a></li>
      </ul>
    </li>
    <!--<li><a href="#experiments">Experiments</a></li>-->
    <li><a href="#results-and-discussion">Results and Discussion</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#contributors">Contributors</a></li>
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
  <!--figcaption>Caption goes here</figcaption-->
</p>

<!-- PREREQUISITES -->
<h2 id="prerequisites"> :fork_and_knife: Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>

<!--This project is written in Python programming language. <br>-->
The following python packages are used in this project:
* Numpy
* Pandas
* Matplotlib
* Scikit-Learn
* Scikit-tda
* Giotto-tda
* TensorFlow
* Keras

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> :cactus: Folder Structure</h2>

    code
    .
    │
    ├── data
    │   ├── raw_data
    │   │   ├── phone
    │   │   │   ├── accel
    │   │   │   └── gyro
    │   │   ├── watch
    │   │       ├── accel
    │   │       └── gyro
    │   │
    │   ├── transformed_data
    │   │   ├── phone
    │   │   │   ├── accel
    │   │   │   └── gyro
    │   │   ├── watch
    │   │       ├── accel
    │   │       └── gyro
    │   │
    │   ├── feature_label_tables
    │   │    ├── feature_phone_accel
    │   │    ├── feature_phone_gyro
    │   │    ├── feature_watch_accel
    │   │    ├── feature_watch_gyro
    │   │
    │   ├── wisdm-dataset
    │        ├── raw
    │        │   ├── phone
    │        │   ├── accel
    │        │   └── gyro
    │        ├── watch
    │            ├── accel
    │            └── gyro
    │
    ├── CNN_Impersonal_TransformedData.ipynb
    ├── CNN_Personal_TransformedData.ipynb  
    ├── CNN_Impersonal_RawData.ipynb    
    ├── CNN_Personal_RawData.ipynb 
    ├── Classifier_SVM_Personal.ipynb
    ├── Classifier_SVM_Impersonal.ipynb
    ├── statistical_analysis_time_domain.py
    ├── Topological data analysis.ipynb  

<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
  The WISDM (Wireless Sensor Data Mining) dataset includes raw time-series data collected from accelerometer and gyroscope sensors of a smartphone and smartwatch with their corresponding labels for each activity. The sensor data was collected at a rate of 20 Hz (i.e., every 50ms). Weiss et.al., collected this dataset from 51 subjects who performed 18 different activities listed in Table 2, each for 3 minutes, while having the smartphone in their right pant pocket and wearing the smartwatch in their dominant hand. Each line of the time-series sensor file is considered as input.

<p align="center">
  <img src="images/Human Activity.gif" alt="Human Activity.gif" display="inline-block" width="60%" height="50%">
</p>


 _The WISDM dataset is publicly available. Please refer to the [Link](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)_ 

  The following table shows the 18 activities represented in data set.
</p>

<p align="center">
  <img src="images/Activity Table.png" alt="Table1: 18 Activities" width="45%" height="45%">
</p>

<!-- ROADMAP -->
<h2 id="roadmap"> :dart: Roadmap</h2>

<p align="justify"> 
  Weiss et. al. has trained three models namely Decision Tree, k-Nearest Neighbors, and Random Forest for human activity classification by preprocessing the raw time series data using statistical feature extraction from segmented time series. 
  The goals of this project include the following:
<ol>
  <li>
    <p align="justify"> 
      Train the same models - Decision Tree, k Nearest Neighbors, and Random Forest using the preprocessed data obtained from topological data analysis and compare the
      performance against the results obtained by Weiss et. al.
    </p>
  </li>
  <li>
    <p align="justify"> 
      Train SVM and CNN using the preprocessed data generated by Weiss et. al. and evaluate the performance against their Decision Tree, k Nearest Neighbors, and Random Forest models.
    </p>
  </li>
</ol>
</p>

<!-- PREPROCESSING -->
<h2 id="preprocessing"> :hammer: Preprocessing</h2>

<p align="justify"> 
  The WISDM (Wireless Sensor Data Mining) dataset includes raw time-series data collected from accelerometer and gyroscope sensors of a smartphone and smartwatch with their corresponding labels for each activity. The sensor data was collected at a rate of 20 Hz (i.e., every 50ms). Weiss et.al., collected this dataset from 51 subjects who performed 18 different activities listed in the previous table, each for 3 minutes, while having the smartphone in their right pant pocket and wearing the smartwatch in their dominant hand. <br>
  In this project we tried three different feature sets, extracted from the raw data, which are as follows: 
  <ol>
    <li><b>Pre-processed data</b> generated by Weiss et. al.</li> 
    <li><b>Statistical feature extraction</b></li>
    <li><b>Topological feature extraction</b></li>
  </ol>
  
All these three approaches used windowing technique to segment the raw time series and extract features from each segment. <br> <br>
<b>Windowing technique</b>: A single input from time series data is meaningless, however, a consecutive subset of data points in a signal is equivalent to a specific duration in time and it carries information. Sliding window technique is commonly used in timeseries data to generate data points. Using this technique the raw data is split into segments of size x and stride x to avoid overlaps. The choice of window size is important as different size can carry varying amounts of information.
</p>


<!-- PRE-PROCESSED DATA -->
<h2 id="preprocessed-data"> :diamond_shape_with_a_dot_inside: Pre-processed data</h2>

<p align="justify"> 
  Weiss et.al used windowing technique with window size of 10 seconds to extract statistical features. They extracted 93 features out of which 43 were used to train their models. We also used the same 43 features to train our SVM and CNN. The 43 features are 1. average sensor value 2. standard deviation 3. absolute difference 4. average resultant acceleration 5. Binned distribution (10 equal sized bins per axis) and 5. time between peaks, for each axis.
</p>

<!-- STATISTICAL FEATURE -->
<h2 id="statistical-feature"> :large_orange_diamond: Statistical feature</h2>

<p align="justify"> 
  For this approach, we segmented the dataset using 10 second window size (200 datapoints) with no overlapping. We decided to keep the window size same as whatWeiss et.al. applied in their study, for the sake of comparison. After segmentation, for each segment we calculated eight statistical features, namely, ‘min’, ‘max’, ‘mean’, ‘standard deviation’, ‘median’, ‘variance’, ‘zero crossing’ and ‘mean crossing’, for each axes. The zero and mean crossing features are calculated by counting the rate of when a signal passes line y=0 (if we let y-axis to be the specific measurement and x-axis to represent time) and the frequency at which the signal passes the line y = mean(signal), respectively. However, these two features did not show a significant difference between different activities, so we decided to ignore them.
</p>
  
<!-- TOPOLOGICAL FEATURE -->
<h2 id="topological-feature"> :large_blue_diamond: Topological feature</h2>

<p align="justify"> 
  Topological data analysis provides various techniques toexplore the topological properties and shape of the data.
  Since time series sensor data obtained from performing an activity may have topological properties, we tried extracting features using the topology of the data and perform the classification task on those features. For a given time segment we explore the topology of each segment using persistence diagram generated via persistence homology. Persistent homology can be created through filtrations such as Vietoris- Rips, SparseRips, Cubical Persistence etc., on the data and capture the growth, birth, and death of different topological features across dimensions (e.g., components, tunnels, voids) [2]. One of the main challenges in computing the persistent homology is finding the appropriate filtration for the time segments. In total 18 topological features where extracted for each time segment.
</p>

<!-- EXPERIMENTS -->
<!--<h2 id="experiments"> :microscope: Experiments</h2>-->



<!-- RESULTS AND DISCUSSION -->
<h2 id="results-and-discussion"> :mag: Results and Discussion</h2>

<p align="justify">
  The overall accuracy score of personal and impersonal models are shown in Table 3 and 4. Some of the results we observed are similar to the results obtained by Weiss et.al and they are discussed below: <br>
</p>
<p align="justify">
<ul>
  <li>
    Since accelerometers senses acceleration based on vibration which can be more prominent during an activity and gyroscope only senses rotational changes, accelerometers outperformed gyroscope in all our models. <br>
  </li>
  <li>
    As the style of performing an activity differs from each person, it is difficult to aggregate those features among all subjects. So our personal models vastly outperformed our impersonal models.
  </li>
  <li>
    It is also observed that non hand-oriented activities are classified better with sensors from smartphone and handoriented activities are classified better with sensors from smartwatch. Refer appendix for activity wise recall scores. Some key take-aways based on our results are listed below:
  </li>
  <li>
    CNN trained on raw sensor data performed better for personal models, however it performed poorly on impersonal models.
  </li>
</ul>
</p>
<!-- REFERENCES -->
<h2 id="references"> :books: References</h2>

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

<!-- CONTRIBUTORS -->
<h2 id="contributors"> :scroll: Contributors</h2>

<p>
  :mortar_board: <i>All participants in this project are graduate students in the <a href="https://www.concordia.ca/ginacody/computer-science-software-eng.html">Department of Computer Science and Software Engineering</a> <b>@</b> <a href="https://www.concordia.ca/">Concordia University</a></i> <br> <br>
  
  :woman: <b>Divya Bhagavathiappan Shiva</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>divya.bhagavathiappanshiva@mail.concordia.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/divyabhagavathiappan">@divyabhagavathiappan</a> <br>
  
  :woman: <b>Reethu Navale</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>reethu.navale@mail.concordia.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/reethunavale">@reethunavale</a> <br>

  :woman: <b>Mahsa Sadat Afzali Arani</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>mahsasadat.afzaliarani@mail.concordia.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/MahsaAfzali">@MahsaAfzali</a> <br>

  :boy: <b>Mohammad Amin Shamshiri</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>mohammadamin.shamshiri@mail.concordia.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/ma-shamshiri">@ma-shamshiri</a> <br>
</p>

<br>
✤ <i>This was the final project for the course COMP 6321 - Machine Learning (Fall 2020), at <a href="https://www.concordia.ca/">Concordia University</a><i>
