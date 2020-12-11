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
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>        
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
<p align="justify"> 
  This project aims to classify human activities using data obtained from accelerometer and gyroscope sensors from phone and watch. The raw data will be preprocessed using two
  different approaches such as topological data analysis and statistical features extraction from segmented time series. The goal is to compare and evaluate the performance of
  different classifiers (Decision Tree, k Nearest Neighbors, Random Forest, SVM and CNN) which are trained on the two sets of preprocessed data.
</p>

<p align="center">
  <img src="images/WISDM Activities.png" alt="Table1: 18 Activities" width="50%" height="50%">        
  <figcaption>Caption goes here</figcaption>
</p>

### Prerequisites

The following python packages were used in this project.
* Scikit-Learn
  ```sh
  npm install npm@latest -g
  ```
* Numpy
  ```sh
  npm install npm@latest -g
  ```
* Pandas
  ```sh
  npm install npm@latest -g
  ```
<!-- DATASET -->
<h2 id="dataset"> :books: Dataset</h2>
<p> 
  Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.
  
  _For more examples, please refer to the [Documentation](https://example.com)_
  The following table shows the 18 activities represented in data set.
</p>

<p align="center">
  <img src="images/activity_table.png" alt="Table1: 18 Activities" width="40%" height="40%">
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
