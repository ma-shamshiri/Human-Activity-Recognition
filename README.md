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
    <li><a href="#acknowledgements">Acknowledgements</a></li>
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
## Dataset
<p> 
  Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.
  _For more examples, please refer to the [Documentation](https://example.com)_
  The following table shows the 18 activities represented in data set.
</p>

<p align="center">
  <img src="images/activity_table.png" alt="Table1: 18 Activities" width="40%" height="40%">        
  <figcaption>Caption goes here</figcaption>
</p>

<!-- REFERENCES -->
## References
<h2> :books: Refrences</h2>
<ul>
  <li>
    <p>Matthew B. Kennel, Reggie Brown, and Henry D. I. Abarbanel. Determining embedding dimension for phase-space reconstruction using a geometrical construction. Phys. Rev. A, 45:3403–3411, Mar 1992.
    </p>
  </li>
  <li><p>Wikipedia.org, 'Naive Bayes Classifier'. [Online].</p>
      <p>Available: https://en.wikipedia.org/wiki/Naive_Bayes_classifier</p>
  </li>
  <li><p>Youtube.com, 'Naive Bayes for Spam Detection'. [Online].</p>
      <p>Available: https://www.youtube.com/watch?v=8aZNAmWKGfs</p>
  </li>
  <li><p>Youtube.com, 'Text Classification Using Naive Bayes'. [Online].</p>
      <p>Available: https://www.youtube.com/watch?v=EGKeC2S44Rs</p>
  </li>
  <li><p>Manisha-sirsat.blogspot.com, 'What is Confusion Matrix and Advanced Classification Metrics?'. [Online].</p>
      <p>Available: https://manisha-sirsat.blogspot.com/2019/04/confusion-matrix.html</p>
  </li>
  <li><p>Pythonforengineers.com, 'Build a Spam Filter'. [Online].</p>
      <p>Available: https://www.pythonforengineers.com/build-a-spam-filter/</p>
  </li>
</ul>
