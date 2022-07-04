
<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">RDHPO</h3>

  <p align="center">
    Reliable Distributed Hyper Parameter Optimization for Deep Learning
    <br />
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Abstract
Training and validation of Neural Networks (NN) are very computationally intensive. In this project, we want to implement a NN infrastructure to accelerate model training, specifically for hyper-parameter optimization (HPO). By accelerating model training, we can obtain a large set of potential models and compare them in a shorter amount of time. Automating this process reduces development time and provides an easy way to compare different classifiers or hyper-parameters. Our application will run different classifiers on different servers with a single training data set, each with tweaked hyper-parameters. In previous work, we let the user select a range for each hyper-parameter they desired to tune and created the different models based on a black-box exhaustive search architecture. We want to explore better and more efficient ways of creating these different models like Bayesian, multi-fidelity, and population solutions for this project. To give more control over the automation process, the user sets the degree to which these hyper-parameters will be tweaked. Since our solution is a distributed system, we make our implementation robust to common distributed system failures (servers going down, loss of communication among some nodes, and others). We use a gossip-style heartbeat protocol for failure detection and recovery. Some preliminary results using a black-box approach to generate the training models show that our infrastructure times improved by a factor of 15.



### Built With

All major languages and libraries used are listed below:

* [GoLang](https://go.dev/)
* [Python](https://www.python.org/)
* [Gorgonia](https://gorgonia.org/)
* [GoNet](https://github.com/dathoangnd/gonet)
* [Goptuna](https://github.com/c-bata/goptuna)
* [andlabs UI](https://github.com/andlabs/ui)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Installation
1. Make sure GoLang is at least version 1.17.8 (This has not been tested with any earlier versions)
```sh
  go version
  ```
2. Install required packages:
```sh
  go get github.com/andlabs/ui github.com/andlabs/ui/winmanifest github.com/c-bata/goptuna github.com/c-bata/goptuna/successivehalving github.com/c-bata/goptuna/tpe golang.org/x/sync/errgroup github.com/dathoangnd/gonet
  ```
3. Clone the repo
```sh
   git clone https://github.com/jziyangli/xsede_empower.git
   ```
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Running
```sh
  cd xsede_empower/main && go run App.go
  ```
### Modifying
This application is currently set up to optimize training MNIST using the Gonet library.
To use with other machine learning libraries such as Gorgoni or for optimizing other functions, the objective function can be modified:
```go
  func  objective(trial goptuna.Trial) (float64, error)
  ```
It might also be useful to modify to the parseCSV function:
```go
  func  parseCSV(path string) [][][]float64
  ```
  
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

John Li - jzl011@ucsd.edu

Paper Link: 

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Thanks to my mentor Maria Pantoja for guiding me through this project.
Additional thanks to the XSEDE EMPOWER program for funding my research and conferences.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
