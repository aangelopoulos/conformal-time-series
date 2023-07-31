<h1 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Conformal PID Control</h1>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">for Time-Series Prediction</h3>

<p align="center">
    <a style="text-decoration:none !important;" href="" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

<p align="center">
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img width=50% src="https://github.com/aangelopoulos/conformal-classification/blob/master/media/pid-simplified.svg"></a>
</p>

<p>
This repository is meant to make it easy to understand and extend existing methods for time-series conformal prediction. 
We focus on methods that guarantee coverage in the adversarial sequence setting, where the time series is potentially adversarial. 
It also reproduces the experiments from our paper, Conformal Scorecasting.
</p>

<p align="center"> <b>We make it easy to extend methods/add new datasets.</b></p>

<p>
The only thing you need to do is install the dependencies in the <code>environment.yml</code> file, then run
<code>cd tests
bash run_tests.sh
</code>
Then to generate the plots, run
<code>
bash make_plots.sh
</code>
</p>

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Adding New Methods</h3>
The <code>core/methods.py</code> file contains all methods.
Define a new method there, using the same template as the rest.
Then ... 

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Adding New Datasets</h3>
First, download your dataset and put it in <code>tests/datasets</code>.
Then, edit the <code>tests/datasets.py</code> file to add a name for your dataset and some processing code for it. 
Make sure the dataset follows the same standard format as the rest.
Then ...

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Workarounds for Known Bugs</h3>
On M1/M2 Mac, in order to use Prophet, follow the instructions at this link: <code>https://github.com/facebook/prophet/issues/2250</code>.
