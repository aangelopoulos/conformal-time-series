<h1 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Conformal PID Control</h1>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">for Time-Series Prediction</h3>

<p align="center">
    <a style="text-decoration:none !important;" href="" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

<p align="center">
    <img width=50% src="https://github.com/aangelopoulos/conformal-time-series/blob/master/media/PID-simplified.svg">
</p>

<p>
This repository is about producing <b>prediction sets</b> in <b>time-series prediction</b>.
    
We take a <b>control systems outlook</b> on performing this task, introducing a method called <a style="text-decoration:none !important;" href="" alt="arXiv">Conformal PID Control</a>.

The method is formally valid in the sense that for any, possibly <i>adversarial sequence</i>, coverage will be guaranteed.
It also includes adaptive conformal prediction as a subset.
</p>

<p align="center"> <b>This codebase makes it easy to extend the methods/add new datasets.</b>
We will describe how to do so below.
</p>

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Getting Started</h3>
<p>
To reproduce the experiments in our paper, clone this repo and run the following code from the root directory.
<pre>
conda create --name pid
pip install -r requirements.txt
cd tests
bash run_tests.sh
bash make_plots.sh
</pre>
</p>

The one exception is the COVID experiment. For that experiment, you must first run the jupyter notebook in <code>conformal-time-series/tests/datasets/covid-ts-proc/statewide
/death-forecasting-perstate-lasso-qr.ipynb</code>.
It requires the <code>deaths.csv</code> data file, which you can download from <a style="text-decoration:none !important;" href="" alt="arXiv">this Drive link</a>.

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Adding New Methods</h3>
The <code>core/methods.py</code> file contains all methods.
Consider the following header as an example:

<pre>
def quantile(
    scores,
    alpha,
    lr,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
</pre>

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Adding New Datasets</h3>
First, download your dataset and put it in <code>tests/datasets</code>.
Then, edit the <code>tests/datasets.py</code> file to add a name for your dataset and some processing code for it. 
Make sure the dataset follows the same standard format as the rest.
Then ...

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Workarounds for Known Bugs</h3>
On M1/M2 Mac, in order to use Prophet, follow the instructions at this link: <code>https://github.com/facebook/prophet/issues/2250</code>.
