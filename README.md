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
It requires the <code>deaths.csv</code> data file, which you can download from <a style="text-decoration:none !important;" href="https://drive.google.com/file/d/1p_l3bKJjypmJDmIZ0tqrwIZWDo8JDjvY/view?usp=sharing" alt="arXiv">this Drive link</a>.

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Adding New Methods</h3>

<h5>Step 1: Defining the method. </h5>
The <code>core/methods.py</code> file contains all methods.
Consider the following method header, for the P controller/quantile tracker, as an example:

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
The first three arguments, <code>scores</code>, <code>alpha</code>, and <code>lr</code>, are <i>required</i> arguments for all methods.
The first argument, <code>scores</code>, expects a numpy array of conformal scores. The second argument, <code>alpha</code>, is the desired miscoverage. Finally, the third argument, <code>lr</code>, is the learning rate. (In our paper, this is $\eta$, and in the language of control, this is $K_p$.)

The rest of the arguments listed are required arguments specific to the given method. The argument <code>ahead</code> determines how many steps ahead the prediction is made --- for example, if <code>ahead=4</code>, that means we are making 4-step-ahead predictions (one step is defined by the resolution of the input array <code>scores</code>). The function of <code>*args</code> and <code>**kwargs</code> is to allow methods to take arguments given in a dictionary form.

All methods should <code>return</code> a dictionary of results that includes the method name and the sequence of $q_{t}$. In the quantile example case, the dictionary should look like the following, where <code>qs</code> is a numpy array of quantiles the same length as <code>scores</code>:
<code>results = {"method": "Quantile", "q" : qs}</code>
Methods that follow this formatting will be able to be processed automatically by our testing infrastructure.

<h5>Step 2: Creating a config file for the testing infrastructure.</h5>
We built our own automated testing infrastructure for online conformal.
The infrastructure spawns a parallel process for every dataset, making it efficient to test one method on all datasets with only one command (the command to run the tests is <code>bash run_tests.sh</code>, and to plot the results is <code>bash make_plots.sh</code>).

The infrastructure works like this.
<ul>
    <li>The user defines a file in <code>tests/configs/</code> describing an experiment, i.e., a dataset name and a combination of methods and settings for each method to run. </li>
    <li>The script <code>tests/run_tests.sh</code> calls <code>tests/base_test.py</code> on every <code>.yaml</code> file in the <code>tests/configs</code> directory.</li>
</ul>

<b>The entry point for testing is <code>tests/base_test.py</code>.</b> 

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Adding New Datasets</h3>
First, download your dataset and put it in <code>tests/datasets</code>.
Then, edit the <code>tests/datasets.py</code> file to add a name for your dataset and some processing code for it. 
Make sure the dataset follows the same standard format as the rest.
Then ...

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Workarounds for Known Bugs</h3>
On M1/M2 Mac, in order to use Prophet, follow the instructions at this link: <code>https://github.com/facebook/prophet/issues/2250</code>.
