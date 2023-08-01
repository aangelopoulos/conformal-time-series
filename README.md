<h1 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Conformal PID Control for Time-Series Prediction</h1>

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2307.16895" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

<p align="center">
    <img width=70% src="https://github.com/aangelopoulos/conformal-time-series/blob/main/media/PID-simplified.svg">
</p>

<p>
This repository is about producing <b>prediction sets for time series</b>.

The methods here are guaranteed to have coverage for any, possibly <i>adversarial sequence</i>.
We take a <b>control systems outlook</b> on performing this task, introducing a method called <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2307.16895" alt="arXiv">Conformal PID Control</a>. 

Several methods are implemented herein, including online quantile regression (quantile tracking/P control), adaptive conformal prediction, and more!
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

<h5>Step 2: Edit the config to include your method.</h5>
Tl;Dr: go to each config in <code>tests/configs</code>, and add a line under <code>methods:</code> for each method you want to run, along with what learning rates to test. The below example, from <code>tests/configs/AMZN.yaml</code>, will ask the testing suite to run the quantile tracker on the Amazon stock price dataset with five different learning rate choices.
<pre>
  Quantile:
    lrs:
      - 1
      - 0.5
      - 0.1
      - 0.05
      - 0
</pre>

As background, this is part of our little testing infrastructure for online conformal.
The infrastructure spawns a parallel process for every dataset, making it efficient to test one method on all datasets with only one command (the command to run the tests is <code>bash run_tests.sh</code>, and to plot the results is <code>bash make_plots.sh</code>).

The infrastructure works like this.
<ul>
    <li>The user defines a file in <code>tests/configs/</code> describing an experiment, i.e., a dataset name and a combination of methods and settings for each method to run. </li>
    <li>The script <code>tests/run_tests.sh</code> calls <code>tests/base_test.py</code> on every <code>.yaml</code> file in the <code>tests/configs</code> directory.</li>
    <li>The script <code>tests/make_plots.sh</code> calls <code>inset_plot.py</code></li> and <code>base_plots.py</code> to produce the plots in the main text and appendix of our paper, respectively.
</ul>

<h5>Step 3: Edit <code>base_test.py</code> to include your method.</h5>
The code in <a style="text-decoration:none !important;" href="https://github.com/aangelopoulos/conformal-time-series/blob/e6419ac4345f4a4cad254a76f1f232e815679087/tests/base_test.py#L5C3-L5C3">line 5</a> of <code>base_test.py</code> imports all the methods --- import yours as well.
Then add your method to the big <code>if/else</code> block starting on <a style="text-decoration:none !important;" href="https://github.com/aangelopoulos/conformal-time-series/blob/e6419ac4345f4a4cad254a76f1f232e815679087/tests/base_test.py#L103">line 103</a>.

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Adding New Datasets</h3>

<h5 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Step 1: Load and preprocess the dataset.</h5>

First, download your dataset and put it in <code>tests/datasets</code>.
Then, edit the <code>tests/datasets.py</code> file to add a name for your dataset and some processing code for it. 
The dataset should be a <code>pandas</code> dataframe with a valid <code>datetime</code> index (it has to be evenly spaced, and correctly formatted with no invalid values), and at least one column simply titled <code>y</code>. This column represents the target value.

Alternatively, including a column titled <code>forecasts</code> or <code>scorecasts</code> will cause the infrastructure to use these forecasts/scorecasts instead of the ones it would have produced on its own. This is useful if you have defined a good forecaster/scorecaster outside our framework, and you just want to use our code to run conformal on top of that.
Extra columns can be used to add information for more complex forecasting/scorecasting strategies.

<h5 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Step 2: Create a config file for the dataset.</h5>
As mentioned above, a config file should be made for each dataset, describing what methods should be run with what parameters.
The example of <code>tests/configs/AMZN.yaml</code> can be followed.

After executing these two steps, you should be able to run <code>python base_test.py configs/your_dataset.yaml</code> and the results will be computed!
Alternatively, you can just execute the bash scripts.

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Workarounds for Known Bugs</h3>
On M1/M2 Mac, in order to use Prophet, follow the instructions at this link: <code>https://github.com/facebook/prophet/issues/2250</code>.
