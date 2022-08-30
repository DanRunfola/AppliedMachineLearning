# AppliedMachineLearning

## Version Notes
This course uses some specific numpy and sklearn versions, and as commands regularly change
I cannot guarantee that the lab videos will work with different versions.
These are contained in requirements.txt, and after you activate your venv can be installed
with 
python -m pip install -r requirements.txt

## Development Environment Notes
This course was developed with Python==3.5.4

You can use jupyter notebooks (recommended if you are not a common user of python)
or python files (*.py) directly in this course.

The videos are all using Jupyter notebooks, so I will assume you are operating in a notebook environment.

This course largely uses Python 3.5, which is compatible with other binary files required.
You can, of course, use a later version, but please note lots of things may break and I will
not google with you to figure out why :)

### Installation Notes for Most Everyone
Generally, it will be easiest to use Anaconda to manage your installations.
Anaconda is a software suite that has a number of different modules, including
modules that help you manage python environments.

It can be downloaded at anaconda.com.

Once installed, confirm you can run conda from your command line environment:
conda -V 
(This example set is run using conda 4.12.0).

To create our new environment, run:
conda create -n conda_aML python=3.5

Activate the environment:
conda activate conda_aML

Install requirements:
conda install scikit-learn==0.19.2
conda install pandas==0.23.4

Once you've installed all the pre-reqs, you can use python to run directly, or jump into a Jupyter
notebook:
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
jupyter notebook

Note: in some cases, you may need to run conda install nbconvert=5.4.1

### Installation Notes for the Brave
I will not help you with this.  Godspeed.

On ubuntu or other debian based systems, you can compile 3.5 using:
sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev openssl 
wget https://www.python.org/ftp/python/3.5.4/Python-3.5.4.tgz
tar xzvf Python-3.5.4.tgz
cd Python-3.5.4
./configure
make
sudo make install

Note, you may also need to install pip manually in this case:
python3.5 -m ensurepip

You can also specify 3.5.* using an anaconda environment.

One of the most important things is to ensure your "kernel" is configured as you expect. 
A kernel defines the version of python (and accompanying packages) you will be using 
to execute your code.

I use a program called "venv" to configure my python kernel.  You may prefer anaconda, as it will
handle python versioning for you, but abstracts away some other things.
To use venv, I:
1) Install python3.5 (see the above - for this course, make sure you're in py3.5 world).
2) Run python3.5 -m venv /home/dan/envs/aML  
In the above command, the path (/home/dan/envs/aML) is a path to a folder where I want to put a fresh copy of python.
You may need to install some other packages depending on your system; in most cases it will tell you.
3) Run source /home/dan/envs/aML/bin/activate
After running that line, you will see (aML) in front of your command prompt.

After you setup your venv, you should be able to select it as a kernel in jupyter.

Note that depending on your Jupyter install, you may need to register the new venv. 
In vscode, this is controlled by the setting: python.venvPath

