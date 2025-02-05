# 1. Introduction
Modern NLP systems typically employ the methods of deep learning. 
This has the practical consequence that they are so computationally intensive that they need a graphics processing unit (GPU, aka "graphics card") in order to be run.
We use Big Red 200 in order to run our jobs that require GPUs.

Big Red 200 is maintained by a division of the university that maintains large fleets of computers that have GPUs and other high-performance computing devices. 
As IU affiliates, we can use these for our research for free, though it requires some additional setup.

Here, I'll show you how to get set up on Big Red 200 and walk you through running a job.
You can regard this document as a distillation of [the official Big Red 200 docs](https://servicenow.iu.edu/kb?id=kb_article_view&sysparm_article=KB0026317) with some additional content added.
If you're wondering about anything, you can either check there or ask me.

# 2. Using Big Red 200
First, you will need to make an account. Follow the instructions here under ["System access"](https://servicenow.iu.edu/kb?id=kb_article_view&sysparm_article=KB0026317#access).
After that, you can `ssh` into Big Red 200 with your IU username:
```bash
ssh lgessler@bigred200.uits.iu.edu
```

## File System
You can check your current usage with `quota`. 
You have easy access to the following locations on UITS's file system:

1. `/N/u/$USER/BigRed200`: this is your home directory (`~`), limited to 100GB.
2. `/N/slate/$USER`: this is an additional storage area that is only available on request (but it's automatically granted). See [how to request](https://servicenow.iu.edu/kb?id=kb_article_view&sysparm_article=KB0022439#slate).

## Big Red 200
Big Red 200 is a specific collection of computer nodes which all have NVIDIA A100 40GB GPUs. 
You will generally _not_ run your jobs directly on them like you would on your personal computer--instead, you will submit "jobs" which tell a centralized job scheduling system how to run your code, and your job will eventually be taken from the queue to be run.
This helps UITS distribute GPU resources fairly, and it also allows you to potentially run many jobs in parallel.

### Software
Software that requires non-trivial installation on the system is managed using `module`.
By default, almost no packages will be made available to you--instead, you will dynamically load them a la carte using the `module` command.
For example, `module load conda` can be used to load Anaconda and gain access to the `conda` command.

# 3. Tutorial: Fine-tuning RoBERTa

Let's get set up and run a real job on Alpine! 

(Note: I'll be using `vim` to edit files in the terminal, but feel free to use whatever terminal text editor you like.)


## Connecting
First, log in to Big Red 200 via SSH and connect to a compile node:

```bash
ssh $USER@bigred200.uits.iu.edu
```

If you are using Slate, I recommend creating a symlink between so that `cd ~/slate` will take you to `/N/slate/$USER`, though this is not necessary:

```bash
ln -s /N/slate/lgessler slate
```

## Anaconda Configuration
Now we want to [set up Anaconda](https://curc.readthedocs.io/en/latest/software/python.html?highlight=anaconda). 
Use `module` to load Anaconda:

```bash
module add conda
```
Verify that the load was successful by running `python`.

## Environment Setup

Now, let's make a new Python environment and add some dependencies:

```bash
conda create --name hf python=3.10
conda activate hf
pip install "transformers[torch]" datasets evaluate numpy scikit-learn
```

Let's put our code (a simplified version of [this tutorial code from HuggingFace](https://huggingface.co/docs/transformers/en/training)) under the projects folder:

```bash
mkdir ~/hf_demo
cd ~/hf_demo
wget https://raw.githubusercontent.com/lgessler/br200-guide/main/main.py
```

## Slurm Setup
Things are not as simple as running `python main.py` now. 
You are currently logged into a compile node, and compile nodes do not have GPUs.
Instead, you will submit a job to Slurm, a system that manages job submissions.

Slurm is a system that takes specifications of jobs and manages how those jobs are executed. The number of jobs typically far outnumbers the number of GPUs that are available, so a central system needs to control the process of which jobs get released to which nodes at what times.

First, create two directories:

```bash
mkdir ~/logs
mkdir ~/scripts
```

Now, create a new file `scripts/hf_demo.sh` and fill it with the following:


```bash
#!/bin/bash
#SBATCH --account=r00000
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=hf-demo
#SBATCH --output=logs/hf-demo-%j.output

module purge
module load conda 

cd hf_demo
python main.py
```

This is a script for submitting a job.
As you can see, the script looks like a normal `sh` script except for all the commented lines at the top, which are telling Slurm important information about this job. 
Briefly:

* `account`: all jobs are associated with an account (or an "allocation") which tracks the number of **service units** (SUs) that have been used for a particular project. Service units are "device-hours" which you can spend on CPUs and GPUs. Speaking loosely, the more SUs you use, the *lower priority* your jobs will be, meaning some others will be able to queue jobs before you during times of high demand. **NOTE:** the value shown here is a dummy, and you will need to replace it. **If you are a student in one of my classes, send me an email**.
* `nodes`: this is the number of devices needed to run the job. You will always have this set to 1 unless you're doing something advanced.
* `ntasks`: this is the number of CPU cores assigned to your job, and this will also determine the amount of CPU RAM your job will have. Consider decreasing if your job won't need much CPU contribution.
* `time`: the maximum amount of time your job will run for. Your job will be killed after this time is exceeded. The maximum time that can be standardly assigned is 24h.
* `partition`: the "subcluster" of machines that will process your job. For GPU jobs, you will always keep this on `gpu`.
* `gres`: this is used to specify the number of GPUs you need. You will always keep this at one.
* `job-name`: a meaningful human-readable name for your job. Name it after the project you're running the job for.
* `output`: the path to a log file that will receive a pipe of the `stdout` produced by the job. Note that the `%j` in the name will be replaced by the numeric ID of the job, which will prevent collisions across jobs.

## Running with Slurm
Now, run `sbatch scripts/hf_demo.sh`.
This submits your job to Slurm, and it will now wait until a node is available to execute your job.
You can track your job's status by entering `squeue --user=$USER`. 
After it's done, you will no longer see it with `squeue`. 
Your job's output will be visible under `logs/` with the file name you specified.

If all goes well, you should now see output from your first Slurm job!

## Cheatsheet
CURC has a [cheatsheet](https://curc.readthedocs.io/en/stable/additional-resources/CURC-cheatsheet.html) which you may find helpful.
