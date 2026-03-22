# IITD HPC — Quick Reference Guide

---

## 1. Login

```bash
ssh username@hpc.iitd.ac.in
```

> For GPU login node:
> ```bash
> ssh username@gpu.hpc.iitd.ac.in
> ```

> With graphical apps (MATLAB etc.):
> ```bash
> ssh -X username@hpc.iitd.ac.in
> ```

---

## 2. Initial Setup (First Time Only)

Run this once after your first login:

```bash
source /home/apps/skeleton/oneTimeHPCAccountEnvSetup.sh
```

Or manually:

```bash
cp /home/apps/skeleton/.bashrc $HOME/.bashrc
cp /home/apps/skeleton/.bash_profile $HOME/.bash_profile
```

Secure your directories:

```bash
chmod og-rx $SCRATCH
chmod og-rx $HOME
```

Then logout and login again:

```bash
exit
```

---

## 3. Adding Files to Scratch (from your laptop)

> ⚠️ Run these commands from your **local terminal**, NOT from inside HPC.

Copy a folder to scratch:

```bash
scp -r /path/to/folder username@hpc.iitd.ac.in:~/scratch/
```

Example:

```bash
scp -r ~/parallel-Assignment-2 mcs252117@hpc.iitd.ac.in:~/scratch/
```

Copy back from HPC to laptop:

```bash
scp -r username@hpc.iitd.ac.in:~/scratch/folder /local/path/
```

Check files are there (from inside HPC):

```bash
ls ~/scratch/
```

---

## 4. Loading Modules

See all available modules:

```bash
module avail
```

Load CUDA modules (for GPU jobs):

```bash
module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0
```

Load Anaconda:

```bash
module load apps/anaconda/3
```

---

## 5. Starting a Job (always with walltime)

> ⚠️ Always include walltime to avoid wasting HPC credits.

Basic GPU job:

```bash
qsub -I -P col7880.username.course -N jobname -l select=1:ncpus=1:ngpus=1,walltime=1:00:00
```

**Walltime format: `HH:MM:SS`**

| Walltime | Meaning |
|---|---|
| `0:30:00` | 30 minutes |
| `1:00:00` | 1 hour |
| `2:30:00` | 2.5 hours |

After job starts, load modules then compile and run:

```bash
module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0

nvcc filename.cu -arch=sm_35 -Xcompiler -fopenmp -o output
./output
```

Exit the job when done:

```bash
exit
```

---

## 6. Checking & Deleting Jobs

Check your running/pending jobs:

```bash
qstat -u $USER
```

Delete a job (get JOBID from qstat — 6 digit number before the dot):

```bash
qdel JOBID
```

See which node is allocated to you:

```bash
cat $PBS_NODEFILE
```

---

## 7. Storage

| Location | Use | Size | Persistent? |
|---|---|---|---|
| `/home` | Code, important files | 100 GB | Yes, permanent |
| `/scratch` | Running jobs, large data | 25 TB | Semi (auto-deleted eventually) |

Check disk quota:

```bash
lfs quota -hu $USER /home
lfs quota -hu $USER /scratch
```

Copy important files from scratch to home:

```bash
cp -r ~/scratch/myfolder ~/
```

Delete a folder permanently:

```bash
rm -rf foldername
```

> ⚠️ `rm -rf` is **permanent** — no undo, no recycle bin. Double check before running.

---

## 8. CUDA Out of Memory Fix

If you killed a GPU program mid-run and get `CUDA OUT OF MEMORY` next time:

```bash
nvidia-smi
```

Note the PID, then kill it:

```bash
kill -9 PID
```

---

## 9. Logout

```bash
exit
```

Or press `Ctrl + D`

---

## Help

Email: hpchelp@iitd.ac.in