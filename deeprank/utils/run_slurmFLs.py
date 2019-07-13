#!/usr/bin/env python
# Li Xue
# 20-Feb-2019 10:50

# Split multiple jobs into batches and submit to cartesius

#INPUT: a file that contains all the jobs, for example,




import sys
import re
import os
import glob
import subprocess
import time
import shutil

logDIR='/projects/0/deeprank/BM5/scripts/slurm/change_BINCLASS'
slurmDIR=logDIR
num_cores = 24 # run 24 cores for each slurm job
batch_size = num_cores # number of jobs per slurm file


def write_slurmscript(all_job_FL, batch_size, slurmDIR, logDIR):

    #- split all_jobs.sh into mutliple files
    subprocess.check_call(f'cp {all_job_FL} {slurmDIR}', shell = True)
    subprocess.check_call(f'split -a 3 -d -l {batch_size} --additional-suffix=".slurm" {slurmDIR}/{all_job_FL} {slurmDIR}/batch', shell=True)

    #-- add slurm header and tail to each file

    batchID = 0
    for slurmFL in glob.glob(f'{slurmDIR}/batch*'):
        logFL = slurmFL + '.out'
        write_slurm_header(slurmFL, batchID, batch_size, logFL)
        write_slurm_tail(slurmFL)
        print(slurmFL + ' generated ')

def submit_slurmscript(slurm_dir, batch_size = 100):
    # submit slurm scripts in batches
    # each batch waits for the previous batch to finish first.
    slu_FLs = glob.glob(slurm_dir + "/*.slurm")

    jobIDs=[]
    newjobIDs=[]
    num = 0
    flag = 0
    for i in range(0,len(slu_FLs)):

        outFL=os.path.splitext(slu_FLs[i])[0] + '.out'

        if os.path.isfile(outFL):
            print(f"{outFL} exists. Skip submitting slurm file.")
            continue

        num = num + 1

        if num <= batch_size:
            command = ['sbatch',  slu_FLs[i] ]
            print (" ".join(command))
            jobID = subprocess.check_output(command)
            jobID = re.findall(r'\d+', str(jobID))
            jobID = jobID[0]
            print (num)
            print (jobID)
            newjobIDs.append(jobID) # these IDs will used for dependency=afterany

        if num >batch_size:
            command=['sbatch', '--dependency=afterany:'+ ":".join(jobIDs), slu_FLs[i] ]
            print (" ".join(command))
            jobID = subprocess.check_output(command)
            jobID = re.findall(r'\d+', str(jobID))
            jobID = jobID[0]
            print (num)
            print (jobID)
            newjobIDs.append(jobID)

        if  num%batch_size ==0:
            print (newjobIDs)
            jobIDs=newjobIDs
            newjobIDs=[]
            print ("------------- new batch --------- \n")

        time.sleep(1)

# def submit_slrumscript(slurmDIR):
#     # submit one by one.
#     # Each job waits for the previous job to finish first.
#     slu_FLs = glob.glob(slurmDIR+"*.h5.slurm")
#
#     jobID_prev=''
#     for slurmFL in slu_FLs:
#         if jobID_prev == '':
#             # submit the first slurm file
#             command=['sbatch', slurmFL]
#             print (command)
#             jobID = subprocess.check_output(command)
#             jobID = parse_jobID(jobID)
#             jobID_prev = jobID
#         else:
#             command=['sbatch', f"--dependency=afterany:{jobID_prev}", slurmFL ]
#             print (command)
#             jobID = subprocess.check_output(command)
#             print(jobID)
#             jobID = parse_jobID(jobID)
#             jobID_prev = jobID
#         time.sleep(5)
#

def parse_jobID(jobID):
    # input: b'Submitted batch job 5442433\n '
    # output: 5442433
    jobID = re.findall(r'\d+', str(jobID))
    jobID = jobID[0]
    return (jobID)

def write_slurm_header(slurmFL, batchID, batch_size, logFL):

    #- 1. prepare the header string
    header=''

    header = header + "#!/usr/bin/bash\n"
    header = header + "#SBATCH -p normal\n"

    jobName = 'batch' + str(batchID) + ".h5"
    header = header + "#SBATCH -J " + jobName + "\n"
    header = header + "#SBATCH -N 1\n"
    header = header + "#SBATCH --ntasks-per-node={num_cores}\n"
    header = header + "#SBATCH -t 04:00:00\n"

    header = header + "#SBATCH -o " + logFL + "\n"
    header = header + "#SBATCH -e " + logFL + "\n"

    common_part = """
    start=`date +%s`

    """
    header = header + common_part


    #- 2. add the header to slurmFL
    f = open(slurmFL,'r')
    content = f.readlines()
    f.close()

    content.insert(0, header)

    f = open(slurmFL,'w')
    f.write(''.join(content))
    f.close()

    print(f"slurm header added to {slurmFL}")


def write_slurm_tail(slurmFL):

    tail = """
    wait
    end=`date +%s`

    runtime=$((end-start))
    echo
    echo "total runtime: $runtime sec"
    """

    f = open(slurmFL,'a+')
    f.write(tail)
    f.close()


if not os.path.isdir(slurmDIR):
    os.makedirs(slurmDIR)


write_slurmscript('all_jobs.sh', batch_size, slurmDIR, logDIR)
#submit_slurmscript(slurmDIR, 100)

