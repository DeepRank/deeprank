#!/usr/bin/env python
# Li Xue
# 20-Feb-2019
#
# This script generates hdf5 files (input of our network) for a batch of docking cases
# This script is designed for large-scale calculations using slurm.
# This script calls hdf5_generate_premap.py

import sys
import re
import os
import glob
import subprocess
import time
import shutil

batch_size = 1 # the number of cases in one slurm file
decoyDIR='docked_models/' #contains decoys and ref.pdb
pssm_DIR='pssm/'
outDIR='hdf5_premap/'
logDIR='slurm/hdf5_gen'
slurmDIR=logDIR
bashFL_DIR = slurmDIR
SHM_DIR='/dev/shm/'

caseID_FL = 'caseID_dimers.lst'
num_cores = 24 # run 24 cores for each slurm job


def write_bashscript(caseIDs, bashFL_DIR):


    for caseID in caseIDs:

        bashFL = bashFL_DIR + caseID + '.h5.sh'
        f = open(bashFL, 'w')
        f.write("caseID=" + caseID)

        variables=f"""
        s=`date +%s`
        num_cores={num_cores}
        decoyDIR='{decoyDIR}'
        pssm_DIR='{pssm_DIR}'
        outDIR='{outDIR}'
        logDIR='{logDIR}'
        slurmDIR='{slurmDIR}'
        #SHM_DIR='{SHM_DIR}'
        SHM_DIR=$TMPDIR
        SHM_DIR_oneCase=$SHM_DIR/$caseID/
        tmp_pssmDIR=${{SHM_DIR_oneCase}}pssm/

        if [ ! -d $SHM_DIR_oneCase ];then
            mkdir -p $SHM_DIR_oneCase
        fi

        if [ ! -d $tmp_pssmDIR ];then
            mkdir -p $tmp_pssmDIR
        fi

        """
        f.write(variables)


        action = """
        # -- generate hdf5 file for one case ----------------
        #
        cp $pssm_DIR/$caseID.*.pssm $tmp_pssmDIR
        echo "pssm files copied to $tmp_pssmDIR"

        tmp_decoyDIR=$SHM_DIR_oneCase/decoys/
        tmp_outDIR="$SHM_DIR_oneCase/"
        tmp_nativeDIR=$SHM_DIR_oneCase/native/

        cp $decoyDIR/$caseID.tgz $SHM_DIR_oneCase
        cd $SHM_DIR_oneCase
        tar -xzf $caseID.tgz

        #-- copy decoy pdb files to tmp_decoyDIR
        mkdir -p $tmp_decoyDIR
        ln -s $SHM_DIR_oneCase/$caseID/*pdb $tmp_decoyDIR

        #-- copy ref.pdb to the folder of native
        if [ ! -e $tmp_decoyDIR/${caseID}_refe.pdb ];then
            echo
            echo "Error: $tmp_decoyDIR/${caseID}_refe.pdb does not exist!"
            exit
        fi

        mkdir -p $tmp_nativeDIR

        mv $tmp_decoyDIR/${caseID}_refe.pdb $tmp_nativeDIR/$caseID.pdb
        rm $SHM_DIR_oneCase/$caseID.tgz
        echo "$caseID.tgz copied to $tmp_decoyDIR and untared."

        cd /projects/0/deeprank/BM5/scripts
        command="srun /home/lixue1/tools/anaconda3/bin/python hdf5_generate_premap_prealign.py $caseID $tmp_decoyDIR $tmp_nativeDIR $tmp_pssmDIR $tmp_outDIR "
        echo $command
        eval $command

        #rm $outDIR/*${caseID}.hdf5
        cp --no-preserve=ownership $tmp_outDIR/*$caseID.hdf5 $outDIR
#        cp --no-preserve=ownership $tmp_outDIR/*${caseID}_norm.pckl $outDIR

        rm -rf $SHM_DIR_oneCase

        e=`date +%s`
        time=$((e-s))
        echo "copy file and untar runtime: $time sec"
        """
        f.write(action)
        f.close()
        print(bashFL+ ' generated.')

def write_slurmscript(caseIDs, batch_size, slurmDIR, logDIR):

    count = 0
    batchID = 0
    for caseID in caseIDs:
        #caseID='1KXQ\n'
        print(count)

        print("caseID: " + caseID)

        if count >= batch_size:
            batchID = batchID+1
            count = 0
            write_slurm_tail(slurmFL)
            print("write tail")
            print(slurmFL + ' generated ')

        slurmFL=slurmDIR + "batch" + str(batchID) + '.h5.slurm'
        logFL = logDIR  + "batch" + str(batchID) + '.h5.out'

        if count == 0:
            write_slurm_header(slurmFL, batchID, batch_size, logFL)
            print("write header")

        f = open(slurmFL, 'a+')
        bashFL = bashFL_DIR + caseID + '.h5.sh'
        f.write("stdbuf -oL bash " + bashFL + " &\n")
        f.close()

        count =count + 1


    write_slurm_tail(slurmFL)
    print("write tail")
    print(slurmFL + ' generated ')

def submit_slurmscript(slurm_dir, batch_size = 32):
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
    f = open(slurmFL,'w')

    f.write("#!/usr/bin/bash\n")
    f.write("#SBATCH -p normal\n")


    jobName = 'batch' + str(batchID) + ".h5"
    f.write("#SBATCH -J " + jobName + "\n")
    f.write("#SBATCH -N 1\n" )
   # f.write(f"#SBATCH -c {num_cores}\n" )
    f.write(f"#SBATCH --ntasks-per-node={num_cores}\n" )
    f.write("#SBATCH -t 08:00:00\n" )

    f.write("#SBATCH -o " + logFL +'.o' + "\n")
    f.write("#SBATCH -e " + logFL + '.e' + "\n")


    common_part = """
    start=`date +%s`

    """
    f.write(common_part)

    f.close()

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

def read_caseIDs(caseID_FL):
    f = open(caseID_FL,'r')

    caseIDs = []
    for line in f:
        if re.search('^#', line) or re.search('^$',line):
            continue
        caseIDs.append( line.strip() )
    f.close()

    num_cases = len(caseIDs)
    print(f"There are {num_cases} cases in {caseID_FL}")
    return (caseIDs)


if not os.path.isdir(outDIR):
    os.mkdir(outDIR)

if not os.path.isdir(slurmDIR):
    os.makedirs(slurmDIR)


caseIDs = read_caseIDs(caseID_FL)
write_bashscript(caseIDs, bashFL_DIR)
write_slurmscript(caseIDs, batch_size, slurmDIR, logDIR)
submit_slurmscript(slurmDIR, batch_size = 2)

