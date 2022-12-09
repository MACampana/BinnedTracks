#!/bin/sh

scratch_dir="/scratch/mcampana"

#===================================================================================================
#Make directories within scratch for submission.

mkdir "${scratch_dir}/outputs/"
mkdir "${scratch_dir}/errors/"
mkdir "${scratch_dir}/logs/"
mkdir "${scratch_dir}/job_files/"

mkdir "${scratch_dir}/job_files/execs/"

mkdir "${scratch_dir}/job_files/subs/"

mkdir "${scratch_dir}/job_files/dags/"

runs=( 21002 21220 )
#---------------------------------------------

#Create DAGMAN file
dag_path="${scratch_dir}/job_files/dags/Process_L2Sim_dagman.dag"
touch ${dag_path}

filelist=(/data/user/mcampana/analysis/binned_tracks/data/level2/sim/hdf/*.hdf5)

#Create executable job file
exec_path="${scratch_dir}/job_files/execs/Process_L2Sim_${d}${y}_exec.sh"
touch ${exec_path}
echo "#!/bin/sh" >> ${exec_path}

#THIS IS THE IMPORTANT LINE TO MAKE CHANGES TO!
#These arguments will work, but you may want/need to change them for your own purposes...
#(See README and Do_Trials_Sensitivities_Biases.py for description of options)
echo "python /data/user/mcampana/analysis/binned_tracks/level2.py --input ${filelist[@]} --output /data/user/mcampana/analysis/binned_tracks/data/level2/sim/npy/Level2_21002_21220_sim.npy --fix-leap --hdf --MC --time 57388 --ow-file /data/user/mcampana/analysis/binned_tracks/data/level2/sim/ows.npy" >> ${exec_path}

#Create submission job file with generic parameters and 8GB of RAM requested
sub_path="${scratch_dir}/job_files/subs/Process_L2Sim_${d}${y}_submit.submit"
touch ${sub_path}
echo "executable = ${exec_path}" >> ${sub_path}
echo "output = ${scratch_dir}/outputs/Process_L2Sim_${d}${y}.out" >> ${sub_path}
echo "error = ${scratch_dir}/errors/Process_L2Sim_${d}${y}.err" >> ${sub_path}
echo "log = ${scratch_dir}/logs/Process_L2Sim_${d}${y}.log" >> ${sub_path}        
echo "getenv = true" >> ${sub_path}
echo "universe = vanilla" >> ${sub_path}
echo "notifications = never" >> ${sub_path}
echo "should_transfer_files = YES" >> ${sub_path}
echo "request_memory = 6000" >> ${sub_path}
echo "queue 1" >> ${sub_path}

#Add the job to be submitted into the DAGMAN file
echo "JOB Process_L2Sim_${d}${y} ${sub_path}" >> ${dag_path}

#This is the Submit file. After running Make_Cluster_Jobs.sh, run the below shell file to submit the jobs.
runThis="${scratch_dir}/job_files/SubmitMyJobs_Process_L2Sim.sh"
touch ${runThis}
echo "#!/bin/sh" >> ${runThis}
echo "condor_submit_dag -maxjobs 500 ${dag_path}" >> ${runThis}

#End.