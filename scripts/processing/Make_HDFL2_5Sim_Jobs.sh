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
dag_path="${scratch_dir}/job_files/dags/Process_L2_5Sim_dagman.dag"
touch ${dag_path}

for r in ${runs[@]}; do

    if [ $r == 21002 ]; then
        filelist=(/data/sim/IceCube/2016/filtered/level2/neutrino-generator/$r/0000000-0000999/Level2_IC86.2016_NuMu.0$r.000[0-1]*.i3*)
        output="Level2_5_${r}_000000-000199_sim"
    elif [ $r == 21220 ]; then
        filelist=(/data/sim/IceCube/2016/filtered/level2/neutrino-generator/$r/0001000-0001999/Level2_IC86.2016_NuMu.0$r.00*.i3* /data/sim/IceCube/2016/filtered/level2/neutrino-generator/$r/0001000-0001999/Level2_IC86.2016_NuMu.0$r.00*.i3*)
        output="Level2_5_${r}_000000-001999_sim"
    fi

    #Create executable job file
    exec_path="${scratch_dir}/job_files/execs/Process_L2_5Sim_${r}_exec.sh"
    touch ${exec_path}
    echo "#!/bin/sh" >> ${exec_path}

    #THIS IS THE IMPORTANT LINE TO MAKE CHANGES TO!
    #These arguments will work, but you may want/need to change them for your own purposes...
    #(See README and Do_Trials_Sensitivities_Biases.py for description of options)
    echo "python /data/user/mcampana/analysis/binned_tracks/scripts/processing/level2_5_i32hdf.py --input ${filelist[@]} --output /data/user/mcampana/analysis/binned_tracks/data/level2_5/sim/hdf/${output}.hdf5" >> ${exec_path}

    #Create submission job file with generic parameters and 8GB of RAM requested
    sub_path="${scratch_dir}/job_files/subs/Process_L2_5Sim_${r}_submit.submit"
    touch ${sub_path}
    echo "executable = ${exec_path}" >> ${sub_path}
    echo "output = ${scratch_dir}/outputs/Process_L2_5Sim_${r}.out" >> ${sub_path}
    echo "error = ${scratch_dir}/errors/Process_L2_5Sim_${r}.err" >> ${sub_path}
    echo "log = ${scratch_dir}/logs/Process_L2_5Sim_${r}.log" >> ${sub_path}        
    echo "getenv = true" >> ${sub_path}
    echo "universe = vanilla" >> ${sub_path}
    echo "notifications = never" >> ${sub_path}
    echo "should_transfer_files = YES" >> ${sub_path}
    echo "request_memory = 6000" >> ${sub_path}
    echo "queue 1" >> ${sub_path}

    #Add the job to be submitted into the DAGMAN file
    echo "JOB Process_L2_5Sim_${r} ${sub_path}" >> ${dag_path}
done

#This is the Submit file. After running Make_Cluster_Jobs.sh, run the below shell file to submit the jobs.
runThis="${scratch_dir}/job_files/SubmitMyJobs_Process_L2_5Sim.sh"
touch ${runThis}
echo "#!/bin/sh" >> ${runThis}
echo "condor_submit_dag -maxjobs 500 ${dag_path}" >> ${runThis}

#End.
