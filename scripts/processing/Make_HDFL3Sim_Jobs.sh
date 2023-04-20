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
dag_path="${scratch_dir}/job_files/dags/Process_L3Sim_dagman.dag"
touch ${dag_path}

for r in ${runs[@]}; do
    if [ $r == 21002 ]; then
        filelist=(/data/ana/PointSource/muon_level3/sim/IC86.2016/$r/Level3_IC86.2016_NuMu.0$r.0000*.i3*)
        output="Level3_${r}_000000-000099_sim"
    elif [ $r == 21220 ]; then
        filelist=(/data/ana/PointSource/muon_level3/sim/IC86.2016/$r/Level3_IC86.2016_NuMu.0$r.000*.i3*)
        output="Level3_${r}_000000-000999_sim"
    fi

    #Create executable job file
    exec_path="${scratch_dir}/job_files/execs/Process_L3Sim_${r}_exec.sh"
    touch ${exec_path}
    echo "#!/bin/sh" >> ${exec_path}

    #THIS IS THE IMPORTANT LINE TO MAKE CHANGES TO!
    #These arguments will work, but you may want/need to change them for your own purposes...
    #(See README and Do_Trials_Sensitivities_Biases.py for description of options)
    echo "python /data/user/mcampana/analysis/binned_tracks/scripts/processing/level3_i32hdf.py --input ${filelist[@]} --output /data/user/mcampana/analysis/binned_tracks/data/level3/sim/hdf/${output}.hdf5" >> ${exec_path}

    #Create submission job file with generic parameters and 8GB of RAM requested
    sub_path="${scratch_dir}/job_files/subs/Process_L3Sim_${r}_submit.submit"
    touch ${sub_path}
    echo "executable = ${exec_path}" >> ${sub_path}
    echo "output = ${scratch_dir}/outputs/Process_L3Sim_${r}.out" >> ${sub_path}
    echo "error = ${scratch_dir}/errors/Process_L3Sim_${r}.err" >> ${sub_path}
    echo "log = ${scratch_dir}/logs/Process_L3Sim_${r}.log" >> ${sub_path}        
    echo "getenv = true" >> ${sub_path}
    echo "universe = vanilla" >> ${sub_path}
    echo "notifications = never" >> ${sub_path}
    echo "should_transfer_files = YES" >> ${sub_path}
    echo "request_memory = 6000" >> ${sub_path}
    echo "queue 1" >> ${sub_path}

    #Add the job to be submitted into the DAGMAN file
    echo "JOB Process_L3Sim_${r} ${sub_path}" >> ${dag_path}
done

#This is the Submit file. After running Make_Cluster_Jobs.sh, run the below shell file to submit the jobs.
runThis="${scratch_dir}/job_files/SubmitMyJobs_Process_L3Sim.sh"
touch ${runThis}
echo "#!/bin/sh" >> ${runThis}
echo "condor_submit_dag -maxjobs 500 ${dag_path}" >> ${runThis}

#End.
