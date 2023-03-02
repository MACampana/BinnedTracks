#!/bin/sh

scratch_dir="/scratch/mcampana"
base_dir="/data/user/mcampana/analysis/binned_tracks"
#===================================================================================================
#Make directories within scratch for submission.

mkdir "${scratch_dir}/outputs/"
mkdir "${scratch_dir}/errors/"
mkdir "${scratch_dir}/logs/"
mkdir "${scratch_dir}/job_files/"

mkdir "${scratch_dir}/job_files/execs/"

mkdir "${scratch_dir}/job_files/subs/"

mkdir "${scratch_dir}/job_files/dags/"
#---------------------------------------------

#Create DAGMAN file
dag_path="${scratch_dir}/job_files/dags/BinnedTrials_dagman.dag"
touch ${dag_path}

#Script Args
data_path="${base_dir}/data/level2/binned/Level2_2020.binned_data.npy"
sig_path="${base_dir}/data/level2/sim/npy/Level2_sim.npy"
grl_path="${base_dir}/GRL.npy"
savedir="${base_dir}/data/level2/binned"
template_path="${base_dir}/templates/Fermi-LAT_pi0_map.npy"
gamma="2.7"
nside="128"
cutoff="0.1"
min_dec_deg="-80"
max_dec_deg="80"
num_trials="100"

ana_type="allsky"
name="Fermi_pi0_${ana_type}"

nsigs=( 15000 20000 25000 )
for nsig in ${nsigs[@]}; do

    seeds=({0..99..1})
    for s in ${seeds[@]}; do
        #Create executable job file
        exec_path="${scratch_dir}/job_files/execs/BinnedTrials_${ana_type}_${nsig}_${s}_exec.sh"
        touch ${exec_path}
        echo "#!/bin/sh" >> ${exec_path}

        #THIS IS THE IMPORTANT LINE TO MAKE CHANGES TO!
        #These arguments will work, but you may want/need to change them for your own purposes...

        if [[ ${nsig} == "0" ]]; then
            save_trials_dir="${base_dir}/trials/bkg/${ana_type}/cutoff/${cutoff}"
        else
            save_trials_dir="${base_dir}/trials/sig/${ana_type}/cutoff/${cutoff}/nsig/${nsig}"
        fi

        echo "python ${base_dir}/scripts/trials_${ana_type}.py --data-path ${data_path} --is-binned --sig-path ${sig_path} --grl-path ${grl_path} --savedir ${savedir} --name ${name} --template-path ${template_path} --gamma ${gamma} --cutoff ${cutoff} --nside ${nside} --min-dec-deg ${min_dec_deg} --max-dec-deg ${max_dec_deg} --verbose --num-trials ${num_trials} --nsig ${nsig} --seed ${s} --save-trials ${save_trials_dir}" >> ${exec_path}

        #Create submission job file with generic parameters and 8GB of RAM requested
        sub_path="${scratch_dir}/job_files/subs/BinnedTrials_${ana_type}_${nsig}_${s}_submit.submit"
        touch ${sub_path}
        echo "executable = ${exec_path}" >> ${sub_path}
        echo "output = ${scratch_dir}/outputs/BinnedTrials_${ana_type}_${nsig}_${s}.out" >> ${sub_path}
        echo "error = ${scratch_dir}/errors/BinnedTrials_${ana_type}_${nsig}_${s}.err" >> ${sub_path}
        echo "log = ${scratch_dir}/logs/BinnedTrials_${ana_type}_${nsig}_${s}.log" >> ${sub_path}        
        echo "getenv = true" >> ${sub_path}
        echo "universe = vanilla" >> ${sub_path}
        echo "notifications = never" >> ${sub_path}
        echo "should_transfer_files = YES" >> ${sub_path}
        echo "request_memory = 4000" >> ${sub_path}
        echo "queue 1" >> ${sub_path}

        #Add the job to be submitted into the DAGMAN file
        echo "JOB BinnedTrials_${ana_type}_${nsig}_${s} ${sub_path}" >> ${dag_path}

    done
done

#Below is the Submit file. After running this script, run the below shell file to submit the jobs.
runThis="${scratch_dir}/job_files/SubmitMyJobs_BinnedTrials.sh"
touch ${runThis}
echo "#!/bin/sh" >> ${runThis}
echo "condor_submit_dag -maxjobs 500 ${dag_path}" >> ${runThis}

#End.
