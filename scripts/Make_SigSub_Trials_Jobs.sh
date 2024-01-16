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

#Top Level
num_bins="50"
ana_type="10yr_50EqStatsBins_DirectHistBGSig_GaussFilter_OnePixNoSmear_SigSub"
level="3"
name="Fermi_pi0_${ana_type}_level${level}"

#Module Args
gamma="2.7"
nside="32"
min_dec_deg="-80"
max_dec_deg="80"

num_trials="100"

#data_path="${base_dir}/data/level${level}/binned/Level${level}_10yr_${num_bins}bins.binned_data.nside${nside}.npy"
data_path="${base_dir}/data/level${level}/binned/Level${level}_10yr_${num_bins}EqStatsBins.binned_data.nside${nside}.npy"
#data_path="${base_dir}/data/level${level}/binned/2020/Level${level}_2020_${num_bins}bins.binned_data.nside${nside}.npy"
#data_path="${base_dir}/test/sin_binned_data_25bins.nside32.npy"
sig_path="${base_dir}/data/level${level}/sim/npy/Level${level}_sim_NuNubarWeights.npy"
grl_path="${base_dir}/GRL.npy"
savedir="${base_dir}/data/level${level}/binned"
#template_path="${base_dir}/templates/Fermi-LAT_pi0_map.npy"
template_path="${base_dir}/test/templates/OnePix32.npy"
#kde_path="${base_dir}/data/level${level}/binned/kdes/Level3_10yr_25bins.kde_pdfs_0.nside32.npy"

#Create DAGMAN file
dag_path="${scratch_dir}/job_files/dags/BinnedTrials_${ana_type}_dagman.dag"
touch ${dag_path}

nsigs=( 0 3000 )
for nsig in ${nsigs[@]}; do

    seeds=({0..99..1})
    for s in ${seeds[@]}; do
        #Create executable job file
        exec_path="${scratch_dir}/job_files/execs/BinnedTrials_${ana_type}_${nsig}_${s}_exec.sh"
        touch ${exec_path}
        echo "#!/bin/sh" >> ${exec_path}


        if [[ ${nsig} == "0" ]]; then
            save_trials_dir="${base_dir}/trials/level${level}/${ana_type}/bkg/nside/${nside}/gamma/${gamma}"
        else
            save_trials_dir="${base_dir}/trials/level${level}/${ana_type}/sig/nside/${nside}/gamma/${gamma}/nsig/${nsig}"
        fi

        echo "python ${base_dir}/scripts/sigsub_trials.py --data-path ${data_path} --is-binned --sig-path ${sig_path} --grl-path ${grl_path} --savedir ${savedir} --name ${name} --template-path ${template_path} --gamma ${gamma} --nside ${nside} --min-dec-deg ${min_dec_deg} --max-dec-deg ${max_dec_deg} --verbose --num-trials ${num_trials} --nsig ${nsig} --seed ${s} --save-trials ${save_trials_dir} --qtot --force --sigsub --poisson " >> ${exec_path}

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
        echo "request_memory = 6000" >> ${sub_path}
        echo "queue 1" >> ${sub_path}

        #Add the job to be submitted into the DAGMAN file
        echo "JOB BinnedTrials_${ana_type}_${nsig}_${s} ${sub_path}" >> ${dag_path}

    done
done

#Below is the Submit file. After running this script, run the below shell file to submit the jobs.
runThis="${scratch_dir}/job_files/SubmitMyJobs_BinnedTrials_${ana_type}.sh"
touch ${runThis}
echo "#!/bin/sh" >> ${runThis}
echo "condor_submit_dag -maxjobs 500 ${dag_path}" >> ${runThis}

#End.
