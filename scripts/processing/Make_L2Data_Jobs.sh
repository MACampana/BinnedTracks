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

years=( 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022)
dates_2012=( 0515 0516 0517 0518 0519 0520 0521 0522 0523 0524 0525 0526 0527 0528 0529 0530 0531 0601 0602 0603 0604 0605 0606 0607 0608 0609 0610 0611 0612 0613 0614 0615 0616 0617 0618 0619 0620 0621 0622 0623 0624 0625 0626 0627 0628 0629 0630 0701 0702 0703 0704 0705 0706 0707 0708 0709 0710 0711 0712 0713 0714 0715 0716 0717 0718 0719 0720 0721 0722 0723 0724 0725 0726 0727 0728 0729 0730 0731 0801 0802 0803 0804 0805 0806 0807 0808 0809 0810 0811 0812 0813 0814 0815 0816 0817 0818 0819 0820 0821 0822 0823 0824 0825 0826 0827 0828 0829 0830 0831 0901 0902 0903 0904 0905 0906 0907 0908 0909 0910 0911 0912 0913 0914 0915 0916 0917 0918 0919 0920 0921 0922 0923 0924 0925 0926 0927 0928 0929 0930 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1101 1102 1103 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 1117 1118 1119 1120 1121 1122 1123 1124 1125 1126 1127 1128 1129 1130 1201 1202 1203 1204 1205 1206 1207 1208 1209 1210 1211 1212 1213 1214 1215 1216 1217 1218 1219 1220 1221 1222 1223 1224 1225 1226 1227 1228 1229 1230 1231 )
dates_full=( 0101 0102 0103 0104 0105 0106 0107 0108 0109 0110 0111 0112 0113 0114 0115 0116 0117 0118 0119 0120 0121 0122 0123 0124 0125 0126 0127 0128 0129 0130 0131 0201 0202 0203 0204 0205 0206 0207 0208 0209 0210 0211 0212 0213 0214 0215 0216 0217 0218 0219 0220 0221 0222 0223 0224 0225 0226 0227 0228 0229 0301 0302 0303 0304 0305 0306 0307 0308 0309 0310 0311 0312 0313 0314 0315 0316 0317 0318 0319 0320 0321 0322 0323 0324 0325 0326 0327 0328 0329 0330 0331 0401 0402 0403 0404 0405 0406 0407 0408 0409 0410 0411 0412 0413 0414 0415 0416 0417 0418 0419 0420 0421 0422 0423 0424 0425 0426 0427 0428 0429 0430 0501 0502 0503 0504 0505 0506 0507 0508 0509 0510 0511 0512 0513 0514 0515 0516 0517 0518 0519 0520 0521 0522 0523 0524 0525 0526 0527 0528 0529 0530 0531 0601 0602 0603 0604 0605 0606 0607 0608 0609 0610 0611 0612 0613 0614 0615 0616 0617 0618 0619 0620 0621 0622 0623 0624 0625 0626 0627 0628 0629 0630 0701 0702 0703 0704 0705 0706 0707 0708 0709 0710 0711 0712 0713 0714 0715 0716 0717 0718 0719 0720 0721 0722 0723 0724 0725 0726 0727 0728 0729 0730 0731 0801 0802 0803 0804 0805 0806 0807 0808 0809 0810 0811 0812 0813 0814 0815 0816 0817 0818 0819 0820 0821 0822 0823 0824 0825 0826 0827 0828 0829 0830 0831 0901 0902 0903 0904 0905 0906 0907 0908 0909 0910 0911 0912 0913 0914 0915 0916 0917 0918 0919 0920 0921 0922 0923 0924 0925 0926 0927 0928 0929 0930 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1101 1102 1103 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 1117 1118 1119 1120 1121 1122 1123 1124 1125 1126 1127 1128 1129 1130 1201 1202 1203 1204 1205 1206 1207 1208 1209 1210 1211 1212 1213 1214 1215 1216 1217 1218 1219 1220 1221 1222 1223 1224 1225 1226 1227 1228 1229 1230 1231 )

#---------------------------------------------

#Create DAGMAN file
dag_path="${scratch_dir}/job_files/dags/Process_L2Data_dagman.dag"
touch ${dag_path}

sub="_IT"
for y in ${years[@]}; do
    if [[ $y == "2012" ]]; then
        dates=${dates_2012[@]}
    else
        dates=${dates_full[@]}
    fi
    for d in ${dates[@]}; do
    
        filelist1=(/data/exp/IceCube/$y/filtered/level2/$d/Run*_*/Level2_IC86*Subrun*.i3.zst)
        filelist2=()
        for f in ${filelist1[@]}; do
            [[ ! $f =~ $sub ]] && filelist2+=($f)
        done
        
        #Create executable job file
        exec_path="${scratch_dir}/job_files/execs/Process_L2Data_${d}${y}_exec.sh"
        touch ${exec_path}
        echo "#!/bin/sh" >> ${exec_path}

        #THIS IS THE IMPORTANT LINE TO MAKE CHANGES TO!
        #These arguments will work, but you may want/need to change them for your own purposes...
        #(See README and Do_Trials_Sensitivities_Biases.py for description of options)
        echo "python /data/user/mcampana/analysis/binned_tracks/scripts/processing/level2.py --input ${filelist2[@]} --output /data/user/mcampana/analysis/binned_tracks/data/level2/exp/npy/Level2_${d}${y}_exp.npy --fix-leap " >> ${exec_path}

        #Create submission job file with generic parameters and 8GB of RAM requested
        sub_path="${scratch_dir}/job_files/subs/Process_L2Data_${d}${y}_submit.submit"
        touch ${sub_path}
        echo "executable = ${exec_path}" >> ${sub_path}
        echo "output = ${scratch_dir}/outputs/Process_L2Data_${d}${y}.out" >> ${sub_path}
        echo "error = ${scratch_dir}/errors/Process_L2Data_${d}${y}.err" >> ${sub_path}
        echo "log = ${scratch_dir}/logs/Process_L2Data_${d}${y}.log" >> ${sub_path}        
        echo "getenv = true" >> ${sub_path}
        echo "universe = vanilla" >> ${sub_path}
        echo "notifications = never" >> ${sub_path}
        echo "should_transfer_files = YES" >> ${sub_path}
        echo "request_memory = 6000" >> ${sub_path}
        echo "queue 1" >> ${sub_path}

        #Add the job to be submitted into the DAGMAN file
        echo "JOB Process_L2Data_${d}${y} ${sub_path}" >> ${dag_path}

    done
done

#This is the Submit file. After running Make_Cluster_Jobs.sh, run the below shell file to submit the jobs.
runThis="${scratch_dir}/job_files/SubmitMyJobs_Process_L2Data.sh"
touch ${runThis}
echo "#!/bin/sh" >> ${runThis}
echo "condor_submit_dag -maxjobs 500 ${dag_path}" >> ${runThis}

#End.
