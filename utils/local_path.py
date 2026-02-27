import os 
import glob 


def get_inhouse_data_path():
    all_dirs = sorted(glob.glob("/share/project/zhaohuxing/data/inhouse数据/inhouse/number/*/*"))

    # print(f"inhouse data is {len(all_dirs)}")
    # exit(0)
    data_dicts = []
    for each_case in all_dirs:
        print(f"each case is {each_case}")

        if len(glob.glob(os.path.join(each_case, "*.nii"))) < 4:
            continue
        try:
            t1n = glob.glob(os.path.join(each_case, "T1.nii"))[0]
            t2w = glob.glob(os.path.join(each_case, "T2.nii"))[0]
            t1c = glob.glob(os.path.join(each_case, "T1C.nii"))[0]
            t2f = glob.glob(os.path.join(each_case, "Flair.nii"))[0]
        except:
            print(f"error in {each_case}")
            continue
        data_dicts.append({
            "t1n": t1n,
            "t1c": t1c,
            "t2w": t2w,
            "t2f": t2f,
        })

    len_train = int(0.8 * len(data_dicts))
    train_files, val_files = data_dicts[:len_train], data_dicts[len_train:]

    return train_files, val_files

def get_ucsf_data_path():
    # only for inference and external validation
    all_dirs = sorted(glob.glob("/mnt/swarm_beta/xuewei/data/UCSF-PDGM/*"))

    data_dicts = []
    for each_case in all_dirs:
        print(f"each case is {each_case}")

        if len(glob.glob(os.path.join(each_case, "*.nii.gz"))) < 4:
            continue
        
        t1n = glob.glob(os.path.join(each_case, "*_T1_bias.nii.gz"))[0]
        t2w = glob.glob(os.path.join(each_case, "*_T2_bias.nii.gz"))[0]
        t1c = glob.glob(os.path.join(each_case, "*_T1gad_bias.nii.gz"))[0]
        t2f = glob.glob(os.path.join(each_case, "*_FLAIR_bias.nii.gz"))[0]
        
        data_dicts.append({
            "t1n": t1n,
            "t1c": t1c,
            "t2w": t2w,
            "t2f": t2f,
        })

    len_train = int(0.8 * len(data_dicts))
    train_files, val_files = data_dicts[:len_train], data_dicts[len_train:]

    return train_files, val_files


def get_gbm_data_path():
    data_path_gbm = "/mnt/swarm_beta/xuewei/data/UPENN-GBM"

    data_dicts = []
    dirs = os.listdir(data_path_gbm)
    dirs = sorted(dirs)
    for each_case in dirs:
        case_files = sorted(glob.glob(os.path.join(data_path_gbm, each_case, "*.nii.gz")))
        if len(case_files) == 4:
            data_dicts.append({
                "t1c": case_files[1],
                "t1n": case_files[2],
                "t2f": case_files[0],
                "t2w": case_files[3],
            })

    len_train = int(0.8 * len(data_dicts))
    train_files_gbm, val_files_gbm = data_dicts[:len_train], data_dicts[len_train:]

    return train_files_gbm, val_files_gbm


def get_egd_data():
    data_path_egd = "/mnt/swarm_beta/xuewei/data/EGD"
    data_dicts = []
    dirs = os.listdir(data_path_egd)
    dirs = sorted(dirs)
    for each_case in dirs:
        case_files = sorted(glob.glob(os.path.join(data_path_egd, each_case, "[1-4]_*", "NIFTI", "[A-Z]*.nii.gz")))
        if len(case_files) == 4:
            data_dicts.append({
                "t1n": case_files[0],
                "t1c": case_files[1],
                "t2w": case_files[2],
                "t2f": case_files[3],
            })

    len_train = int(0.8 * len(data_dicts))

    return data_dicts[:len_train], data_dicts[len_train:]


def list_brats24_paths(path):
    data_dicts = []
    dirs = os.listdir(path)
    dirs = sorted(dirs)
    for each_case in dirs:
        case_files = sorted(glob.glob(os.path.join(path, each_case, "*-t*.nii.gz")))
        if len(case_files) == 4:
            data_dicts.append({
                "t1c": case_files[0],
                "t1n": case_files[1],
                "t2f": case_files[2],
                "t2w": case_files[3],
            })

    return data_dicts

def get_brats24_data():
    path_1_training_data1 = "/mnt/swarm_beta/xuewei/data/BraTS2024/training_data1"
    path_1_validation_data ='/mnt/swarm_beta/xuewei/data/BraTS2024/validation_data'
    data_dicts_1 = list_brats24_paths(path_1)

    path_2_Training_1 = "/mnt/swarm_beta/xuewei/data/BraTS24_MET/MICCAI-BraTS2024-MET-Challenge-TrainingData_1/MICCAI-BraTS2024-MET-Challenge-Training_1"

    path_2v ='/mnt/swarm_beta/xuewei/data/BraTS24_MET/MICCAI-BraTS2024-MET-Challenge-TrainingData_2'
    path_2_f ='/mnt/swarm_beta/xuewei/data/BraTS24_MET/MICCAI-BraTS2024-MET-Challenge-TrainingData_2-fixed-cases/MICCAI-BraTS2024-MET-Challenge-TrainingData_2-fixed-cases'
    path_2_c ='/mnt/swarm_beta/xuewei/data/BraTS24_MET/MICCAI-BraTS2024-MET-Challenge-ValidationData/MICCAI-BraTS2024-MET-Challenge-Validation'
    data_dicts_2 = list_brats24_paths(path_2)

    path_3 = "/mnt/swarm_beta/xuewei/data/BraTS-PEDs2024_Training/BraTS2024-PED-Challenge-TrainingData/BraTS-PEDs2024_Training"
    path_3_2     = "/mnt/swarm_beta/xuewei/data/BraTS-PEDs2024_Training/BraTS2024-PED-Challenge-ValidationData/BraTS_Validation_Data_backup"

    data_dicts_3 = list_brats24_paths(path_3)

    len_train = int(0.8 * len(data_dicts_1))
    train_files_brats24_1, val_files_brats24_1 = data_dicts_1[:len_train], data_dicts_1[len_train:]

    len_train = int(0.8 * len(data_dicts_2))
    train_files_brats24_2, val_files_brats24_2 = data_dicts_2[:len_train], data_dicts_2[len_train:]

    len_train = int(0.8 * len(data_dicts_3))
    train_files_brats24_3, val_files_brats24_3 = data_dicts_3[:len_train], data_dicts_3[len_train:]

    train_files_brats24 = train_files_brats24_1 + train_files_brats24_2 + train_files_brats24_3
    val_files_brats24 = val_files_brats24_1 + val_files_brats24_2 + val_files_brats24_3
    del train_files_brats24[1245]
    return train_files_brats24, val_files_brats24


def get_brats21_data():
    brats21_data_path = "/mnt/swarm_beta/xuewei/data/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
    brats21_data_path_v = "/mnt/swarm_beta/xuewei/data/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_ValidationData"


    data_dicts_brats21 = []

    dirs = os.listdir(brats21_data_path)
    dirs = sorted(dirs)
    for each_case in dirs:
        case_files = sorted(glob.glob(os.path.join(brats21_data_path, each_case, "*.nii.gz")))
        if len(case_files) == 5:
            data_dicts_brats21.append({
                "t1c": case_files[3],
                "t1n": case_files[2],
                "t2f": case_files[0],
                "t2w": case_files[4],
            })
    
    len_train = int(0.8 * len(data_dicts_brats21))

    return data_dicts_brats21[:len_train], data_dicts_brats21[len_train:]


def get_data_path():
    brats21_train, brats21_test = get_brats21_data()
    brats24_train, brats24_test = get_brats24_data()
    egd_train, egd_test = get_egd_data()
    gbm_train, gbm_test = get_gbm_data_path()

    # print(gbm_test[0])
    train_data = brats21_train + brats24_train + egd_train + gbm_train
    test_data = brats21_test + brats24_test + egd_test + gbm_test

    # with open('./data_path_3d.json', 'w') as f:
    #     f.write(json.dumps(train_data))

    return train_data, test_data