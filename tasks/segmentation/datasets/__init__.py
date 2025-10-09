from .ISPRS_dataset import ISRPS_Dataset


def build_dataset(dataset_name, data_type="test", **kwargs):
    if dataset_name == "Potsdam":
        ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12'
               ] if data_type == "test" else [
                   '6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10',
                   '3_12', '5_12', '7_11', '7_9', '6_9', '7_7', '4_12', '6_8',
                   '6_12', '6_7', '4_11'
               ]
        root_dir = "/home/yyyjvm/SS-datasets/ISPRS_dataset/"
        data_dir = root_dir + "Potsdam/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif"
        label_dir = root_dir + "Potsdam/5_Labels_for_participants/top_potsdam_{}_label.tif"
        return ISRPS_Dataset(ids=ids,
                             data_dir=data_dir,
                             label_dir=label_dir,
                             dataset_name=dataset_name,
                             data_type=data_type,
                             window_size=kwargs.get("window_size", (224, 224)))
    elif dataset_name == "Vaihingen":
        ids = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38
               ] if data_type == "test" else [
                   '1', '3', '5', '7', '11', '13', '15', '17', '21', '23',
                   '26', '28', '30', '32', '34', '37'
               ]
        root_dir = "/home/yyyjvm/SS-datasets/ISPRS_dataset/"
        data_dir = root_dir + "Vaihingen/top/top_mosaic_09cm_area{}.tif"
        label_dir = root_dir + "Vaihingen/gts_complete/top_mosaic_09cm_area{}.tif"
        return ISRPS_Dataset(ids=ids,
                             data_dir=data_dir,
                             label_dir=label_dir,
                             dataset_name=dataset_name,
                             data_type=data_type,
                             window_size=kwargs.get("window_size", (224, 224)))
