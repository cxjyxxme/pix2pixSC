
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
def CreateConDataLoader(opt):
    from data.con_dataset_data_loader import ConDatasetDataLoader
    data_loader = ConDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
def CreateFaceConDataLoader(opt):
    from data.face_con_dataset_data_loader import FaceConDatasetDataLoader
    data_loader = FaceConDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
def CreatePoseConDataLoader(opt):
    from data.pose_con_dataset_data_loader import PoseConDatasetDataLoader
    data_loader = PoseConDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
