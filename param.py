import torch


class Params():
    def __init__(self):
        self.root_trainA = 'dicom_3/trainA'
        self.root_trainB = 'dicom_3/trainB'
        self.root_testA = 'dicom_3/testA'
        self.root_testB = 'dicom_3/testB'

        self.size = 512
        self.batch_size = 1

        self.input_nc = 1
        self.output_nc = 1

        self.start_epoch = 0
        self.n_epochs = 50
        self.decay_start_epoch =  self.n_epochs // 2
        self.lr = 0.0002

        self.device_name = "cuda:0"
        self.device = torch.device(self.device_name)
        self.load_weight = False
        self.n_cpu = 8

#
