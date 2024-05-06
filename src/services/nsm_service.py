from src.datamodules.data_spaces.nsm_space import NSMDataSpace


class NSMService(NSMDataSpace):
    def __init__(self):
        super().__init__(method='none')
