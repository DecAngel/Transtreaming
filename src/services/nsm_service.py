from src.data.components.td_space import NSMTensorDictSpace


class NSMService(NSMTensorDictSpace):
    def __init__(self):
        super().__init__(method='none')
