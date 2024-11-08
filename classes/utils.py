class TensorManager():
    """
    Class to manage the tensors with overall applications.
    """
    def __init__(self):
        self.object_type = "Tensor Manager"

    def full_squeeze(self, *tensors):
        """
        Squeeze tensor until it does not have any dimension of size 1.
        """
        list = []
        for tensor in tensors:
            while tensor.dim() != 0 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

            list.append(tensor)

        return list
    
class DataFromJSON():
    """
    Class to manage the data of the model.
    """
    def __init__(self, json_dict, data_type: str):
        self.__class_name = "DataFromJSON"
        self.__data_type = data_type
        self.loop(json_dict)

    def __str__(self):
        return f"{self.__class_name} object with data type: {self.__data_type}"

    def loop(self, json_dict):
        if not isinstance(json_dict, dict):
            return
        for key, value in json_dict.items():
            if isinstance(value, dict):
                self.loop(value)
            else:
                if hasattr(self, key):
                    raise ValueError(f"Variable {key} already exists in the class. Rename the json key in your configuration file.")
                else:
                    setattr(self, key, value)