from codes.utils.util import get_product_of_iterable

def move_data_to_device(data, device):
    if(type(data) in [list, tuple]):
        return list(map(lambda x: x.to(device), data)), get_product_of_iterable(data[1].shape[:2])
    else:
        return data.to(device), len(data)