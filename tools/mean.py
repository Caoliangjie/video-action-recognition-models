def get_mean(norm_value=255, dataset='kinetics'):
    assert dataset in ['kinetics']
    if dataset == 'kinetics':
        return [
            255 * 0.485 / norm_value, 255 * 0.456 / norm_value,
            255 * 0.406 / norm_value
        ]
 
 
def get_std(norm_value=255):
    return [
        255 * 0.229 / norm_value, 255 * 0.224 / norm_value,
        255 * 0.225 / norm_value
    ]
