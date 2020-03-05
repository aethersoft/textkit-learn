def move_to_device(device, x):
    if device == 'cuda':
        return x.cuda()
    return x


def move_from_device(device, x):
    if device == 'cuda':
        return x.cpu()
    return x
