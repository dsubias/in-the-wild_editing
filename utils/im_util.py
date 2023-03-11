from utils.batch_transforms import Normalize


def denorm(x, device,add_bg):
    # get from [-1,1] to [0,1]
    if add_bg:
        norm = Normalize(mean=(-1, -1, -1), std=(2, 2, 2), device=device)
    else:
        norm = Normalize(mean=(-1, -1, -1, 0), std=(2, 2, 2, 1), device=device)
    return norm(x)

