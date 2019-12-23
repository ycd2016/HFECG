import os


class Config:
    root = r'./'
    train_dir = os.path.join(root, 'hf_round2_train')
    test_dir = './hf_round2_testB'
    train_label = os.path.join(root, 'hf_round2_train.txt')
    test_label = './hf_round2_subB.txt'
    arrythmia = os.path.join(root, 'hf_round2_arrythmia.txt')
    train_data = os.path.join(root, 'train.pth')
    model_name = 'after'
    stage_epoch = [24, 48]
    batch_size = 192
    num_classes = 34
    max_epoch = 28
    target_point_num = 2500
    ckpt = './ckpt'
    sub_dir = './submit'
    lr = 4e-5
    current_w = 'current.pth'
    best_w = 'best.pth'
    lr_decay = 10
    temp_dir = os.path.join(root, 'temp')


config = Config()
