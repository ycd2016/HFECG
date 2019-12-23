import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from torch import nn, optim
from adamw import AdamW
from torch.utils.data import DataLoader
from dataset import ECGDataset
from config import config
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, criterion, train_dataloader,
                show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d\tloss:%.4f\tf1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1 = utils.calc_f1(target, output, threshold)
            f1_meter += f1
    return loss_meter / it_count, f1_meter / it_count


def my_collate_fn(batch):
    data, label = zip(*batch)
    new_data = []
    new_label = []
    batch_size = len(label)
    len_ = int(1250 + np.random.rand() * 1250)
    for i in range(batch_size):
        start = int(np.random.rand() * (config.target_point_num - len_))
        tmp_data = data[i].transpose(0, 2)
        if i == 0:
            new_data = (tmp_data[start:(start + len_)].transpose(0, 2))
            new_label = label[i]
        else:
            new_data = torch.cat((
                new_data, (tmp_data[start:(start + len_)].transpose(0, 2))
            ), 0)
            new_label = torch.cat((new_label, label[i]), 0)
    return new_data.reshape((batch_size, 1, 8, -1)), new_label.reshape(
        (batch_size, -1))


def train(args):
    model = models.myecgnet()
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=my_collate_fn,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=8)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                num_workers=8)
    print("train_datasize", len(train_dataset), "val_datasize",
          len(val_dataset))
    optimizer = AdamW(model.parameters(), lr=config.lr)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = utils.WeightedMultilabel(w)
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name,
                                   time.strftime("%Y%m%d%H%M"))
    os.mkdir(model_save_dir)
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    if args.resume:
        if os.path.exists(args.ckpt):
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion,
                                           train_dataloader,
                                           show_interval=10)
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
        print(
            '#epoch:%03d\tstage:%d\ttrain_loss:%.4f\ttrain_f1:%.3f\tval_loss:%0.4f\tval_f1:%.3f\ttime:%s\n'
            % (epoch, stage, train_loss, train_f1, val_loss, val_f1,
               utils.print_time_cost(since)))
        state = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            'f1': val_f1,
            'lr': lr,
            'stage': stage
        }
        save_ckpt(state, best_f1 < val_f1, model_save_dir)
        best_f1 = max(best_f1, val_f1)
        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)


def val(args):
    list_threhold = [0.5]
    model = models.myecgnet()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt,
                                         map_location='cpu')['state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                num_workers=8)
    for threshold in list_threhold:
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader,
                                     threshold)
        print('threshold %.2f\tval_loss:%0.3e\tval_f1:%.3f\n' %
              (threshold, val_loss, val_f1))


def test(args):
    from dataset import transform
    from data_process import name2index
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    utils.mkdirs(config.sub_dir)
    model = models.myecgnet()
    model.load_state_dict(torch.load(args.ckpt,
                                     map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()
    sub_file = 'result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in open(config.test_label, encoding='utf-8'):
            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.test_dir, id)
            df = pd.read_csv(file_path, sep=' ').values
            x = transform(df).unsqueeze(0).to(device)
            output = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            ixs = [i for i, out in enumerate(output) if out > 0.5]
            for i in ixs:
                fout.write("\t" + idx2name[i])
                print(i, end=',')
            fout.write('\n')
    fout.close()
    print('\n', end='')


def toplayer(args):
    from dataset import transform
    from data_process import name2index
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    utils.mkdirs(config.sub_dir)
    model = models.myecgnet()
    model.load_state_dict(torch.load(args.ckpt,
                                     map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()
    sub_file = '%s.txt' % args.ex
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(open(config.train_label, encoding='utf-8')):
            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.train_dir, id)
            df = pd.read_csv(file_path, sep=' ').values
            x = transform(df).reshape((1, 1, 8, 2500)).to(device)
            output = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            for i in output:
                fout.write("\t" + str(i))
            fout.write('\n')
    fout.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt",
                        type=str,
                        help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    if (args.command == "test"):
        test(args)
    if (args.command == "toplayer"):
        toplayer(args)
