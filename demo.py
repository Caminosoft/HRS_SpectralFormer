import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix
from helpers.utils import *
from helpers.inference import perform_inference

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'Custom'], default='Indian', help='dataset to use')
parser.add_argument('--dataset_dir', type=str, default='./custom_datasets', help='Directory containing custom datasets')
parser.add_argument('--flag_test', choices=['test', 'train', 'inference'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
    data = loadmat('./data/Houston.mat')
elif args.dataset == 'Custom':
    # dataset_path = input("Enter the Custom Dataset Path:")
    # data = loadmat(dataset_path)
    pass
else:
    raise ValueError("Unkknow dataset")
def function_requirement(flag, data):
    color_mat = loadmat('./data/AVIRIS_colormap.mat')
    TR = data['TR']
    TE = data['TE']
    input = data['input'] #(145,145,200)
    label = TR + TE
    num_classes = np.max(TR)

    color_mat_list = list(color_mat)
    color_matrix = color_mat[color_mat_list[3]] #(17,3)
    # normalize data by band norm
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    # data size
    height, width, band = input.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    #-------------------------------------------------------------------------------
    # obtain train and test data
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
    #-------------------------------------------------------------------------------
    # load data
    x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    Label_train=Data.TensorDataset(x_train,y_train)
    x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
    Label_test=Data.TensorDataset(x_test,y_test)
    x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    Label_true=Data.TensorDataset(x_true,y_true)

    label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)

    #-------------------------------------------------------------------------------
    # create model
    model = ViT(
        image_size = args.patches,
        near_band = args.band_patches,
        num_patches = band,
        num_classes = num_classes,
        dim = 64,
        depth = 5,
        heads = 4,
        mlp_dim = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
        mode = args.mode
    )
    model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
    model_state_path = './log/custom_model_state.pt'

    if (flag == 'test'):
        return model_state_path, model, label_test_loader, criterion, optimizer, height, width, label_true_loader, total_pos_true, color_matrix, label
    elif (flag == 'train'):
        return model, model_state_path, label_train_loader, criterion, optimizer, scheduler, label_test_loader
    elif (flag == 'inference'):
        return model, label_test_loader, label_true_loader, height, width, total_pos_true, color_matrix, label, model_state_path, criterion, optimizer
#-------------------------------------------------------------------------------
if args.flag_test == 'test':
    model_state_path, model, label_test_loader, criterion, optimizer, height, width, label_true_loader, total_pos_true, color_matrix, label = function_requirement(args.flag_test, data)
    if (args.dataset == 'Custom'):
        model.load_state_dict(torch.load(model_state_path))
    elif args.mode == 'ViT':
        model.load_state_dict(torch.load('./log/ViT.pt'))
    elif (args.mode == 'CAF') and (args.patches == 1):
        model.load_state_dict(torch.load('./log/SpectralFormer_pixel_indian.pt'))
    elif (args.mode == 'CAF') and (args.patches == 7):
        model.load_state_dict(torch.load('./log/SpectralFormer_pixel_indian.pt'))
    else:
        raise ValueError("Wrong Parameters")
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    # Output classification maps
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
    plt.subplot(1,1,1)
    plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    savemat('./data/matrix.mat',{'P':prediction_matrix, 'label':label})
    plt.savefig('./data/testplot.png')

elif args.flag_test == 'train':
    model, model_state_path, label_train_loader, criterion, optimizer, scheduler, label_test_loader = function_requirement(args.flag_test, data)
    print("Start training")
    tic = time.time()

    load_model_state(model, model_state_path)  # Load the model state (if exists)

    # Training loop
    for epoch in range(args.epoches):
        model.train()  # Set the model to training mode

        # Train the model
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)

        print("\rEpoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
              .format(epoch + 1, train_obj, train_acc), end='', flush=True)

        optimizer.step()
        scheduler.step()

        if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1):
            # Save the model state at the end of each epoch
            save_model_state(model, model_state_path)
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
elif args.flag_test == 'inference':
    # print("inference Started")
    # if (args.dataset == 'Custom'):
    #     performance_metrics = perform_inference(model, label_test_loader, label_true_loader, height, width, total_pos_true, color_matrix, label, model_state_path, "Custom", criterion, optimizer)
    #     print(performance_metrics)
    # else:
    #     print("Custom did not loaded")
    #     exit()
    print("Inference Started")
    if args.dataset == 'Custom':
        dataset_dir = args.dataset_dir  # Directory containing custom datasets
        dataset_files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith('.mat')]

        for dataset_file in dataset_files:
            print(f"Processing dataset: {dataset_file}")
            data = loadmat(dataset_file)
            model, label_test_loader, label_true_loader, height, width, total_pos_true, color_matrix, label, model_state_path, criterion, optimizer = function_requirement(args.flag_test, data)
            # Extract necessary data from the loaded file
            # Data Preparation
            # Modify the perform_inference call to include the dataset_file as the dataset name
            OA2, AA_mean2, Kappa2, AA2 = perform_inference(model, label_test_loader, label_true_loader, height, width, total_pos_true, color_matrix, label, model_state_path, dataset_file, criterion, optimizer)

            print(f"Result for {dataset_file}")
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))

        print("Inference Completed")
    else:
        print("Custom dataset directory not provided")


print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")
print("Parameter:")

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

print_args(vars(args))









