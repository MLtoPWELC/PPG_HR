file_name = 'TSECN_acc_single'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch.utils.data as Data
import numpy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from ptflops import get_model_complexity_info

from nni.compression.pytorch.utils.counter import count_flops_params

numpy.random.seed(1708)
for index_look in range(15):
    print(index_look)
    train_epoch = 300
    batch_size = 1024
    # index_look = 4     # 0-14

    ppg_slice = numpy.load('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/ppg_slice_25.npy',
                           allow_pickle=True)
    HR_label = numpy.load('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/HR_true.npy', allow_pickle=True)
    signal_slice_len = numpy.load('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/signal_slice_len.npy', allow_pickle=True)
    acc_slice = numpy.load('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/acc_single_slice_25.npy', allow_pickle=True)

    XTrain_all = numpy.concatenate((ppg_slice.reshape((64697, 1, 200)), acc_slice.reshape((64697, 1, 200))), axis=1)

    train_ppg_num = [index_look//3, index_look % 3]      # 0-4  0-2
    pot1 = sum(signal_slice_len[0:train_ppg_num[0]*3])
    pot2 = sum(signal_slice_len[0:train_ppg_num[0]*3+3])

    train_index = list(range(sum(signal_slice_len)))
    del train_index[pot1:pot2]
    XTrain = XTrain_all[train_index]
    YTrain = HR_label[train_index]

    test_index = list(range(sum(signal_slice_len[0:index_look]),sum(signal_slice_len[0:index_look+1])))
    vali_index = [x for x in range(pot1,pot2) if x not in test_index]
    XValid = XTrain_all[vali_index]
    YValid = HR_label[vali_index]
    XTest = XTrain_all[test_index]
    YTest = HR_label[test_index]

    XTrain = torch.from_numpy(numpy.array(XTrain)).float()
    YTrain = torch.from_numpy(numpy.array(YTrain)).float()
    XValid = torch.from_numpy(numpy.array(XValid)).float()
    YValid = torch.from_numpy(numpy.array(YValid)).float()
    XTest = torch.from_numpy(numpy.array(XTest)).float()
    YTest = torch.from_numpy(numpy.array(YTest)).float()

    torch_train = Data.TensorDataset(XTrain, YTrain)
    loader_train = Data.DataLoader(
        dataset=torch_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    torch_valid = Data.TensorDataset(XValid, YValid)
    loader_valid = Data.DataLoader(
        dataset=torch_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    ##############################################################################
    class SELayer(nn.Module):
        def __init__(self, channel, reduction=5):
            super(SELayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel//reduction, channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1)
            return x * y.expand_as(x)


    class Chomp1d(nn.Module):
        def __init__(self, chomp_size):
            super(Chomp1d, self).__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            return x[:, :, :-self.chomp_size].contiguous()


    class TemporalBlock(nn.Module):
        def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
            super(TemporalBlock, self).__init__()
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.se = SELayer(n_outputs)
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2,
                                     self.se)
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()
            self.init_weights()

        def init_weights(self):
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class TemporalConvNet(nn.Module):
        def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
            super(TemporalConvNet, self).__init__()
            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = num_inputs if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class TCN(nn.Module):
        def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
            super(TCN, self).__init__()
            self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
            self.linear1 = nn.Linear(num_channels[-1], num_channels[-1]//2)
            self.linear2 = nn.Linear(num_channels[-1]//2, output_size)

        def forward(self, inputs):
            """Inputs have to have dimension (N, C_in, L_in)"""
            y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
            o = self.linear1(y1[:, :, -1])
            o = self.linear2(o)
            return o
    ##############################################################################
    nhid = 60
    levels = 9
    channel_sizes = [nhid] * levels
    kernel_size = 2
    dropout = 0.0011
    model = TCN(input_size=2, output_size=1, num_channels=channel_sizes,
                kernel_size=kernel_size, dropout=dropout).cuda()

    # flops, params, results = count_flops_params(model, (1, 2, 200))
    # print(f'flops: {flops/1e6:.3f}M, Params: {params/1e6:.3f}M')
    # print(results)
    # print(model)

    # macs, params = get_model_complexity_info(model, (1, 512), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters())
    min_Mean_err = float('inf')
    model_index = 0
    min_Mean_loss = float('inf')
    model_index_min_loss = 0
    #train_epoch
    for epoch in range(train_epoch):
        # print('epoch:%d' % epoch)
        for step, (batch_x, batch_y) in enumerate(loader_train):
            x = batch_x.cuda()
            y = batch_y.unsqueeze(1).cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = nn.SmoothL1Loss()(out, y)
            loss.backward()
            optimizer.step()

        pred_valid = []
        error = 0
        vali_loss = 0
        for step2, (batch_x2, batch_y2) in enumerate(loader_valid):
            # print('valid step:%d' % step2)
            x2 = batch_x2.cuda()
            y2 = batch_y2.unsqueeze(1).cuda()
            output = model(x2)
            vali_loss += nn.SmoothL1Loss(reduction='sum')(output, y2).item()
            error += sum(abs(y2.squeeze().cpu().numpy() - output.detach().cpu().squeeze().numpy()))

        vali_loss /= len(YValid)

        Mean_err = error / len(YValid)
        # print('Vali_Mean_loss:%f' % vali_loss)
        # print('Vali_Mean_err:%f' % Mean_err)
        if min_Mean_err > Mean_err:
            model_index = epoch
            min_Mean_err = Mean_err
            torch.save(model, 'model_'+file_name+'.pkl')

    #################################################
    # torch.cuda.empty_cache()
    print('test start!')
    print('model_index = '+str(model_index))
    print('Vali_Mean_err = '+str(min_Mean_err))
    model = torch.load('model_'+file_name+'.pkl')
    model.eval()

    batch_size = 512
    torch_test = Data.TensorDataset(XTest, YTest)
    loader_test = Data.DataLoader(
        dataset=torch_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    pred_test = []
    for step3, (batch_x3, batch_y3) in enumerate(loader_test):
        # print('valid step:%d' % step2)
        x3 = batch_x3.cuda()

        if step3 == 0:
            out = model(x3).data.cpu().squeeze().numpy()
            pred_test = out
        else:
            out = model(x3).data.cpu().squeeze().numpy()
            pred_test = numpy.hstack((pred_test, out))


    Mean_err = sum(abs(YTest.numpy() - pred_test)) / len(YTest)
    print('Test_Mean_err:%f' % Mean_err)
    # plt.plot(YTest.numpy())
    # plt.plot(pred_test)
    # plt.show()

    ################################################################
    mean_pred_test = numpy.zeros((len(pred_test,)))
    for i in range(len(pred_test)):
        mean_pred_test[i] = numpy.mean(pred_test[max(0, i-20):i+1])

    Mean2_err = sum(abs(YTest.numpy() - mean_pred_test)) / len(YTest)
    print('Test_Mean2_err:%f' % Mean2_err)
    # plt.plot(YTest.numpy())
    # plt.plot(mean_pred_test)
    # plt.show()

    model_dir = '/home/yinyibo/PycharmProjects/pytorch/final_paper/PPG_HR/result/test_' + file_name + '.txt'
    with open(model_dir, 'a') as file:
        file.write('index:' + str(index_look) + '\n')
        file.write('nhid:' + str(nhid) + '\n')
        file.write('levels:' + str(levels) + '\n')
        file.write('kernel_size:' + str(kernel_size) + '\n')
        file.write('dropout:' + str(dropout) + '\n')
        file.write('Mean_err:' + str(Mean_err) + '\n')
        file.write('Mean2_err:' + str(Mean2_err) + '\n')
        file.write('-------------------------------------------------------------------' + '\n')
