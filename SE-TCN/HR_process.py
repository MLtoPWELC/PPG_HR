from numba import cuda
import pickle
import numba
import numpy
import math
import csv
import librosa
import matplotlib.pyplot as plt
print(cuda.gpus)

@cuda.jit
# def gpu_dft(ppg_slice, n):
def gpu_dft(ppg_slice, acc_slice, a, b, c, n, a_ACC, b_ACC, c_ACC):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        temp = ppg_slice[idx]
        acc_x = acc_slice[idx, :, 0]
        acc_y = acc_slice[idx, :, 1]
        acc_z = acc_slice[idx, :, 2]
        for k in range(k_R - k_L + 1):
            for ii in range(len(temp)):
                a[idx][k] += 2 / DFT_for_ppg_point * temp[ii][0] * math.cos(2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
                b[idx][k] += 2 / DFT_for_ppg_point * temp[ii][0] * math.sin(2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
##############################################################
                #test!!!!!!!!!!!!
                a_ACC[0][idx][k] += 2 / DFT_for_ppg_point * acc_x[ii] * math.cos(
                    2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
                b_ACC[0][idx][k] += 2 / DFT_for_ppg_point * acc_x[ii] * math.sin(
                    2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
                a_ACC[1][idx][k] += 2 / DFT_for_ppg_point * acc_y[ii] * math.cos(
                    2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
                b_ACC[1][idx][k] += 2 / DFT_for_ppg_point * acc_y[ii] * math.sin(
                    2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
                a_ACC[2][idx][k] += 2 / DFT_for_ppg_point * acc_z[ii] * math.cos(
                    2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
                b_ACC[2][idx][k] += 2 / DFT_for_ppg_point * acc_z[ii] * math.sin(
                    2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
            c[idx][k] = (a[idx][k] * a[idx][k] + b[idx][k] * b[idx][k])**0.5
            c_ACC[0][idx][k] = (a_ACC[0][idx][k] * a_ACC[0][idx][k] + b_ACC[0][idx][k] * b_ACC[0][idx][k])**0.5
            c_ACC[1][idx][k] = (a_ACC[1][idx][k] * a_ACC[1][idx][k] + b_ACC[1][idx][k] * b_ACC[1][idx][k])**0.5
            c_ACC[2][idx][k] = (a_ACC[2][idx][k] * a_ACC[2][idx][k] + b_ACC[2][idx][k] * b_ACC[2][idx][k])**0.5


@cuda.jit
def gpu_stft(ppg_slice, stft_slice, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        temp_signal = ppg_slice[idx]
        temp_signal_len = len(temp_signal)
        start = 0
        hoplen = 8
        littlt_slice_len = 32
        step = 0

        while start+littlt_slice_len <= temp_signal_len:
            temp = temp_signal[start:start+littlt_slice_len]
            for k in range(k_R - k_L + 1):
                a = 0.0
                b = 0.0
                for ii in range(len(temp)):
                    a += 2 / DFT_for_ppg_point * temp[ii] * math.cos(2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
                    b += 2 / DFT_for_ppg_point * temp[ii] * math.sin(2 * math.pi * (k + k_L) * ii / DFT_for_ppg_point)
                stft_slice[idx][step][k] = (a * a + b * b)**0.5
            step += 1
            start += hoplen


if __name__ == '__main__':
    # slice_len = 2  # second
    param_change = 1

    if param_change == 1:
        file_path = '/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/S'
        ppg = []
        acc = []
        HR_true = []
        for i in range(1, 16):
            print(i)
            signal_file = file_path + str(i) + '/S' + str(i) + '.pkl'
            with open(signal_file, 'rb') as fr:
                all_signal = pickle.load(fr, encoding='bytes')

            tempp = numpy.array(all_signal[b'signal'][b'wrist'][b'BVP'])
            tempp = tempp.reshape((len(tempp),))
            numpy.save(str(i)+'.npy', tempp)

            tempp = numpy.array(all_signal[b'signal'][b'wrist'][b'ACC'])
            numpy.save(str(i) + 'acc.npy', tempp)

            tempp = numpy.array(all_signal[b'label'])
            numpy.save(str(i) + 'HR.npy', tempp)

            ppg.append(all_signal[b'signal'][b'wrist'][b'BVP'])
            # plt.plot(ppg[-1])
            # plt.show()
            acc.append(all_signal[b'signal'][b'wrist'][b'ACC'])
            # plt.plot(acc[-1])
            # plt.show()
            HR_true.extend(all_signal[b'label'])

        ppg_slice = []
        acc_slice = []
        acc_single_slice = []
        # ppg_fs = 64
        # acc_fs = 32
        ppg_fs = 25
        acc_fs = 25
        for i in range(15):
            ppg_temp = ppg[i]
            acc_temp = acc[i]
            ppg_temp = librosa.resample(ppg_temp.reshape((len(ppg_temp),)), 64, 25)
            temp = []
            for j in range(3):
                temp.append(librosa.resample(acc_temp[:, j].reshape((len(acc_temp),)), 32, 25))
            acc_temp = numpy.array(temp).transpose()

            start_slice = 0
            len_temp_ppg = len(ppg_temp)
            while start_slice + ppg_fs * 8 <= len_temp_ppg:
                temp = ppg_temp[start_slice:start_slice + ppg_fs * 8]
                # temp = (temp - sum(temp) / len(temp)) / numpy.std(temp)
                ppg_slice.append(temp)
                start_slice += ppg_fs * 2
            start_slice = 0
            len_temp_acc = len(acc_temp)
            acc_single = numpy.zeros((len(acc_temp)))
            for acc_index in range(len(acc_temp)):
                acc_single[acc_index] = numpy.sqrt(
                    acc_temp[acc_index][0] ** 2 + acc_temp[acc_index][1] ** 2 + acc_temp[acc_index][2] ** 2)

            while start_slice + acc_fs * 8 <= len_temp_acc:
                temp = acc_temp[start_slice:start_slice + acc_fs * 8][:]
                # temp = (temp-sum(temp)/len(temp))/numpy.std(temp)
                acc_slice.append(temp)
                temp = acc_single[start_slice:start_slice + acc_fs * 8]
                acc_single_slice.append(temp)
                start_slice += acc_fs * 2

        # numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/acc_slice.npy',
        #            numpy.array(acc_slice))
        # numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/acc_single_slice.npy',
        #            numpy.array(acc_single_slice))
        # numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/ppg_slice.npy', numpy.array(ppg_slice))
        numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/acc_slice_25.npy',
                   numpy.array(acc_slice))
        numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/acc_single_slice_25.npy',
                   numpy.array(acc_single_slice))
        numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/ppg_slice_25.npy',
                   numpy.array(ppg_slice))
        numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/HR_true.npy', numpy.array(HR_true))
    else:
        ppg_slice = numpy.load('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/ppg_slice.npy',
                               allow_pickle=True)
        HR_true = numpy.load('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/HR_true.npy',
                              allow_pickle=True)
        acc_slice = numpy.load('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/acc_slice.npy',
               allow_pickle=True)

    ppg_fs = 64
    DFT_for_ppg_point = int(ppg_fs / (2.5 / 150))
    df = ppg_fs / DFT_for_ppg_point
    k_L = math.floor(0.5 / df)
    k_R = math.ceil(3 / df)
    findex = numpy.arange(k_L, k_R + 1) * df
    n = len(ppg_slice)
    a = numpy.zeros((n, k_R - k_L + 1))
    b = numpy.zeros((n, k_R - k_L + 1))
    c = numpy.zeros((n, k_R - k_L + 1))

    a_ACC = numpy.zeros((3, n, k_R - k_L + 1))
    b_ACC = numpy.zeros((3, n, k_R - k_L + 1))
    c_ACC = numpy.zeros((3, n, k_R - k_L + 1))

    HR_est_nofilter = numpy.zeros((len(ppg_slice, )))

    threads_per_block = 512
    blocks_per_grid = math.ceil(n/threads_per_block)
    ppg_slice = numpy.array(ppg_slice)
    gpu_dft[blocks_per_grid, threads_per_block](ppg_slice, acc_slice, a, b, c, n, a_ACC, b_ACC, c_ACC)
    cuda.synchronize()

    ppg_slice = ppg_slice.reshape((len(ppg_slice), 512))
    stft_slice = numpy.zeros((len(ppg_slice), 61, 121))
    gpu_stft[blocks_per_grid, threads_per_block](ppg_slice, stft_slice, n)

    for i in range(n):
        HR_est_nofilter[i] = findex[numpy.argmax(c[i])]*60

    numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/hr_est_nofilter_cuda.npy', HR_est_nofilter)
    numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/stft_slice.npy', stft_slice)
    numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/acc_dft_slice.npy', c_ACC)
    numpy.save('/home/yinyibo/PycharmProjects/pytorch/PPG/DATA/PPG_FieldStudy/ppg_dft_slice.npy', c)

