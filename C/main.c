#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define pi 3.1415926

int main() {
    int signal_len = 94676;
    int HR_len = 3787,HR_est = 3784;
    int i,ii, j, left, right, count;
    FILE *fp = NULL;
    char *line;
    char buffer[1024];
    int ppg[signal_len];
    int mean_new_ppg[signal_len];
    int acc[signal_len];
    int mean_new_acc[signal_len];
    int HR[HR_len],HRest[HR_est], HR_matlabest[HR_est],HRest_mean[HR_est],ind[HR_len+1],ind_mid;

    int K=100,N;
    int ppgstart=0;
    int ppgend=99;

    int ppg_temp_slice[K], acc_temp_slice[K];
    int kL,kR,index_slice,mean_ppg_temp_slice,max_ppg_temp_slice_index,diff_ppg_slice_mean_max;
    int mean_acc_temp_slice,max_acc_temp_slice_index,diff_acc_slice_mean_max;


    int a[N],b[N],c[N],axx[N],bxx[N],cxx[N], temp_a, temp_b;
    int c_max, cxx_max, range, c_max_aft, c_mean, cxx_mean;
//    long long a[N],b[N],c[N],axx[N],bxx[N],cxx[N],temp_a, temp_b;
//    long long c_max, cxx_max, range, c_max_aft, c_mean, cxx_mean;

    //float只是为了初始化计算方便开发，固定算法后可以不使用float
    float fs=25;
    float Nfft=1500;
    float df;
    float f[N];
    df = fs/Nfft;
    kL = (int)(1/df+0.5);
    kR = (int)(3/df)+1;
    N = kR-kL+1;
    for(i=0;i<N;i++){
        f[i] = (float)(kL+i)*df;
    }

    //读取原始ppg数据，并32位定点化
    i = 0;
    if ((fp = fopen("new_ppg.csv", "r+")) != NULL) {
        fseek(fp, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
            ppg[i] = (int) (atof(line)*100 + 0.5);
            i++;
        }
        fclose(fp);
        fp = NULL;
    }
    //读取acc数据，并32位定点化
    i = 0;
    if ((fp = fopen("new_acc.csv", "r+")) != NULL) {
        fseek(fp, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
            acc[i] = (int) (atof(line)*100 + 0.5);
            i++;
        }
        fclose(fp);
        fp = NULL;
    }
    //读取心率带记录心率
    i = 0;
    if ((fp = fopen("HRBand.csv", "r+")) != NULL) {
        fseek(fp, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
            HR[i] = (int) (atof(line));
            i++;
        }
        fclose(fp);
        fp = NULL;
    }
    //读取matlab处理结果
    i = 0;
    if ((fp = fopen("matlab_HRest.csv", "r+")) != NULL) {
        fseek(fp, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
            HR_matlabest[i] = (int) (atof(line)+0.5);
            i++;
        }
        fclose(fp);
        fp = NULL;
    }


    //ppg去均值
    for (i = 0; i < signal_len; i++) {
        if (i - 3 < 0) {
            left = 0;
        } else {
            left = i - 3;
        }
        if (i + 3 >= signal_len) {
            right = signal_len - 1;
        } else {
            right = i + 3;
        }
        mean_new_ppg[i] = 0;
        count = 0;
        for (j = left; j <= right; j++) {
            mean_new_ppg[i] += ppg[j];
            count += 1;
        }
        mean_new_ppg[i] = mean_new_ppg[i] / count;
    }

    for (i = 0; i < signal_len; i++) {
        ppg[i] = (ppg[i]-mean_new_ppg[i]);

    }
    //均值滤波ppg
    for (i = 0; i < signal_len; i++) {
        if (i - 1 < 0) {
            left = 0;
        } else {
            left = i - 1;
        }
        if (i + 1 >= signal_len) {
            right = signal_len - 1;
        } else {
            right = i + 1;
        }
        mean_new_ppg[i] = 0;
        count = 0;
        for (j = left; j <= right; j++) {
            mean_new_ppg[i] += ppg[j];
            count += 1;
        }
        mean_new_ppg[i] = mean_new_ppg[i] / count;
    }

    //acc去均值
    for (i = 0; i < signal_len; i++) {
        if (i - 3 < 0) {
            left = 0;
        } else {
            left = i - 3;
        }
        if (i + 3 >= signal_len) {
            right = signal_len - 1;
        } else {
            right = i + 3;
        }
        mean_new_acc[i] = 0;
        count = 0;
        for (j = left; j <= right; j++) {
            mean_new_acc[i] += acc[j];
            count += 1;
        }
        mean_new_acc[i] = mean_new_acc[i] / count;
    }

    for (i = 0; i < signal_len; i++) {
        acc[i] = acc[i]-mean_new_acc[i];
    }
    //均值滤波acc
    for (i = 0; i < signal_len; i++) {
        if (i - 1 < 0) {
            left = 0;
        } else {
            left = i - 1;
        }
        if (i + 1 >= signal_len) {
            right = signal_len - 1;
        } else {
            right = i + 1;
        }
        mean_new_acc[i] = 0;
        count = 0;
        for (j = left; j <= right; j++) {
            mean_new_acc[i] += acc[j];
            count += 1;
        }
        mean_new_acc[i] = mean_new_acc[i] / count;
    }
    //ppg 分段处理
    index_slice = 1;//心率序列编号
    while(ppgend<signal_len){
        //ppg归一化
        mean_ppg_temp_slice = 0;
        max_ppg_temp_slice_index = 0;

        for(i=0;i<K;i++){
            ppg_temp_slice[i] = mean_new_ppg[ppgstart+i];
            mean_ppg_temp_slice += mean_new_ppg[ppgstart+i];
            if(abs(ppg_temp_slice[i])>abs(ppg_temp_slice[max_ppg_temp_slice_index])){
                max_ppg_temp_slice_index = i;
            }
        }
        mean_ppg_temp_slice /= K;

        for(i=0;i<K;i++){
            ppg_temp_slice[i] -= mean_ppg_temp_slice;
        }

        diff_ppg_slice_mean_max = abs(ppg_temp_slice[max_ppg_temp_slice_index]/8);
        if(diff_ppg_slice_mean_max>0) {
            for (i = 0; i < K; i++) {//类似除以max操作,范围1024~-1024
                ppg_temp_slice[i] = (ppg_temp_slice[i]) * 128 / (diff_ppg_slice_mean_max);
            }
        }

        //acc归一化
        mean_acc_temp_slice = 0;
        max_acc_temp_slice_index = 0;

        for(i=0;i<K;i++){
            acc_temp_slice[i] = mean_new_acc[ppgstart+i];
            mean_acc_temp_slice += mean_new_acc[ppgstart+i];
            if(abs(acc_temp_slice[i])>abs(acc_temp_slice[max_acc_temp_slice_index])){
                max_acc_temp_slice_index = i;
            }
        }
        mean_acc_temp_slice /= K;

        for(i=0;i<K;i++){
            acc_temp_slice[i] -= mean_acc_temp_slice;
        }

        diff_acc_slice_mean_max = abs(acc_temp_slice[max_acc_temp_slice_index]/8);
        if(diff_acc_slice_mean_max>0) {
            for (i = 0; i < K; i++) {//类似除以max操作
                acc_temp_slice[i] = acc_temp_slice[i] * 128 / diff_acc_slice_mean_max;
            }
        }

        //没做hanning窗操作

        //DFT操作
        //默认初始化均为0
        for(i=0;i<N;i++){
            a[i] = 0;
            b[i] = 0;
            c[i] = 0;
            axx[i] = 0;
            bxx[i] = 0;
            cxx[i] = 0;
            for(ii=0;ii<K;ii++){//cos和sin在固定算法后可以直接用定点数值代替，目前是为了开发方便
                a[i]+= ppg_temp_slice[ii]*(int)(cos(2*pi*(i+kL)*ii/Nfft)*256);
                b[i]+= ppg_temp_slice[ii]*(int)(sin(2*pi*(i+kL)*ii/Nfft)*256);
                axx[i]+= acc_temp_slice[ii]*(int)(cos(2*pi*(i+kL)*ii/Nfft)*256);
                bxx[i]+= acc_temp_slice[ii]*(int)(sin(2*pi*(i+kL)*ii/Nfft)*256);
            }
            //依据定点范围缩放
            temp_a = a[i]/256;
            temp_b = b[i]/256;
            c[i]= temp_a*temp_a+temp_b*temp_b;
            temp_a = axx[i]/256;
            temp_b = bxx[i]/256;
            cxx[i] = temp_a*temp_a+temp_b*temp_b;
//            c[i]= ((a[i])*(a[i])+(b[i])*(b[i]));
//            cxx[i] = ((axx[i])*(axx[i])+(bxx[i])*(bxx[i]));
//            c[i]= ((a[i]/128)*(a[i]/128)+(b[i]/128)*(b[i]/128));
//            cxx[i] = ((axx[i]/128)*(axx[i]/128)+(bxx[i]/128)*(bxx[i]/128));
            //c[i]= (int)sqrt(a[i]*a[i]+b[i]*b[i]);
            //cxx[i] = (int)sqrt(axx[i]*axx[i]+bxx[i]*bxx[i]);
        }



        c_max = 0;
        cxx_max = 0;
        for(i=0;i<N;i++){
            if(c_max<c[i]){
                c_max = c[i];
            }
            if(cxx_max<cxx[i]){
                cxx_max = cxx[i];
            }
        }

        for(i=0;i<N;i++){
            if(c_max/4096 >0){
                c[i] = (c[i])/(c_max/4096);
            }
            if(cxx_max/4096 >0){
                cxx[i] = (cxx[i])/(cxx_max/4096);
            }
            c[i] -= cxx[i];
        }

        //寻峰规则
        range = 30;

        if(ppgstart == 0){
            ind_mid = 30;
        }
        else{
            if(index_slice<21){
                ind_mid = 0;
                for(i=0;i<(index_slice-1);i++){
                    ind_mid += ind[i];
                }
                ind_mid /= (index_slice-1);
            }
            else{
                ind_mid = 0;
                for(i=index_slice-21;i<(index_slice-1);i++){
                    ind_mid += ind[i];
                }
                ind_mid /= 20;
            }
        }

        ind[index_slice-1] = ind_mid;
        if(ind_mid-30<0){
            left=0;
        }
        else{
            left = ind_mid-30;
        }
        if(ind_mid+30>N-1){
            right=N-1;
        }
        else{
            right = ind_mid+30;
        }
        c_max_aft = 0;
        for(i=left+1;i<=right-1;i++){
            if(c[i]>=c[i-1]){
                if(c[i]<c[i+1]){
                    if(c_max_aft<c[i]){
                        c_max_aft = c[i];
                        ind[index_slice-1] = i;
                    }
                }
            }

        }

//        if ppgstart ==1
//        ind = 30;
//        else
//        ind = mean(ind_list(max(1, end-10):end));
//        end

        HRest[index_slice-1] = f[ind[index_slice-1]]*(float)60;

        ppgstart += 25;
        ppgend += 25;
        index_slice += 1;
    }


    for(i=0;i<HR_est;i++){
        HRest_mean[i] = 0;
        if(i<20){
            for(j=0;j<=i;j++){
                HRest_mean[i] += HRest[j];
            }
            HRest_mean[i] /= i+1;
        }
        else{
            for(j=i-19;j<=i;j++){
                HRest_mean[i] += HRest[j];
            }
            HRest_mean[i] /= 20;
        }

    }

    if ((fp = fopen("HRest.csv", "r+")) != NULL) {
        for(i=0;i<HR_est;i++){
            fprintf(fp, "%d,\n", HRest_mean[i]);  //写入a,b,c到文件中
        }
        fclose(fp);
        fp = NULL;
    }


    return 0;
}
