#include "nn.h"


//表示
void print(int m, int n, const float * x){
    int i, j;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            printf("%8.4f ", x[n*i+j]);
        }
        printf("\n");
    }
}

//行列の掛け算
void mul(int m, int n, const float * x, const float * A, float * o){
    int i, j;
    float sum;
    for (i = 0; i < m; i++){
        sum = 0;
        for (j = 0; j < n; j++){
            sum += A[n * i + j] * x[j];
        }
        o[i] = sum;
    }
}

//行列の足し算
void add(int n, const float * x, float *o){
    int i;
    for (i = 0; i < n;i++){
        o[i] = x[i] + o[i];
    }
}

//行列計算
void fc(int m, int n, const float * x, const float * A, const float * b, float * o){
    mul(m, n, x, A, o);
    add(m, b, o);
}

//ReLU関数
void relu(int n, const float * x, float * y){
    int i;
    for (i = 0; i < n; i++){
        if(x[i] < 0){
            y[i] = 0;
        }
        else{
            y[i] = x[i];
        }
    }
}

//Softmax関数
void softmax(int n, const float * x, float * y){
    int i;
    float sum = 0;
    float max = y[0];
    for (i = 0; i < n; i++){
        if(max <=y[i]){
            max = y[i];
        }
    }
    for (i = 0; i < n; i++){
        sum += exp(x[i] - max);
    }
    for (i = 0; i < n; i++){
        y[i] = exp(x[i] - max) / sum;
    }
}

//誤差逆伝搬(softmax層)
void softmaxwithloss_bwd(int n, const float * y, unsigned char t, float *dEdx){
    int i;
    float onehot[n];
    //tのベクトル表示
    for (i = 0; i < n; i++){
        onehot[i] = 0;
    }
    onehot[t] = 1;
    //計算
    for (i = 0; i < n; i++){
        dEdx[i] = y[i] - onehot[i];
    }
}

//誤差逆伝搬(ReLU層)
void relu_bwd(int n, const float * x, const float * dEdy, float * dEdx){
    int k;
    for (k = 0; k < n; k++){
        if(x[k] > 0){
            dEdx[k] = dEdy[k];
        }
        else{
            dEdx[k] = 0;
        }
    }
}

//誤差逆伝搬(FC層) 問題あり
void fc_bwd(int m, int n, const float * x, const float *dEdy, const float * A, float * dEdA, float *dEdb, float * dEdx){
    int k, l;
    for (k = 0; k < m; k++){
        for (l = 0; l < n; l++){
            dEdA[k * n + l] = dEdy[k] * x[l];
        }
    }
    for (k = 0; k < m; k++){
        dEdb[k] = dEdy[k];
    }
    for (k = 0; k < n; k++){
        for (l = 0; l < m; l++){
            dEdx[k] += A[l * n + k] * dEdy[l];
        }
    }
}

//誤差逆伝搬(3層)
void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dEdA, float * dEdb){
    float dEdx1[10], dEdx2[10], dEdx3[784];
    float y1[10], y2[10];
    for (int i = 0; i < 10; i++){
        dEdx1[i] = 0;
        dEdx2[i] = 0;
        dEdx3[i] = 0;
        y1[i] = 0;
        y2[i] = 0;
    }
    for (int i = 0; i < 784; i++){
        dEdx3[i] = 0;
    }
    //推論（３層) Step1
    fc(10, 784, x, A, b, y1);
    relu(10, y1, y2);
    softmax(10, y2, y);
    //Step2
    softmaxwithloss_bwd(10, y, t, dEdx1); //dEdxが計算された
    //Step3
    relu_bwd(10, y1, dEdx1, dEdx2); //dEdyが計算された
    //Step4
    fc_bwd(10, 784, x, dEdx2, A, dEdA, dEdb, dEdx3); //dEdA dEdb dEdxが計算された
}

int main(){
    //補助関数
    float * train_x = NULL;
    unsigned char * train_y = NULL;
    int train_count = -1;
    float * test_x = NULL;
    unsigned char * test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
    float *y = malloc(sizeof(float) * 10);
    float *dEdA = malloc(sizeof(float) * 784 * 10);
    float *dEdb = malloc(sizeof(float) * 10);
    backward3(A_784x10, b_784x10, train_x + 784 * 8, train_y[8], y, dEdA, dEdb);
    print(10, 784, dEdA);
    print(1, 10, dEdb);
    free(y);
    free(dEdA);
    free(dEdb);
    return 0;
}