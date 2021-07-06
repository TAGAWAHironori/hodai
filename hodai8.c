#include "nn.h"

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
//計算
void fc(int m, int n, const float * x, const float * A, const float * b, float * o){
    mul(m, n, x, A, o);
    add(m, b, o);
}

//誤差逆伝搬softmax層
void softmaxwithloss_bwd(int n, const float * y, unsigned char t, float *dEdx){
    int i;
    float onehot[n];
    //tのベクトル表示
    for (i = 0; i < n; i++){
        onehot[i] = 0;
    }
    onehot[t] = 1;
    for (i = 0; i < n; i++){
            dEdx[i] = y[i] - onehot[i];
        }
}

int main(){
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
    float *y = malloc(sizeof(float) * 10);
    float dEdx[10];
    fc(10, 784, train_x, A_784x10, b_784x10, y);
    softmaxwithloss_bwd(10, y, train_y[0], dEdx);
    return 0;
}