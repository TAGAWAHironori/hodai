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
    }
}

//Softmax関数
void softmax(int n, const float * x, float * y){
    int i;
    float sum;
    for (i = 0; i < n; i++){
        y[i] = exp(x[i]);
        sum += y[i];
    }

    for (i = 0; i < n; i++){
        y[i] /= sum;
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

//誤差逆伝搬(FC層)
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
    for (k = 0; k < m; k++){
        for (l = 0; l < n;l++){
            dEdx[k] += A[k * n + l] * dEdy[k];
        }
    }
}

//誤差逆伝搬(3層)
void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dEdA, float * dEdb){
    float dEdx1[10], dEdx2[10], dEdx3[10];
    for (int i = 0; i < 10; i++){
        dEdx1[i] = 0;
        dEdx2[i] = 0;
        dEdx3[i] = 0;
    }
    //推論（３層) Step1
    fc(10, 784, x, A, b, y);
    relu(10, y, y);
    softmax(10, y, y);
    //Step2
    softmaxwithloss_bwd(10, y, t, dEdx1); //dEdxが計算された
    //Step3
    relu_bwd(10, y, dEdx1, dEdx2); //dEdyが計算された
    //Step4
    fc_bwd(10, 784, x, dEdx2, A, dEdA, dEdb, dEdx3); //dEdA dEdb dEdxが計算された
}

//ランダムシャッフル
void shuffle(int n, int *x){
     for(int i = 0; i < n; i++) {
        int j = rand() % n;
        int t = x[i];
        x[i] = x[j];
        x[j] = t;
     }
}

//損失関数
float cross_entropy_error(const float * y, int t){
    return (-1 * t * log(y + 1e-7));
}

//配列の足し算
void add(int n, const float * x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] += x[i];
    }
}

//
void scale(int n, float x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] += x;
    }
}

//
void init(int n, float x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] = x;
    }
}

//配列の初期化（ランダム）
void rand_init(int n, float *o){
    int i;
    for (i = 0; i < n; i++){
        o[i] = ((float)rand() / RAND_MAX)*2 - 1;
    }
}

//表示
void print(int m, int n, const float * x){
    int i, j;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            printf("%0.4f ", x[n*i+j]);
        }
        printf("\n");
    }
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
    

}