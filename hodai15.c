#include "nn.h"
#include <time.h>

//表示
void print(int m, int n, const float * x){
    int i, j;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            printf("%f ", x[n*i+j]);
        }
        printf("\n");
    }
}

//配列の足し算
void add(int n, const float * x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] += x[i];
    }
}

//配列の掛け算
void scale(int n, float x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] *= x;
    }
}

//配列の要素に同じ数値を代入
void init(int n, float x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] = x;
    }
}

//配列の初期化（ランダム）
void rand_init(int n, float *o){
    int i;
    srand(time(NULL));
    for (i = 0; i < n; i++){
        o[i] = ((float)rand() / RAND_MAX)*2 - 1;
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

//行列計算(全体)
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
        dEdx[k] = 0;
        for (l = 0; l < m; l++){
            dEdx[k] += A[n * l + k] * dEdy[l];
        }
    }
}

//誤差逆伝搬(3層)
void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dEdA, float * dEdb){
    //関数で仮に用いる変数の初期化
    float dEdx1[10], dEdx2[10], dEdx3[784], y1[10], y2[10];
    init(10, 0, dEdx1);
    init(10, 0, dEdx2);
    init(10, 0, y1);
    init(10, 0, y2);
    init(784, 0, dEdx3);
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

//ランダムシャッフル
void shuffle(int n, int *x){
    srand(time(NULL));
    for (int i = 0; i < n; i++){
        int j = rand() % n;
        int t = x[i];
        x[i] = x[j];
        x[j] = t;
     }
}

//損失関数
float cross_entropy_error(const float * y, int t){
    return -log(y[t] + 1e-7);
}

//推論
int inference3(const float * A, const float * b, const float *x){
    int i, index;
    float m;
    float y[10];
    fc(10, 784, x, A, b, y);
    relu(10, y, y);
    softmax(10, y, y);
    m = y[0];
    index = 0;
    for (i = 0; i < 10; i++){
        if (m <= y[i]){
            m = y[i];
        }
    }
    for (i = 0; i < 10; i++){
        if(m == y[i]){
            index = i;
        }
    }
    return index;
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
    //NNの学習
    int batch_size = 100;
    float l_rate = 0.1;
    int epoch = 10;
    int index[10000];
    float *y = malloc(sizeof(float) * 10);
    float *dEdA = malloc(sizeof(float) * 784 * 10);
    float *dEdb = malloc(sizeof(float) * 10);
    float *A = malloc(sizeof(float) * 784 * 10);
    float *b = malloc(sizeof(float) * 10);
    float *avr_dEdA = malloc(sizeof(float) * 784 * 10);
    float *avr_dEdb = malloc(sizeof(float) * 10);
    //行列を乱数で初期化
    rand_init(7840, A);
    rand_init(10, b);
    //添字の初期化
    for (int i = 0; i < 10000; i++){
        index[i] = i;
    }
    //エポックの開始
    for (int i = 0; i < epoch; i++){
        //シャッフル
        shuffle(10000, index);
        //ミニバッチ学習開始
        for (int j = 0; j < 10000/batch_size; j++){
            //平均勾配を０で初期化
            init(7840, 0, avr_dEdA);
            init(10, 0, avr_dEdb);
            //平均勾配にdEdA、dEdbを加える、100の添字について
            for (int k = batch_size * j; k < batch_size * (j + 1); k++){
                backward3(A, b, test_x + 784*index[k], test_y[index[k]], y, dEdA, dEdb);
                add(7840, dEdA, avr_dEdA);
                add(10, dEdb, avr_dEdb);
            }
            //新たな平均勾配をえる
            scale(7840, -l_rate/batch_size, avr_dEdA); 
            scale(10, -l_rate/batch_size, avr_dEdb);
            //学習率をかけ、行列、ベクトルの更新
            add(7840, avr_dEdA, A);
            add(10, avr_dEdb, b);
        }
        //ミニバッチ学習終了
        //正解率を表示clear
        int sum = 0;
        for (int j = 0; j < test_count; j++){
            if(inference3(A, b, test_x + j*width*height) == test_y[j]){
                sum++;
            }
        }
        printf("%f%%\n", sum * 100.0 / test_count);
    }
    free(y);
    free(dEdA);
    free(dEdb);
    free(A);
    free(b);
    free(avr_dEdA);
    free(avr_dEdb);

    return 0;
}