#include "nn.h"
#include <time.h>

//パラメータの保存
void save(const char * filename, int m, int n, const float * A, const float * b){
    FILE *fp = NULL;
    fp = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
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
    float max = x[0];
    for (i = 0; i < n; i++){
        if(max <=x[i]){
            max = x[i];
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
    for (k = 0; k < n; k++){
        dEdx[k] = 0;
        for (l = 0; l < m; l++){
            dEdx[k] += A[n * l + k] * dEdy[l];
        }
    }
}

//誤差逆伝搬(6層)
void backward6(const float * A1, const float * b1, const float * A2, const float * b2, 
                const float * A3, const float * b3, const float * x, unsigned char t, 
                float * y, float * dEdA1, float * dEdb1, float * dEdA2, float * dEdb2, 
                float * dEdA3, float * dEdb3){
    //関数で仮に用いる変数の初期化
    float dEdx1[10], dEdx2[100], dEdx3[100], dEdx4[50], dEdx5[50], dEdx6[784], 
        y1[50], y2[50], y3[100], y4[100];
    //推論（6層) 
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y2);
    fc(100, 50, y2, A2, b2, y3);
    relu(100, y3, y4);
    fc(10, 100, y4, A3, b3, y);
    softmax(10, y, y);
    //推論（6層）完了
    //逆伝搬
    softmaxwithloss_bwd(10, y, t, dEdx1);
    fc_bwd(10, 100, y4, dEdx1, A3, dEdA3, dEdb3, dEdx2);
    relu_bwd(100, y3, dEdx2, dEdx3);
    fc_bwd(100, 50, y2, dEdx3, A2, dEdA2, dEdb2, dEdx4);
    relu_bwd(50, y1, dEdx4, dEdx5);
    fc_bwd(50, 784, x, dEdx5, A3, dEdA1, dEdb1, dEdx6);
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
int inference6(const float * A1, const float * b1, const float * A2,
                const float * b2, const float * A3, const float * b3,
                const float *x){
    int i, index;
    float m;
    float y1[50], y2[100], y[10];
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    fc(100, 50, y1, A2, b2, y2);
    relu(100, y2, y2);
    fc(10, 100, y2, A3, b3, y);
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
    //ファイル読み込み
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
    int batch_size = 100; //バッチサイズ
    float l_rate = 0.1; //学習率
    int epoch = 10; //エポック数
    int index[60000];
    float cee = 0;
    float *y = malloc(sizeof(float) * 10);
    float *dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *dEdb1 = malloc(sizeof(float) * 50);
    float *dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *dEdb2 = malloc(sizeof(float) * 100);
    float *dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *dEdb3 = malloc(sizeof(float) * 10);
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b1 = malloc(sizeof(float) * 50);
    float *b2 = malloc(sizeof(float) * 100);
    float *b3 = malloc(sizeof(float) * 10);
    float *avr_dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *avr_dEdb1 = malloc(sizeof(float) * 50);
    float *avr_dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *avr_dEdb2 = malloc(sizeof(float) * 100);
    float *avr_dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *avr_dEdb3 = malloc(sizeof(float) * 10);
    //行列を乱数で初期化
    rand_init(784 * 50, A1);
    rand_init(50 * 100, A2);
    rand_init(100 * 10, A3);
    rand_init(50, b1);
    rand_init(100, b2);
    rand_init(10, b3);
    //添字の初期化
    for (int i = 0; i < 60000; i++){
        index[i] = i;
    }
    //エポックの開始
    for (int i = 0; i < epoch; i++){
        //シャッフル
        shuffle(60000, index);
        //ミニバッチ学習開始
        for (int j = 0; j < 60000/batch_size; j++){
            //平均勾配を０で初期化
            init(784 * 50, 0, avr_dEdA1);
            init(50, 0, avr_dEdb1);
            init(5000, 0, avr_dEdA2);
            init(100, 0, avr_dEdb2);
            init(1000, 0, avr_dEdA3);
            init(10, 0, avr_dEdb3);
            //平均勾配にdEdA、dEdbを加える、100の添字について
            for (int k = batch_size * j; k < batch_size * (j + 1); k++){
                backward6(A1, b1, A2, b2, A3, b3, train_x + height * width * index[k], train_y[index[k]], y, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3);
                add(784 * 50, dEdA1, avr_dEdA1);
                add(50, dEdb1, avr_dEdb1);
                add(5000, dEdA2, avr_dEdA2);
                add(100, dEdb2, avr_dEdb2);
                add(1000, dEdA3, avr_dEdA3);
                add(10, dEdb3, avr_dEdb3);
                cee += cross_entropy_error(y, train_y[index[k]]);
            }
            //新たな平均勾配をえる
            scale(784 * 50, -l_rate/batch_size, avr_dEdA1); 
            scale(50, -l_rate/batch_size, avr_dEdb1);
            scale(5000, -l_rate/batch_size, avr_dEdA2); 
            scale(100, -l_rate/batch_size, avr_dEdb2);
            scale(1000, -l_rate/batch_size, avr_dEdA3); 
            scale(10, -l_rate/batch_size, avr_dEdb3);
            //学習率をかけ、行列、ベクトルの更新
            add(784 * 50, avr_dEdA1, A1);
            add(50, avr_dEdb1, b1);
            add(5000, avr_dEdA2, A2);
            add(100, avr_dEdb2, b2);
            add(1000, avr_dEdA3, A3);
            add(10, avr_dEdb3, b3);
        }
        //ミニバッチ学習終了
        //正解率を表示
        int sum = 0;
        for (int j = 0; j < test_count; j++){
            if(inference6(A1, b1, A2, b2, A3, b3, test_x + j * width * height) == test_y[j]){
                sum++;
            }
        }
        cee /= 60000; //損失関数の平均
        printf("Loss function: %f ", cee);
        printf("Rate: %f%%\n", sum * 100.0 / test_count);
    }
    save("fc1.dat", 50, 784, A1, b1);
    save("fc2.dat", 100, 50, A2, b2);
    save("fc3.dat", 10, 100, A3, b3);
    free(y);
    free(dEdA3);
    free(dEdb3);
    free(dEdA1);
    free(dEdb1);
    free(dEdA2);
    free(dEdb2);
    free(A1);
    free(A2);
    free(A3);
    free(b1);
    free(b2);
    free(b3);
    free(avr_dEdA1);
    free(avr_dEdb1);
    free(avr_dEdA2);
    free(avr_dEdb2);
    free(avr_dEdA3);
    free(avr_dEdb3);
    return 0;
}