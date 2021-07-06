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
    float * train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
    float *y = malloc(sizeof(float) * 10);
    fc(10, 784, train_x, A_784x10, b_784x10, y);
    relu(10, y, y);
    softmax(10, y, y);
    print(1, 10, y);
    free(y);
    return 0;
}