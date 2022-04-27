#include "nn.h"

void load(const char * filename, int m, int n, const float *A, const float *b){
    FILE *fp = NULL;
    fp = fopen(filename, "rb");
    fread(A, sizeof(float), m * n ,fp);
    fread(b, sizeof(float), n, fp);
    fclose(fp);
}

void add(int n, const float * x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] += x[i];
    }
}

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
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b1 = malloc(sizeof(float) * 50);
    float *b2 = malloc(sizeof(float) * 100);
    float *b3 = malloc(sizeof(float) * 10);
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
    int sum = 0;
    int i;
    load("fc1.dat", 50, 784, A1, b1);
    load("fc2.dat", 100, 10, A2, b2);
    load("fc3.dat", 10, 100, A3, b3);
    for (i = 0; i < test_count; i++){
        if(inference6(A1, b1, A2, b2, A3, b3, test_x + i*width*height) == test_y[i]){
            sum++;
        }
    }
    printf("%f%%\n", sum * 100.0 / test_count);
    free(A1);
    free(A2);
    free(A3);
    free(b1);
    free(b2);
    free(b3);
    return 0;
}