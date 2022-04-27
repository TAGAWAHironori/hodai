#include "nn.h"

//ランダムシャッフル
void shuffle(int n, int *x){
     for(int i = 0; i < n; i++) {
        int j = rand() % n;
        int t = x[i];
        x[i] = x[j];
        x[j] = t;
     }
}

int main(){
    int i;
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
    int *index = malloc(sizeof(int) * train_count);
    for (i = 0; i < train_count; i++){
        index[i] = i;
    }
    shuffle(train_count, index);
    return 0;
}