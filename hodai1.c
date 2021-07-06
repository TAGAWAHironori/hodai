#include "nn.h"

void print(int m, int n, const float * x){
    int i, j;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            printf("%0.4f ", x[n*i+j]);
        }
        printf("\n");
    }
}

int main(void){
    print(1, 10, b_784x10);
    return 0;
}