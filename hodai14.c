#include "nn.h"

void add(int n, const float * x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] += x[i];
    }
}

void scale(int n, float x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] += x;
    }
}

void init(int n, float x, float * o){
    int i;
    for (i = 0; i < n; i++){
        o[i] = x;
    }
}

void rand_init(int n, float *o){
    int i;
    for (i = 0; i < n; i++){
        o[i] = ((float)rand() / RAND_MAX)*2 - 1;
    }
}