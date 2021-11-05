#ifndef COMMON_STRUCT_H 
#define COMMON_STRUCT_H

#include <sys/time.h>    
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
typedef void* mat_cv;
typedef void* cap_cv;
typedef void* write_cv;
typedef struct image {
int w;
int h;
int c;
float *data;
} image;
void calloc_error();
void *xcalloc(size_t nmemb, size_t size);
#endif
