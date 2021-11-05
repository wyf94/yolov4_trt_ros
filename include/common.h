#ifndef COMMON_H 
#define COMMON_H

#include <sys/time.h>    
namespace util {

struct Box {
float start_x = 0;
float start_y = 0;
float end_x = 0;
float end_y = 0;
float score = 0;
float box_conf =0;
float box_class = 0;
};


}
#endif
