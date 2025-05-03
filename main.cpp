#include <iostream>
#define NUM 1
#define PI 3.1415926
#define add(x, y) ((x) + (y))
#define minus(x, y) ((x) - (y))
#if NUM > 1
#define NUM 1
#elif NUM == 1
#define NUM -1
#else
#define NUM 0
#endif
using namespace std;

void printMessage(const char* msg) {
    cout << "[消息]" << msg << endl;
}

string to_string(int num) {
    string str_num="";
    while(num>0){
        str_num += (char)((num % 10) + '0');
        num/=10;
    }
    return str_num;
}

int main() {
    int num = NUM;
    float pi = PI;
    num = add(1, add(1,add(1,num)));
    num = minus(minus(minus(num,1),1), 1);
    if(num>0){
        printMessage("num大于0");
    }
    while(num-->0){
        printMessage(to_string(num));
    }
    printMessage("PI的值是：" + to_string(pi));
    printMessage("程序结束");
    return 0;
}

