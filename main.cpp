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

int pow(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; i++) {
        result *= base;
    }
    return result;
}

string to_string(int num) {
    string str_num="";
    int ori=num;
    int digit=0; // 记录数字的位数
    if(num<0){
        str_num += '-';
        num = -num;
    }
    if(num==0){
        return "0";
    }
    while(num!=0){
        digit++;
        num/=10;
    }
    num=ori;
    while(digit!=0){
        str_num+=(char)(num/pow(10,digit-1)+'0');
        num-=(num/pow(10,digit-1))*pow(10,digit-1);
        digit--;
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
    else{
    printMessage("num小于等于0");
    }
    while(num-->0){
        printMessage(to_string(num));
    }
    printMessage("PI的值是：" + to_string(pi));
    printMessage("程序结束");
    return 0;
}
