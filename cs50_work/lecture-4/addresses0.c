#include<stdio.h>

int main(void)
{
    int n = 50;
    int *p = &n; // p becomes a variable countaining the address of variable n
    printf("%p\n", p);
}