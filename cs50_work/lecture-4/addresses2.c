// #include <cs50.h>
#include <stdio.h>

int main(void)
{
    char *s = "HI!";
    printf("Address is: %p\n", s); // %p shows pointer
    printf("String is: %s\n", s);
    printf("%p\n", &s[0]);
    printf("%p\n", &s[1]);
    printf("%p\n", &s[2]);
    printf("%p\n", &s[3]);
}