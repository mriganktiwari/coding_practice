// #include <cs50.h>
#include <stdio.h>

int main(void)
{   
    // pointer arithmatic
    char *s = "HI!";
    printf("%c\n", s[0]);
    printf("%c\n", *s);
    printf("%c\n", s[1]);
    printf("%c\n", *(s+1));
    printf("%c\n", s[2]);
    printf("%c\n", *(s+2));
    printf("%s\n", s); // prints string starting from address in s
    printf("%s\n", s+1);
    printf("%s\n", s+2);
}