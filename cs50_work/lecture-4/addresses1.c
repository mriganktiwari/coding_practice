#include <cs50.h>
#include <stdio.h>

// int main(void)
// {
//     string s = "HI!";
//     printf("%s\n", s);
// }

// string is a cs50 thing, which points to the first char
// string s <=> char *s

// Its defined as typedef
// typedef char string;

int main(void)
{
    int n = 50;
    int *p = &n;
    printf("%i\n", *p); // p would just print out address of n, but *p will go to that address and print
}