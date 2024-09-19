#include <string.h>
#include <cs50.h>
#include <stdio.h>

int main(void)
{
    string name = get_string("What's your name: ");

    int n = strlen(name);
    printf("length is: %i\n", n);
}