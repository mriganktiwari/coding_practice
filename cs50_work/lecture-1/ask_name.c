#include <cs50.h>
#include <stdio.h>

int main(void)
{
    string name = get_string("What's your name? ");
    // string last = get_string("What's your last name? ");
    printf("hello, %s\n", name);
}