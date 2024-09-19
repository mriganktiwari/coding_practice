#include <cs50.h>
#include <stdio.h>

int main(void)
{
    string s = get_string("s: ");
    string t = get_string("t: ");

    if (s==t) // Now this is comparing 2 memory addresses
    {
        printf("Same\n");
    }

    else
    {
        printf("Different\n");
    }
}