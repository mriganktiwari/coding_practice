#include <cs50.h>
#include <string.h>
#include <stdio.h>

int main(void)
{
    string s = get_string("Before: ");
    printf("After: ");

    for (int i=0; i<strlen(s); i++)
    {
        if (s[i] >= 'a' && s[i] <= 'z')
        {
            printf("%c", s[i] - 32);
        }

        else
        {
            printf("%c", s[i]);
        }
    }
    printf("\n");
}