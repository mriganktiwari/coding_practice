#include <cs50.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

int main(void)
{
    char *s = get_string("s: ");
    // if string happen to be just NULL (not NUL), we must handle it by returning 1
    if (s == NULL)
    {
        return 1;
    }

    // char *t = s; // Copies the address of first byte of string s into t
    char *t = malloc(strlen(s) + 1); // +1 is for the NUL character

    // for (int i = 0, n = strlen(s) + 1; i < n; i++)
    // {
    //     t[i] = s[i];
    // }
    strcpy(t, s);
    if (t == NULL)
    {
        return 1;
    }

    if (strlen(t) > 0)
    {
        t[0] = toupper(t[0]);
    }

    printf("s: %s\n", s);
    printf("t: %s\n", t);

    free(t);

    return 0;
}