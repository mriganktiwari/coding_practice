#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int scores[1024];

    for (int i = 0; i < 1024; i++)
    {
        // All the values printed here are garbage; as we did not do any assignment
        printf("%i\n", scores[i]);
    }
}