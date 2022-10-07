// Implements a list of size 4 with an array of dynamic size (using "realloc")

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int *list = malloc(3 * sizeof(int));
    if (list == NULL)
    {
        return 1;
    }

    list[0] = 1;
    list[1] = 2;
    list[2] = 3;

    int *tmp = realloc(list, 4 * sizeof(int));
    // Instead we could do below, but using a tmp is useful to check if we are not out-of-memory and returning NULL
    // realloc(list, 4 * sizeof(int));
    if (tmp == NULL)
    {
        free(list);
        return 1;
    }

    list = tmp;
    list[3] = 4;

    for (int i = 0; i<4; i++)
    {
        printf("%i\n", list[i]);
    }

}