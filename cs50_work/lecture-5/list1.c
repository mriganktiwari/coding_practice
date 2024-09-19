// Implements a list of numbers with an array of dynamic size (using "malloc")

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int *list = malloc(3 * sizeof(int));
    if (list == NULL)
    {
        // free(list);
        return 1;
    }

    list[0] = 1;
    list[1] = 2;
    list[2] = 3;

    // Making it a list of size of 4 instead
    int *tmp = malloc(4 * sizeof(int));
    if (tmp == NULL)
    {
        free(list);
        return 1;
    }

    // Copy list of size 3 into list of size 4
    for (int i = 0; i<3; i++)
    {
        tmp[i] = list[i];
    }

    // add 4 to list of size 4
    tmp[3] = 4;

    free(list);

    list = tmp;

    for (int i = 0; i<4; i++)
    {
        printf("%i\n", list[i]);
    }

    free(list);

    return 0;

}