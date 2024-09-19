#include <cs50.h>
#include <stdio.h>

// Linear search
int main(void)
{
    int numbers[] = {20, 500, 10, 5, 100, 1, 50};

    int n = get_int("Number: ");
    for (int i=0; i<7; i++)
    {
        if (numbers[i] == n)
        {
            printf("Found\n");
            return 0; // If we don't return exit code 0: progm will continue to print Not found as well
        }
    }
    printf("Not found.\n");
    return 1;
}