#include <cs50.h>
#include <stdio.h>
#include <string.h>

// Linear search
int main(void)
{
    string strings[] = {"battleship", "boot", "cannon", "iron", "thimble", "top hat"};

    string s = get_string("String: ");
    for (int i=0; i<6; i++)
    {
        // strcmp returns 0 if both strings are same, otherwise a number positive or negative
        if (strcmp(strings[i], s) == 0)
        {
            printf("Found\n");
            return 0; // If we don't return exit code 0: progm will continue to print Not found as well
        }
    }
    printf("Not found.\n");
    return 1;
}