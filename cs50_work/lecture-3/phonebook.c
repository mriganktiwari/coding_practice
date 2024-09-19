#include <cs50.h>
#include <string.h>
#include <stdio.h>

int main(void)
{
    string names[] = {"Carter", "David"};
    string numbers[] = {"+1-617-495-1000", "+1-949-468-2750"};

    string name = get_string("Name: ");

    for (int i=0; i<2; i++)
    {
        if (strcmp(names[i], name) == 0)
        {
            printf("Found: %s\n", numbers[i]);
            return 0;
        }
    }
    printf("Not found.");
    return 1;
}

// Because this is not a good way of storing information, we transition to Data Sctructures