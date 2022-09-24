#include <cs50.h>
#include <string.h>
#include <stdio.h>

typedef struct
{
    /* data */
    string name;
    string number;
}
person;

int main(void)
{
    person people[2];

    people[0].name = "Carter";
    people[0].number = "+1-617-495-1000";
    people[1].name = "David";
    people[1].number = "+1-949-468-2750";

    string name = get_string("Name: ");
    for (int i=0; i<2; i++)
    {
        if (strcmp(people[i].name, name) == 0)
        {
            printf("Found: %s\n", people[i].number);
            return 0;
        }
    }
    printf("Not found.");
    return 1;
}

// Because this is not a good way of storing information, we transition to Data Sctructures