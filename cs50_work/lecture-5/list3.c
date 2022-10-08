// Implement a linked list with prepend implementaion; while loop to print
// Stack like formation

#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int number;
    struct node *next;
}
node;

int main(int argc, char *argv[])
{
    node *list = NULL;

    // For each command line argument
    for (int i = 1; i<argc; i++) // i starts from 1 here; coz, argv's 1st element if the name of program
    {
        // converting the strings in argv to integers
        int number = atoi(argv[i]);

        // Allocate node for number
        node *n = malloc(sizeof(node));
        if (n == NULL)
        {
            return 1;
        }

        n->number = number;
        n->next = NULL;

        // Prepend node to list
        n->next = list;
        list = n;
    }

    // a pointer pointing where list is pointing to
    node *ptr = list;

    while (ptr != NULL)
    {
        printf("%i\n", ptr->number);

        // Moving ptr to next node; which will eventually become NULL (last element of the linked list)
        ptr = ptr->next;
    }

    // Free memory
    ptr = list;
    while (ptr != NULL)
    {
        node *next = ptr->next;
        free(ptr);
        ptr = next;
    }
}