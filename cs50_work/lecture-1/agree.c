#include <cs50.h>
#include <stdio.h>

int main(void)
{
    char c = get_char("Do you agree ? ");
    
    // Single quotes mean character; Double quotes mean more than 1 character
    // || means the OR operator on conditionals
    if (c == 'y' || c == 'Y')
    {
        printf("Agreed.\n");
    }
    else if (c == 'n' || c == 'N')
    {
        printf("Not agreed.\n");
    }
}