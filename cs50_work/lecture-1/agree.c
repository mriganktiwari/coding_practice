#include <cs50.h>
#include <stdio.h>

int main(void)
{
    char c = get_char("Do you agree ? ");
    
    // Single quotes mean character; Double quotes mean more than 1 character
    if (c == 'y')
    {
        printf("Agreed.\n");
    }
    else if (c == 'n')
    {
        printf("Not agreed.\n");
    }
}