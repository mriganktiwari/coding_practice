#include <cs50.h>
#include <stdio.h>

// We see how to enable command line arguments
int main(int argc, string argv[]) //Now main will take command line arguments
// argc - argument count
// argv = array or all arguments
{
    if (argc==2)
    {
        printf("Hello, %s\n", argv[1]);
    }
    else
    {
        printf("Hello, world\n");
    }

}