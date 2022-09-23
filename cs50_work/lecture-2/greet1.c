#include <cs50.h>
#include <stdio.h>

// We see how to enable command line arguments
int main(int argc, string argv[]) //Now main will take command line arguments
// argc - argument count
// argv = array or all arguments
{
    printf("Hello, %s %s\n", argv[1], argv[2]);
    // Using argv[0] here - would just give the program name (here ./greet)
}