#include <cs50.h>
#include <stdio.h>

// int main(void)
// {
//     for (int i=0; i<4; i++)
//     {
//         printf("?");
//     }
//     printf("\n");
// }

// int main(void)
// {
//     for (int i=0; i<3; i++)
//     {
//         printf("#\n");
//     }
// }

// int main(void)
// {
//     for (int i=0; i<3; i++)
//     {
//         for (int j=0; j<3; j++)
//         {
//             printf("#");
//         }
//         printf("\n");
//     }
// }

// int main(void)
// {
//    const int n = 5; // const will fix n as a constant, no line of code in future could change it
   
//    for (int i=0; i<n; i++)
//     {
//         for (int j=0; j<n; j++)
//         {
//             printf("#");
//         }
//         printf("\n");
//     }
// }

// int main(void)
// {
//    int n = get_int("Size: ");

//    // conditionals to handle the negative integers entered by user
//    while (n < 1)
//    {
//     n = get_int("Size: ");
//    }

//    for (int i=0; i<n; i++)
//     {
//         for (int j=0; j<n; j++)
//         {
//             printf("#");
//         }
//         printf("\n");
//     }
// }


// Using the same thing as above, but not repeating a line of code n = get_int("Size: ")
int main(void)
{
   int n;
   do
   {
    n = get_int("Size: ");
   }
   while (n < 1);

   for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            printf("#");
        }
        printf("\n");
    }
}