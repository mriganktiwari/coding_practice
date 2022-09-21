#include <stdio.h>
#include <cs50.h>

int main(void)
{
    // Get scores
    int scores[3];
    for (int i = 0; i<3; i++)
    {
        scores[i] = get_int("Score: ");
    }

    // Print average
    printf("Average is: %f\n", (scores[0] + scores[1] + scores[2]) / 3.0);
}