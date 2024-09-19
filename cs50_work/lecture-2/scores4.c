#include <cs50.h>
#include <stdio.h>

const int N = 3;

float average(int length, int array[]);

int main(void)
{
    // Get scores
    int scores[N];
    for (int i=0; i<N; i++)
    {
        scores[i] = get_int("Scores: ");
    }

    // Print average
    printf("Average is: %f\n", average(N, scores));
}

float average(int length, int array[])
{
    // Calculate average
    int sum = 0;
    for (int i=0; i<length; i++)
    {
        sum += array[i];
    }

    return sum / (float) length;
}