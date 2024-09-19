#include <stdio.h>

int main(void)
{
    int score[3];
    score[0] = 72;
    score[1] = 73;
    score[2] = 33;

    printf("Average is: %f\n", (score[0] + score[1] + score[2]) / 3.0);
}