#include <stdio.h>
#include <assert.h>

void test_addition() {
    int a = 2, b = 3;
    assert(a + b == 5);
}

int main() {
    test_addition();
    printf("All tests passed.\n");
    return 0;
}
