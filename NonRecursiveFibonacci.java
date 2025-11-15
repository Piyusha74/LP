import java.util.Scanner;

public class NonRecursiveFibonacci {

    // Non-Recursive Fibonacci Function
    public static void fib(int n) {
        int n1 = 0, n2 = 1;
        System.out.print(n1 + " " + n2 + " ");
        for (int i = 2; i < n; i++) {
            int n3 = n1 + n2;
            System.out.print(n3 + " ");
            n1 = n2;
            n2 = n3;
        }
        System.out.println();
    }

    // Main Method
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter number of terms: ");
        int n = sc.nextInt();

        System.out.print("Fibonacci Series: ");
        long start = System.nanoTime();
        fib(n);
        long end = System.nanoTime();

        double time = (end - start) / 1000.0; // in microseconds
        System.out.printf("\nTime Taken: %.2f microseconds%n", time);

        System.out.println("\n=== Complexity Analysis ===");
        System.out.println("Time Complexity: O(n)");
        System.out.println("Space Complexity: O(1)");

        sc.close();
    }
}
