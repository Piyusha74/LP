import java.util.Scanner;

public class RecursiveFibonacci {

    // Recursive Fibonacci function
    public static int fib(int n) {
        if (n <= 1)
            return n;
        return fib(n - 1) + fib(n - 2);
    }

    // Main method
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the number of terms: ");
        int n = sc.nextInt();

        System.out.print("Fibonacci Series: ");
        long start = System.nanoTime();
        for (int i = 0; i < n; i++) {
            System.out.print(fib(i) + " ");
        }
        long end = System.nanoTime();

        double time = (end - start) / 1000.0; // in microseconds
        System.out.printf("\n\nTime Taken: %.2f microseconds%n", time);

        System.out.println("\n=== Complexity Analysis ===");
        System.out.println("Time Complexity: O(2^n)");
        System.out.println("Space Complexity: O(n)");

        sc.close();
    }
}
