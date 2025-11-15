import java.util.*;

public class KnapsackDP {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Step 1: Take user input
        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        int[] weights = new int[n];
        int[] values = new int[n];

        System.out.println("\nEnter weight and value (profit) for each item separated by space:");
        for (int i = 0; i < n; i++) {
            System.out.print("Item " + (i + 1) + ": ");
            weights[i] = sc.nextInt();
            values[i] = sc.nextInt();
        }

        System.out.print("\nEnter maximum capacity of knapsack: ");
        int W = sc.nextInt();

        // Step 2: Create DP table
        int[][] dp = new int[n + 1][W + 1];

        // Step 3: Build table bottom-up
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= W; w++) {
                if (weights[i - 1] <= w) {
                    int include = values[i - 1] + dp[i - 1][w - weights[i - 1]];
                    int exclude = dp[i - 1][w];
                    dp[i][w] = Math.max(include, exclude);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }

        // Step 4: Output result
        System.out.println("\n Maximum profit that can be obtained: " + dp[n][W]);

        sc.close();
    }
}
