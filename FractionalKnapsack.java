import java.util.Scanner;

public class FractionalKnapsack {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Step 1: Input number of items
        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        double[] weight = new double[n];
        double[] value = new double[n];
        double[] ratio = new double[n];

        // Step 2: Input weight and value
        System.out.println("\nEnter weight and value for each item:");
        for (int i = 0; i < n; i++) {
            System.out.print("Item " + (i + 1) + " - Weight and Value: ");
            weight[i] = sc.nextDouble();
            value[i] = sc.nextDouble();
            ratio[i] = value[i] / weight[i];
        }

        // Step 3: Input knapsack capacity
        System.out.print("\nEnter knapsack capacity: ");
        double capacity = sc.nextDouble();

        // Step 4: Sort items by value/weight ratio (descending)
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (ratio[i] < ratio[j]) {
                    // swap ratio
                    double temp = ratio[i];
                    ratio[i] = ratio[j];
                    ratio[j] = temp;

                    // swap weights
                    double t1 = weight[i];
                    weight[i] = weight[j];
                    weight[j] = t1;

                    // swap values
                    double t2 = value[i];
                    value[i] = value[j];
                    value[j] = t2;
                }
            }
        }

        // Step 5: Apply Greedy method
        double totalValue = 0;
        System.out.println("\nItem selection process:");
        for (int i = 0; i < n; i++) {
            if (capacity == 0)
                break;

            if (weight[i] <= capacity) {
                capacity -= weight[i];
                totalValue += value[i];
                System.out.printf("  Took full item (Weight=%.2f, Value=%.2f)%n", weight[i], value[i]);
            } else {
                double fraction = capacity / weight[i];
                totalValue += value[i] * fraction;
                System.out.printf("  Took %.2f fraction of item (Weight=%.2f, Value=%.2f)%n", fraction, weight[i], value[i]);
                capacity = 0;
            }
        }

        // Step 6: Print result
        System.out.printf("%nMaximum value in knapsack = %.2f%n", totalValue);
        sc.close();
    }
}
