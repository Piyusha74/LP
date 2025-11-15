import java.util.*;

public class HuffmanCoding {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of characters: ");
        int n = sc.nextInt();

        char[] chars = new char[n];
        int[] freq = new int[n];

        // Input characters and frequencies
        for (int i = 0; i < n; i++) {
            System.out.print("Enter character: ");
            chars[i] = sc.next().charAt(0);
            System.out.print("Enter frequency: ");
            freq[i] = sc.nextInt();
        }

        // Sort by frequency (ascending)
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (freq[j] > freq[j + 1]) {
                    int temp = freq[j];
                    freq[j] = freq[j + 1];
                    freq[j + 1] = temp;

                    char t = chars[j];
                    chars[j] = chars[j + 1];
                    chars[j + 1] = t;
                }
            }
        }

        // Display simple "greedy" Huffman-style binary codes
        System.out.println("\nHuffman Codes (Greedy Approach):");
        for (int i = 0; i < n; i++) {
            String code = "";
            for (int j = 0; j <= i; j++)
                code += "0";
            if (i != n - 1) code = code.substring(0, code.length() - 1) + "1";
            System.out.println(chars[i] + " -> " + code);
        }

        sc.close();
    }
}
