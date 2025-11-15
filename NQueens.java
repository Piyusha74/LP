import java.util.*;

public class NQueens {

    int size;
    boolean[][] board;
    int count = 0;
    int fixedRow = -1, fixedCol = -1;

    // Constructor
    NQueens() {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter size of chessboard: ");
        size = sc.nextInt();

        board = new boolean[size][size];
    }

    // Print board
    void printBoard() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                System.out.print(board[i][j] ? "Q " : "X ");
            }
            System.out.println();
        }
        System.out.println();
    }

    // Check if safe to place queen
    boolean isSafe(int row, int col) {

        // Check column
        for (int i = 0; i < size; i++)
            if (board[i][col])
                return false;

        // Back slash (\) diagonal upper
        int i = row, j = col;
        while (i >= 0 && j >= 0) {
            if (board[i][j]) return false;
            i--;
            j--;
        }

        // Back slash (\) diagonal lower
        i = row; j = col;
        while (i < size && j < size) {
            if (board[i][j]) return false;
            i++;
            j++;
        }

        // Forward slash (/) diagonal upper
        i = row; j = col;
        while (i >= 0 && j < size) {
            if (board[i][j]) return false;
            i--;
            j++;
        }

        // Forward slash (/) diagonal lower
        i = row; j = col;
        while (i < size && j >= 0) {
            if (board[i][j]) return false;
            i++;
            j--;
        }

        return true;
    }

    // Set the first queen manually
    void setFirstQueen() {
        Scanner sc = new Scanner(System.in);

        System.out.println("Enter coordinates of first queen:");
        System.out.print("Enter row (1-" + size + "): ");
        fixedRow = sc.nextInt() - 1;

        System.out.print("Enter column (1-" + size + "): ");
        fixedCol = sc.nextInt() - 1;

        board[fixedRow][fixedCol] = true;
        printBoard();
    }

    // Backtracking function
    void solve(int row) {

        if (row == size) {
            count++;
            printBoard();
            return;
        }

        // If fixed queen is in this row → skip placing and move to next row
        boolean rowHasQueen = false;
        for (int j = 0; j < size; j++) {
            if (board[row][j]) {
                rowHasQueen = true;
                break;
            }
        }

        if (rowHasQueen) {
            solve(row + 1);
            return;
        }

        // Try all columns
        for (int col = 0; col < size; col++) {
            if (isSafe(row, col)) {
                board[row][col] = true;

                solve(row + 1);

                board[row][col] = false; // backtrack
            }
        }
    }

    // Final message
    void displayMessage() {
        if (count > 0)
            System.out.println("Solution exists for the given position of the queen.");
        else
            System.out.println("Solution doesn't exist for the given position of the queen.");
    }

    // Main
    public static void main(String[] args) {
        NQueens solver = new NQueens();
        solver.setFirstQueen();
        solver.solve(0);
        solver.displayMessage();
    }
}
