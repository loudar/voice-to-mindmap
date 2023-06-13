import json
import matplotlib.pyplot as plt
from scipy import stats


def draw_benchmark_plot(filename):
    # Load data from file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Separate keys and values into lists for plotting
    x = list(data.keys())
    y = list(data.values())

    # Convert keys to integers (if they are not already)
    x = [int(i) for i in x]

    # Convert values to floats (if they are not already)
    y = [float(i) for i in y]

    # Perform linear regression
    slope, intercept, _, _, _ = stats.linregress(x, y)

    # Create a range of x values for the regression line
    x_range = list(range(min(x), int(max(x)*2)))  # Extend the x range to 200% of the original max x value

    # Calculate corresponding y values for the regression line
    y_range = [slope*xi + intercept for xi in x_range]

    # Plot data and regression line
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x_range, y_range, 'r', label='Fitted line')

    # Add labels and title
    plt.xlabel('Amount of Words')
    plt.ylabel('Time (seconds)')
    plt.title('Benchmark Plot')
    plt.legend()

    # Show plot
    plt.show()
