import os

from benchmarking.benchmark_plotter import draw_benchmark_plot

test_sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]

os.chdir("..")

with open("timings/timings.json", 'w') as f:
    f.write("{}")

for size in test_sizes:
    filename = f"benchmarking/source_texts/text_{size}.txt"
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Skipping...")
        continue

    print(f"Running {filename}...")
    os.system(f"python plotter.py {filename}")
    print("Done.")

draw_benchmark_plot("timings/timings.json")
