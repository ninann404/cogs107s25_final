## Referring to Files Relative to the Current Script with `pathlib.Path`

When writing Python programs that generate or load data, figures, or other resources, it can be handy to refer to file paths in a way that is robust to changes in the working directory. The `pathlib` module provides a clean and reliable way to construct paths relative to the location of the current script.

### Why Not Use Relative Paths Directly?

Code like:

```python
open("config.json")
```

relies on the *current working directory*, which may differ depending on how the script is executed. This leads to brittle code when the script is launched from a different location.

### Using `__file__` with `pathlib.Path`

To construct paths relative to a script's location, use the special `__file__` variable with `pathlib.Path`:

```python
from pathlib import Path

here = Path(__file__).resolve().parent
```

* `__file__` refers to the current script.
* `.resolve()` gives the absolute path.
* `.parent` extracts the containing directory.

### Referring to Resources

Once you have `here`, you can build paths to other files reliably:

```python
config_file = here / "config.json"
data_file = here / "data" / "dataset.csv"
```

These paths will work regardless of where the script is launched from.

### Example: Saving a Table and a Figure

Here is a complete example using `pandas` and `matplotlib` to generate and save a data table and a plot, saving both relative to the script's location:

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Define base directory
here = Path(__file__).resolve().parent  # <-- current directory
output_dir = here / "output"            # <-- new subdirectory
output_dir.mkdir(exist_ok=True)

# Example data
df = pd.DataFrame({
    "x": range(10),
    "y": [i ** 2 for i in range(10)]
})

# Save the table
table_path = output_dir / "squares.csv"
df.to_csv(table_path, index=False)

# Save the figure
plt.plot(df["x"], df["y"])
plt.title("y = x^2")
plt.xlabel("x")
plt.ylabel("y")
figure_path = output_dir / "squares_plot.png"
plt.savefig(figure_path)
plt.close()
```

If `output/` does not already exist, it is created automatically by the script.

This use of `Path(__file__).resolve().parent` ensures that file operations like reading data or saving figures will behave consistently, regardless of the working directory.
