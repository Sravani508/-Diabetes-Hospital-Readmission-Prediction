# Working with Visualizations

This document explains how to generate, save, and use visualizations in your project documentation.

## 📊 Available Visualizations

The project includes the following visualizations:

1. **ROC Curve** (`roc_curve.png`) - Shows model discrimination ability
2. **Precision-Recall Curve** (`precision_recall_curve.png`) - Performance on imbalanced data
3. **Confusion Matrix** (`confusion_matrix.png`) - Prediction breakdown by class
4. **Model Comparison** (`model_comparison.png`) - Side-by-side metric comparison
5. **Feature Importance** (`feature_importance.png`) - Top contributing features
6. **Class Distribution** (`class_distribution.png`) - Target variable balance

## 🎨 Generating Visualizations

### From Notebook

Run the "📊 Save Visualizations for README" cell in your notebook after training models:

```python
# This cell will:
# 1. Extract predictions from trained models
# 2. Generate all visualization plots
# 3. Save high-resolution PNG files to images/ directory
# 4. Display summary of created files
```

### From Python Script

```python
from src.evaluation import evaluate_and_compare
from src.visualization import save_all_visualizations  # You can create this module

# After training and evaluation
best_name, best_model, predictions, comparison = evaluate_and_compare(results, spark)

# Save visualizations
save_all_visualizations(
    predictions=predictions,
    comparison_df=comparison,
    output_dir="images/"
)
```

## 📁 Image Directory Structure

```
images/
├── roc_curve.png              # ROC curve with AUC score
├── precision_recall_curve.png # PR curve for imbalanced data
├── confusion_matrix.png       # Confusion matrix heatmap
├── model_comparison.png       # Bar charts comparing models
├── feature_importance.png     # Top feature importances
└── class_distribution.png     # Target variable distribution
```

## 🖼️ Using Images in Documentation

### In Markdown (README.md)

```markdown
# Relative path from README location
![ROC Curve](images/roc_curve.png)

# With alt text and title
![ROC Curve](images/roc_curve.png "Model ROC Curve")

# With caption
![ROC Curve](images/roc_curve.png)
*Figure 1: ROC Curve showing model discrimination ability (AUC = 0.6494)*
```

### In Jupyter Notebooks

```python
from IPython.display import Image, display

# Display single image
display(Image(filename='images/roc_curve.png'))

# Display multiple images
for img in ['roc_curve.png', 'confusion_matrix.png']:
    display(Image(filename=f'images/{img}'))
```

### In HTML Documentation

```html
<img src="images/roc_curve.png" alt="ROC Curve" width="600">

<!-- With caption -->
<figure>
  <img src="images/roc_curve.png" alt="ROC Curve" width="600">
  <figcaption>Figure 1: ROC Curve (AUC = 0.6494)</figcaption>
</figure>
```

## 🎯 Best Practices

### Image Quality

* **Resolution**: Save at 300 DPI for publication quality
* **Format**: PNG for charts (lossless), JPEG for photos
* **Size**: Keep under 1MB for web use
* **Dimensions**: 800-1200px width for README images

### File Naming

* Use descriptive, lowercase names with underscores
* Include metric values in filename if helpful
* Examples:
  - `roc_curve_auc_0.6494.png`
  - `confusion_matrix_rf_model.png`
  - `feature_importance_top15.png`

### Git Management

The `.gitignore` file is configured to:
* ✅ **Include** images in `images/` directory (for documentation)
* ❌ **Exclude** large data files and model binaries
* ❌ **Exclude** temporary plot files

```gitignore
# In .gitignore
# Include documentation images
!images/*.png
!images/*.jpg

# Exclude temporary plots
*.tmp.png
plots/temp/
```

## 🔄 Updating Visualizations

### When to Regenerate

* After retraining models with new data
* When hyperparameters change significantly
* If performance metrics improve
* For different model versions

### Automation Script

Create a script to regenerate all visualizations:

```python
# scripts/generate_visualizations.py
import sys
sys.path.append('src')

from preprocessing import preprocess_data
from train import train_all_models, prepare_train_test_split
from evaluation import evaluate_and_compare
from visualization import save_all_visualizations

def main():
    # Load and preprocess
    df, num_feat, bin_feat, cat_feat = preprocess_data(spark, "data.csv")
    
    # Train
    train_df, test_df = prepare_train_test_split(df)
    results, pipeline = train_all_models(train_df, test_df, num_feat, bin_feat, cat_feat)
    
    # Evaluate
    best_name, best_model, predictions, comparison = evaluate_and_compare(results, spark)
    
    # Generate visualizations
    save_all_visualizations(predictions, comparison, "images/")
    print("✅ All visualizations updated!")

if __name__ == "__main__":
    main()
```

## 📐 Customizing Plots

### Style Configuration

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set global style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom colors
COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12'
}

# Font settings
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
```

### Plot Templates

```python
def create_styled_plot(figsize=(10, 8)):
    """Create a consistently styled plot."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    return fig, ax

# Usage
fig, ax = create_styled_plot()
ax.plot(x, y, linewidth=2.5, color='#3498db')
ax.set_xlabel('X Label', fontsize=13, fontweight='bold')
ax.set_ylabel('Y Label', fontsize=13, fontweight='bold')
ax.set_title('Plot Title', fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images/my_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

## 🌐 GitHub Display

### Image Sizing in README

```markdown
<!-- Default size -->
![ROC Curve](images/roc_curve.png)

<!-- Specific width (GitHub supports HTML) -->
<img src="images/roc_curve.png" width="600" alt="ROC Curve">

<!-- Side by side -->
<p float="left">
  <img src="images/roc_curve.png" width="400" />
  <img src="images/precision_recall_curve.png" width="400" />
</p>
```

### Image Links

```markdown
<!-- Clickable image -->
[![ROC Curve](images/roc_curve.png)](images/roc_curve.png)

<!-- Link to section -->
[![ROC Curve](images/roc_curve.png)](#performance-analysis)
```

## 🐛 Troubleshooting

### Issue: Images not displaying on GitHub

**Solution:**
* Ensure images are committed to repository
* Check file paths are relative to README location
* Verify image files are not in `.gitignore`
* Use forward slashes `/` in paths (not backslashes)

### Issue: Images too large

**Solution:**
```python
# Reduce DPI
plt.savefig('image.png', dpi=150)  # Instead of 300

# Compress existing images
from PIL import Image
img = Image.open('image.png')
img.save('image.png', optimize=True, quality=85)
```

### Issue: Plots look different on GitHub

**Solution:**
* Use explicit figure sizes: `plt.figure(figsize=(10, 8))`
* Set DPI explicitly: `plt.savefig(..., dpi=300)`
* Use `bbox_inches='tight'` to avoid clipping
* Test locally before committing

## 📚 Additional Resources

* **Matplotlib Documentation**: https://matplotlib.org/stable/gallery/index.html
* **Seaborn Gallery**: https://seaborn.pydata.org/examples/index.html
* **GitHub Markdown Guide**: https://guides.github.com/features/mastering-markdown/
* **Image Optimization**: https://tinypng.com/

---

**Pro Tip**: Keep a `visualization_config.py` file with all your plot styling constants for consistency across all visualizations!
