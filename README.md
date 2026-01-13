# Little Notes

A comprehensive collection of notes on machine learning, scientific computing, mathematics, and statistics, built with Jupyter Book.

## Overview

This book covers four main areas:

- **Machine Learning**: Supervised learning, unsupervised learning, deep learning, and model evaluation
- **Elementary Scientific Computing**: NumPy, data manipulation, visualization, and numerical methods
- **Mathematics**: Linear algebra, calculus, optimization, and probability theory
- **Statistics**: Descriptive statistics, inferential statistics, hypothesis testing, and regression analysis

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd little-notes
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Building the Book

To build the HTML version of the book:

```bash
jupyter-book build .
```

The generated HTML files will be in `_build/html/`. Open `_build/html/index.html` in your browser to view the book.

## Cleaning Build Files

To remove previous build files:

```bash
jupyter-book clean .
```

To remove all build files including cached outputs:

```bash
jupyter-book clean --all .
```

## Publishing

### Automatic Deployment with GitHub Actions

This repository includes a GitHub Actions workflow that automatically builds and deploys the book to GitHub Pages when you push to the `main` branch.

#### Setup Instructions:

1. Push your repository to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

2. Enable GitHub Pages in your repository settings:
   - Go to your repository on GitHub
   - Navigate to **Settings** → **Pages**
   - Under **Source**, select **GitHub Actions**

3. The workflow will automatically run on every push to `main` branch
   - You can also trigger it manually from the **Actions** tab
   - Once complete, your book will be available at: `https://<username>.github.io/<repository-name>/`

#### Workflow Details:

The `.github/workflows/deploy.yml` file includes:
- Automatic builds on push to main
- Python 3.11 environment setup
- Dependency caching for faster builds
- Automatic deployment to GitHub Pages

### Manual Deployment (Alternative)

If you prefer to deploy manually:

```bash
pip install ghp-import
jupyter-book build .
ghp-import -n -p -f _build/html
```

## Development

### Adding New Content

1. Create new markdown (.md) or Jupyter notebook (.ipynb) files in the appropriate section folder
2. Update `_toc.yml` to include the new files
3. Rebuild the book

### Customization

- Edit `_config.yml` to change book settings, title, author, etc.
- Edit `_toc.yml` to modify the table of contents structure
- Add custom CSS in `_static/` folder (create if needed)

## Project Structure

```
little-notes/
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions workflow
├── _config.yml              # Book configuration
├── _toc.yml                 # Table of contents
├── .gitignore               # Git ignore file
├── intro.md                 # Landing page
├── references.bib           # Bibliography
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── machine-learning/        # Machine learning section
│   ├── index.md
│   ├── supervised-learning.md
│   ├── unsupervised-learning.md
│   ├── deep-learning.md
│   └── model-evaluation.md
├── scientific-computing/    # Scientific computing section
│   ├── index.md
│   ├── numpy-basics.md
│   ├── data-manipulation.md
│   ├── visualization.md
│   └── numerical-methods.md
├── mathematics/             # Mathematics section
│   ├── index.md
│   ├── linear-algebra.md
│   ├── calculus.md
│   ├── optimization.md
│   └── probability.md
└── statistics/              # Statistics section
    ├── index.md
    ├── descriptive-statistics.md
    ├── inferential-statistics.md
    ├── hypothesis-testing.md
    └── regression-analysis.md
```

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
