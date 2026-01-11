# Installation and Setup Guide for Python Development

## Table of Contents
1. [Python Interpreter Installation](#python-interpreter-installation)
2. [IDE Installation](#ide-installation)
3. [VSCode Python Extensions](#vscode-python-extensions)
4. [Jupyter Notebook Setup in VSCode](#jupyter-notebook-setup-in-vscode)
5. [Google Colab](#google-colab)

---

## Python Interpreter Installation

### Step 1: Download Python

1. Visit the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click the **Download Python** button (latest version, currently 3.14)
3. Choose the installer for your operating system:
   - **Windows**: Download the Windows installer (executable)
   - **macOS**: Download the macOS installer
   - **Linux**: Use your package manager (apt, yum, etc.)

### Step 2: Install Python on Windows

1. Run the downloaded `.exe` file
2. **Important**: Check the box "Add Python 3.x to PATH" at the bottom
3. Click "Install Now" for default settings, or "Customize installation" for advanced options
4. Wait for the installation to complete
5. Click "Disable path length limit" if prompted (optional but recommended)

### Step 3: Install Python on macOS

1. Run the downloaded `.pkg` file
2. Follow the installation wizard
3. Accept the license agreement
4. Select the installation destination
5. Complete the installation

### Step 4: Install Python on Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# Fedora/RHEL
sudo dnf install python3 python3-pip

# Arch
sudo pacman -S python python-pip
```

### Step 5: Verify Installation

Open your terminal/command prompt and run:

```bash
python --version
pip --version
```

Both commands should display version information without errors.

---

## IDE Installation

### Option 1: Visual Studio Code (VSCode)

#### Installation Steps:

1. Visit [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Click the **Download** button for your operating system
3. Run the installer and follow the installation wizard
4. **Windows users**: Select "Add to PATH" during installation
5. Launch VSCode after installation

#### Initial Setup:

1. Open VSCode
2. Click the Extensions icon on the left sidebar (or press `Ctrl+Shift+X`)
3. Search for "Python"
4. Install the official **Python** extension by Microsoft
5. Reload VSCode when prompted

---

### Option 2: PyCharm

#### Installation Steps:

1. Visit [https://www.jetbrains.com/pycharm/](https://www.jetbrains.com/pycharm/)
2. Choose between:
   - **PyCharm Community Edition** (Free) - Recommended for beginners
   - **PyCharm Professional** (Paid) - Full-featured
3. Download the installer for your OS
4. Run the installer and follow the setup wizard
5. Accept the license agreement
6. Choose installation location and settings
7. Complete the installation

#### Initial Setup:

1. Launch PyCharm
2. Create a new project or open an existing one
3. Select your Python interpreter:
   - Go to **File** → **Settings** → **Project** → **Python Interpreter**
   - Click the gear icon and select "Add..."
   - Choose your Python installation location

---

## VSCode Python Extensions

### Required Extensions

#### 1. Python (by Microsoft)
- Already mentioned above as the primary extension
- Provides IntelliSense, linting, debugging, and code formatting

#### 2. Pylance (Code Intelligence)
- Automatically installed with Microsoft's Python extension
- Provides advanced code analysis and type checking

#### 3. Python Debugger (by Microsoft)
- Included with the Python extension
- Enables breakpoint debugging and variable inspection

#### 4. Jupyter (by Microsoft)
- Enables Jupyter notebook support in VSCode
- Install separately if not included

---

## Jupyter Notebook Setup in VSCode

### Step 1: Install Required Packages

Open your terminal and run:

```bash
pip install jupyter notebook ipykernel
```

### Step 2: Ensure Jupyter Extension is Installed

1. Open VSCode
2. Press `Ctrl+Shift+X` to open Extensions
3. Search for "Jupyter"
4. Install the **Jupyter** extension by Microsoft
5. Reload VSCode

### Step 3: Create and Open a Notebook

**Method 1: Create a new notebook**
1. Press `Ctrl+Shift+P`
2. Type "Jupyter: Create New Blank Notebook"
3. Select the Python kernel when prompted
4. Save the file with `.ipynb` extension

**Method 2: Open existing notebook**
1. Open the notebook file (`.ipynb`) in VSCode
2. Select kernel when prompted

### Step 4: Configure Python Kernel

1. Create a new cell in your notebook
2. Click the kernel selector in the top-right corner
3. Select "Python Environments" or "Select Kernel"
4. Choose your Python interpreter from the list

### Step 5: Run Notebook Cells

1. Write code in a cell
2. Click the play button on the left of the cell, or press `Ctrl+Enter`
3. View output below the cell
4. Markdown cells: Use `Ctrl+Shift+M` to convert a cell to markdown

### Useful Jupyter Keyboard Shortcuts in VSCode

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run current cell |
| `Shift+Enter` | Run current cell and move to next |
| `Alt+Enter` | Run current cell and insert new cell below |
| `Ctrl+Shift+M` | Toggle cell type (code/markdown) |
| `Ctrl+K, Ctrl+C` | Comment/uncomment code |
| `Ctrl+Shift+[` | Collapse cell |
| `Ctrl+Shift+]` | Expand cell |

---

## Google Colab

Google Colab is a free, cloud-based Jupyter notebook environment that requires no local setup.

### Step 1: Access Google Colab

1. Visit [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with your Google account (create one if needed)
3. Click "New notebook" or select from recent notebooks

### Step 2: Create Your First Notebook

1. The default notebook has one empty code cell
2. Write Python code in the cell
3. Press `Ctrl+Enter` to execute the cell
4. Markdown cells work the same as VSCode Jupyter

### Step 3: Save Your Notebook

1. Click "File" in the menu
2. Select "Save" or press `Ctrl+S`
3. Notebook is automatically saved to your Google Drive
4. Name your notebook when prompted

### Step 4: Upload Local Files (Optional)

To work with local files (like CSV datasets):

```python
from google.colab import files
uploaded = files.upload()
```

Or mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 5: Install Packages

Use pip in a cell:

```python
!pip install package_name
```

**Example:**
```python
!pip install pandas numpy scikit-learn
```



### Useful Google Colab Features

- **Free GPU/TPU**: Great for machine learning projects
- **Automatic saving to Google Drive**: Never lose your work
- **Easy sharing**: Click "Share" to collaborate with others
- **Rich output**: Support for plots, DataFrames, and interactive visualizations
- **Pre-installed libraries**: NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow

---

## Quick Start Checklist

- [ ] Python installed and added to PATH
- [ ] Verify Python installation with `python --version`
- [ ] VSCode or PyCharm installed
- [ ] Python extension installed in VSCode (if using VSCode)
- [ ] Jupyter packages installed (`pip install jupyter ipykernel`)
- [ ] Jupyter extension installed in VSCode (if using VSCode)
- [ ] Python interpreter configured in IDE
- [ ] Test with a simple notebook (Hello World)
- [ ] Google Colab account created and tested

---

## Troubleshooting

### Python not found in PATH
- **Windows**: Reinstall Python and check "Add Python to PATH"
- **macOS/Linux**: Add Python to PATH manually in terminal configuration files

### Jupyter kernel not showing in VSCode
```bash
python -m ipykernel install --user
```

### Extension not working in VSCode
- Reload VSCode (`Ctrl+Shift+P` → "Developer: Reload Window")
- Uninstall and reinstall the extension

### Google Colab connection issues
- Ensure you have active internet connection
- Clear browser cache and cookies
- Try a different browser

---

## Additional Resources
- **Basic Python Guide with Practice**: [https://aqemery.github.io/learn-python/basics](https://aqemery.github.io/learn-python/basics)
