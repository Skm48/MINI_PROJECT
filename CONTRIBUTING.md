# Contributing — Team Guide

## Team setup (one-time)

### Step 1: Create a GitHub account
1. Go to [github.com/signup](https://github.com/signup)
2. Use your university email for the free Pro tier
3. Set a username (e.g. `firstname-lastname` or `firstnameL`)
4. **Send your username to Kee** so they can add you as a collaborator

### Step 2: Accept the invite
- Once Kee adds you, check your email for a GitHub invitation
- Click "Accept invitation" — you now have push access

### Step 3: Install Git
- **Windows:** Download from [git-scm.com](https://git-scm.com/download/win), install with defaults
- **Mac:** Run `xcode-select --install` in Terminal
- **Linux:** `sudo apt install git`

Configure your identity (run once):
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@university.ac.uk"
```

### Step 4: Clone the repo
```bash
git clone https://github.com/Skm48/MINI_PROJECT.git
cd MINI_PROJECT
```

### Step 5: Colab setup
1. Open [Google Colab](https://colab.research.google.com)
2. Go to `File → Open notebook → GitHub`
3. Paste the repo URL and select your notebook
4. Or use the first cell in any notebook to clone automatically

---

## Work allocation

| Member | Model | Branch | Additional responsibility |
|--------|-------|--------|--------------------------|
| **Kee (Member 1)** | VGG16 | `feat/vgg16` | Repo setup, data pipeline, fusion lead |
| **Member 2** | ResNet50 | `feat/resnet50` | Comparison charts, metrics logging |
| **Member 3** | EfficientNet-B0 | `feat/efficientnet` | EDA notebook, README polish |

### What each member does

**Your model tasks (everyone does these for their assigned model):**
1. Train with frozen backbone → log metrics to MLflow
2. Fine-tune last conv block → log metrics
3. Evaluate on test set → save confusion matrix + classification report
4. Generate Grad-CAM heatmaps → save to `outputs/gradcam/`
5. Push results to your feature branch

**Shared tasks (after all 3 models are done):**
- Feature extraction + fusion classifier (Kee leads)
- 3-model comparison table + charts
- Final documentation

---

## Git workflow

### Golden rules
1. **Never push directly to `main`** — always use a feature branch
2. **Pull before you push** — avoids merge conflicts
3. **One model = one branch** — keeps work isolated

### Daily workflow
```bash
# Start of session: get latest changes
git checkout main
git pull origin main

# Switch to your branch (or create it first time)
git checkout -b feat/resnet50    # first time only
git checkout feat/resnet50       # after that

# Do your work...

# Save progress
git add .
git commit -m "resnet50: add training loop with frozen backbone"
git push origin feat/resnet50
```

### When your model is done
```bash
# Make sure you're up to date
git checkout main
git pull origin main
git checkout feat/resnet50
git merge main                   # bring in any changes from main

# Push and create a Pull Request on GitHub
git push origin feat/resnet50
# Then go to GitHub → "Compare & pull request" → assign Kee to review
```

### Commit message format
```
model: short description

Examples:
  vgg16: add frozen backbone training loop
  resnet50: fix learning rate scheduler bug
  efficientnet: add grad-cam heatmap generation
  shared: implement fusion classifier head
  docs: update README with results table
```

---

## Branch naming

| Branch | Purpose | Owner |
|--------|---------|-------|
| `main` | Stable, reviewed code only | Protected |
| `setup/data-pipeline` | Initial data + preprocessing | Kee |
| `feat/vgg16` | VGG16 training + eval + gradcam | Member 1 |
| `feat/resnet50` | ResNet50 training + eval + gradcam | Member 2 |
| `feat/efficientnet` | EfficientNet training + eval + gradcam | Member 3 |
| `feat/fusion` | Feature-level fusion model | Kee (all contribute) |
| `feat/gradcam` | Grad-CAM comparison grid | All |

---

## File ownership (avoid conflicts)

Each member should primarily edit these files:

| Member 1 (Kee) | Member 2 | Member 3 |
|-----------------|----------|----------|
| `src/dataset.py` | `src/evaluate.py` | `notebooks/01_setup_eda.ipynb` |
| `src/models.py` | `src/train.py` | `README.md` |
| `src/fusion.py` | Comparison charts | `src/gradcam.py` |
| `configs/config.yaml` | MLflow helpers | EDA visualisations |

**Notebooks:** Each person works in their own model section within `02_baselines.ipynb`, or creates a separate notebook per model to avoid merge conflicts entirely.

---

## Colab tips for teams

### Saving work back to GitHub from Colab
```python
# Run this cell to commit and push from Colab
import os
os.chdir('/content/MINI_PROJECT')

!git add .
!git commit -m "resnet50: training complete, metrics logged"
!git push origin feat/resnet50
```

### Authentication in Colab
```python
# Use a Personal Access Token (PAT) — not your password
# Generate one at: github.com/settings/tokens → "Generate new token (classic)"
# Tick "repo" scope → copy the token

# When git asks for password, paste the token instead
```

### Saving checkpoints to Google Drive (recommended)
```python
# Model files are too large for GitHub
# Save to Drive instead:
import shutil
shutil.copy('models/checkpoints/vgg16_best.pth',
            '/content/drive/MyDrive/MINI_PROJECT/checkpoints/')
```
