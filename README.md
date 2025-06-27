# Orchestration_LM_BF

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 0.19.12`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)

# 🚀 Orchestration_LM_BF

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Ce document décrit comment cloner, exécuter, tester et packager le projet avec Kedro, Docker et GitHub Actions.

---

## 1️⃣ Initialiser le dépôt Git

```bash
git init
git remote add origin https://github.com/BFAUCHERY/Orchestration_LM_BF.git
git pull origin main
```

---

## 2️⃣ Activer github actions / push des modifications

Les workflows GitHub Actions sont automatiquement déclenchés lors des pushes.

```bash
git add .
git commit -m "Mise à jour du projet"
git push origin main
```

🔗 **Voir l'état des actions :**
👉 [GitHub Actions](https://github.com/BFAUCHERY/Orchestration_LM_BF/actions)

---

## 3️⃣ Mettre à jour l'image Docker et relancer le conteneur

```bash
docker pull ludovicmarion/orchestration_lm_bf:latest
docker run -p 5001:5001 ludovicmarion/orchestration_lm_bf:latest```

---

## 4️⃣ Exécuter les tests

```bash
pytest --cov=src --cov-fail-under=80
```

---

## 5️⃣ Lancer le pipeline Kedro

```bash
kedro run
```

---

✅ **Notes**
- Les dépendances sont dans `requirements.txt` (installer via `pip install -r requirements.txt`)