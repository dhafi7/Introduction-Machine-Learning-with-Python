   ]
  },
  {
   "cell_type": "markdown",
   "id": "fda5b645",
   "metadata": {},
   "source": [
    "## 1.10 The scikit-learn API Convention\n",
    "\n",
    "### Theory: The Estimator Interface\n",
    "\n",
    "All scikit-learn models follow a **consistent API** — the **Estimator interface**:\n",
    "\n",
    "```python\n",
    "# 1. Create model object with hyperparameters\n",
    "model = SomeModel(param1=value1, param2=value2)\n",
    "\n",
    "# 2. Train on training data\n",
    "model.fit(X_train, y_train)       # For supervised\n",
    "model.fit(X_train)                # For unsupervised\n",
    "\n",
    "# 3. Make predictions\n",
    "y_pred = model.predict(X_test)   # Classification / Regression\n",
    "\n",
    "# 4. Evaluate\n",
    "score = model.score(X_test, y_test)\n",
    "```\n",
    "\n",
    "This consistency means: **once you learn one model, you can use them all the same way!**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "4f8da4bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Model accuracy on test set: 1.0000 (100.00%)\n",
      "\n",
      " The 5-step ML workflow:\n",
      "  1. Load Data\n",
      "  2. Explore & Visualize\n",
      "  3. Split (Train / Test)\n",
      "  4. Train Model (.fit)\n",
      "  5. Evaluate (.score / .predict)\n"
     ]
    }
   ],
   "source": [
    "# Summary: The complete workflow in just a few lines!\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Load\n",
    "iris = load_iris()\n",
    "X, y = iris.data, iris.target\n",
    "\n",
    "# Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n",
    "\n",
    "# Train\n",
    "model = KNeighborsClassifier(n_neighbors=3)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluate\n",
    "accuracy = model.score(X_test, y_test)\n",
    "print(f\" Model accuracy on test set: {accuracy:.4f} ({accuracy*100:.2f}%)\")\n",
    "print(\"\\n The 5-step ML workflow:\")\n",
    "print(\"  1. Load Data\")\n",
    "print(\"  2. Explore & Visualize\")\n",
    "print(\"  3. Split (Train / Test)\")\n",
    "print(\"  4. Train Model (.fit)\")\n",
    "print(\"  5. Evaluate (.score / .predict)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9bc1c594",
   "metadata": {},
   "source": [
    "---\n",
    "##  Chapter 1 Summary\n",
    "\n",
    "| Concept | Key Takeaway |\n",
    "|---|---|\n",
    "| **Machine Learning** | Learns patterns from data rather than using handcoded rules |\n",
    "| **Supervised Learning** | Uses labeled input-output pairs; includes classification & regression |\n",
    "| **Unsupervised Learning** | Finds structure in unlabeled data (clustering, dimensionality reduction) |\n",
    "| **Train/Test Split** | Crucial to evaluate generalization — never test on training data |\n",
    "| **k-NN** | Simple model: predicts based on k closest neighbors |\n",
    "| **sklearn API** | Consistent: `fit()` → `predict()` → `score()` for all models |\n",
    "\n",
    "> **We achieved ~97% accuracy on the Iris test set **\n",
    "---\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.14.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
