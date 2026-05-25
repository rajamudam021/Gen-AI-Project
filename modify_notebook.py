import json
import os
import subprocess
import sys

notebook_path = 'Gen_AI_Project_Code(1).ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Modify cell 2 (index 2) - Mount Google Drive (make it conditional)
cell_2 = notebook['cells'][2]
print("Original Cell 2 source:", cell_2['source'])
cell_2['source'] = [
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "except ImportError:\n",
    "    print(\"Not running in Google Colab. Skipping Google Drive mount.\")\n"
]

# Modify cell 11 (index 11) - Load CSVs (generate them locally if missing)
cell_11 = notebook['cells'][11]
print("Original Cell 11 source:", cell_11['source'])
cell_11['source'] = [
    "# Load the dataset into pandas DataFrames\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# If train.csv and test.csv don't exist locally or in /content/, we generate them from Keras MNIST\n",
    "if not os.path.exists('train.csv') and not os.path.exists('/content/train.csv'):\n",
    "    print(\"Generating train.csv and test.csv from Keras MNIST dataset for seamless execution...\")\n",
    "    import tensorflow as tf\n",
    "    (X_train_mnist, Y_train_mnist), (X_test_mnist, Y_test_mnist) = tf.keras.datasets.mnist.load_data()\n",
    "    \n",
    "    # train.csv: 42000 rows (labels + pixels)\n",
    "    train_pixels = X_train_mnist[:42000].reshape(42000, 784)\n",
    "    train_labels = Y_train_mnist[:42000]\n",
    "    train_df = pd.DataFrame(train_pixels, columns=[f'pixel{i}' for i in range(784)])\n",
    "    train_df.insert(0, 'label', train_labels)\n",
    "    train_df.to_csv('train.csv', index=False)\n",
    "    print(\"Generated train.csv\")\n",
    "    \n",
    "    # test.csv: 28000 rows (pixels only)\n",
    "    test_pixels_part1 = X_train_mnist[42000:].reshape(18000, 784)\n",
    "    test_pixels_part2 = X_test_mnist[:10000].reshape(10000, 784)\n",
    "    test_pixels = np.vstack((test_pixels_part1, test_pixels_part2))\n",
    "    test_df = pd.DataFrame(test_pixels, columns=[f'pixel{i}' for i in range(784)])\n",
    "    test_df.to_csv('test.csv', index=False)\n",
    "    print(\"Generated test.csv\")\n",
    "\n",
    "if os.path.exists('train.csv'):\n",
    "    valid_data = pd.read_csv('train.csv')\n",
    "    test_data = pd.read_csv('test.csv')\n",
    "else:\n",
    "    valid_data = pd.read_csv('/content/train.csv')\n",
    "    test_data = pd.read_csv('/content/test.csv')\n"
]

# Modify cell 32 (index 32) - Wrap plot_model in try-except to avoid failure if pydot/graphviz is missing
cell_32 = notebook['cells'][32]
print("Original Cell 32 source:", cell_32['source'])
cell_32['source'] = [
    "try:\n",
    "    tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True)\n",
    "except Exception as e:\n",
    "    print(f\"Skipping model plotting: pydot/graphviz not installed. Error: {e}\")\n"
]

# Modify cell 33 (index 33) - Wrap scheduler in float() conversion for Keras 3 compatibility
cell_33 = notebook['cells'][33]
print("Original Cell 33 source:", cell_33['source'])
cell_33['source'] = [
    "### .et the learning rate =\n",
    "lr_rate = 0.0001\n",
    "early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',\n",
    "                                                  patience = 3,\n",
    "                                                  min_delta = 1e-4,\n",
    "                                                  restore_best_weights = True)\n",
    "checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = 'saved_model/best_model_todate.weights.h5',\n",
    "                                                 save_best_only = True,\n",
    "                                                 save_weights_only = True,\n",
    "                                                 monitor='val_accuracy',\n",
    "                                                 mode='max')\n",
    "tn = tf.keras.callbacks.TerminateOnNaN()\n",
    "scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = lr_rate,\n",
    "                                                           decay_steps = steps_per_epoch//4,\n",
    "                                                           decay_rate= 0.80,\n",
    "                                                           staircase=True)\n",
    "lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: float(scheduler(epoch)))\n",
    "lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',\n",
    "                                                  factor = 0.1,\n",
    "                                                  patience = 4,\n",
    "                                                  verbose = 3)"
]

# Modify cell 63 (index 63) - Fix X_test shape and normalization for model.predict
cell_63 = notebook['cells'][63]
print("Original Cell 63 source:", cell_63['source'])
cell_63['source'] = [
    "y_predicted = model.predict(X_test.reshape(-1, 28, 28, 1) / 255.0)\n",
    "y_predicted_labels = [np.argmax(i) for i in y_predicted]\n",
    "print('Predicted Label :',y_predicted_labels[:10])\n",
    "print('Actual Label    :',Y_test[:10])"
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")

# Execute all cells in the notebook programmatically
print("Executing all cells of the notebook...")
try:
    subprocess.run([
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        notebook_path
    ], check=True)
    print("Notebook executed successfully and saved in-place!")
except subprocess.CalledProcessError as e:
    print(f"Error during notebook execution: {e}")
    sys.exit(1)
